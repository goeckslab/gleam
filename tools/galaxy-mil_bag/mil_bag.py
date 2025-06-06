"""
A script for creating bags of instances from embeddings
and metadata for Multiple Instance Learning (MIL) tasks.

Processes embedding and metadata CSV files to generate
bags of instances, saved as a single CSV file. Supports
bagging strategies (by sample, in turns, or random),
pooling methods, and options for balancing, preventing
data leakage, and Ludwig formatting. Handles large
datasets efficiently using temporary Parquet files,
sequential processing, and multiprocessing.

Dependencies:
  - gc: For manual garbage collection to manage memory.
  - argparse: For parsing command-line arguments.
  - logging: For logging progress and errors.
  - multiprocessing (mp): For parallel processing.
  - os: For file operations and temporary file management.
  - tempfile: For creating temporary files.
  - numpy (np): For numerical operations and array.
  - pandas (pd): For data manipulation and I/O (CSV, Parquet).
  - torch: For tensor operations (attention pooling).
  - torch.nn: For NN components (attention pooling).
  - fastparquet: For reading and writing Parquet files.

Key Features:
  - Multiple bagging: by sample (`bag_by_sample`), in
    turns (`bag_in_turns`), or random (`bag_random`).
  - Various pooling methods (e.g., max, mean, attention).
  - Prevents data leakage by splitting at sample level.
  - Balances bags by label imbalance or truncating.
  - Outputs in Ludwig format (whitespace-separated vectors).
  - Efficient large dataset processing (temp Parquet,
    sequential CSV write).
  - GPU acceleration for certain pooling (e.g., attention).

Usage:
  Run the script from the command line with arguments:

  ```bash
  python ludwig_mil_temp.py --embeddings_csv <path_to_embeddings.csv>
    --metadata_csv <path_to_metadata.csv> --bag_size <bag_size>
    --pooling_method <method> --output_csv <output.csv>
    [--split_proportions <train,val,test>] [--dataleak]
    [--balance_enforced] [--by_sample <splits>] [--repeats <num>]
    [--ludwig_format] [--random_seed <seed>]
    [--imbalance_cap <percentage>] [--truncate_bags] [--use_gpu]
"""


import gc
import argparse
import logging
import multiprocessing as mp
import os
import tempfile

import numpy as np

import pandas as pd

import torch
import torch.nn as nn

import fastparquet


def parse_bag_size(bag_size_str):
    """Parses bag size string into a range or single value."""
    try:
        if '-' in bag_size_str:
            start, end = map(int, bag_size_str.split('-'))
            return list(range(start, end + 1))
        return [int(bag_size_str)]
    except ValueError:
        logging.error("Invalid bag_size format: %s", bag_size_str)
        raise


def parse_by_sample(value):
    """Parses by_sample string into a set of split values."""
    try:
        value = str(value)
        splits = [int(x) for x in value.split(",")]
        valid_splits = {0, 1, 2}
        if not all(x in valid_splits for x in splits):
            logging.warning("Invalid splits in by_sample: %s", splits)
            return None
        return splits
    except (ValueError, AttributeError):
        logging.warning("By_Sample not used")
        return None


class BaggingConfig:
    """Configuration class for bagging parameters."""

    def __init__(self, params):
        self.embeddings_csv = params.embeddings_csv
        self.metadata_csv = params.metadata_csv
        self.split_proportions = params.split_proportions
        self.prevent_leakage = params.dataleak
        self.balance_enforced = params.balance_enforced
        self.bag_size = parse_bag_size(params.bag_size)
        self.pooling_method = params.pooling_method
        self.by_sample = parse_by_sample(params.by_sample)
        self.repeats = params.repeats
        self.ludwig_format = params.ludwig_format
        self.output_csv = params.output_csv
        self.random_seed = params.random_seed
        self.imbalance_cap = params.imbalance_cap
        self.truncate_bags = params.truncate_bags
        self.use_gpu = params.use_gpu

    def __str__(self):
        """String representation of the config for logging."""
        return (
            f"embeddings_csv={self.embeddings_csv}, "
            f"metadata_csv={self.metadata_csv}, "
            f"split_proportions={self.split_proportions}, "
            f"prevent_leakage={self.prevent_leakage}, "
            f"balance_enforced={self.balance_enforced}, "
            f"bag_size={self.bag_size}, "
            f"pooling_method={self.pooling_method}, "
            f"by_sample={self.by_sample}, "
            f"repeats={self.repeats}, "
            f"ludwig_format={self.ludwig_format}, "
            f"output_csv={self.output_csv}, "
            f"random_seed={self.random_seed}, "
            f"imbalance_cap={self.imbalance_cap}, "
            f"truncate_bags={self.truncate_bags}, "
            f"use_gpu={self.use_gpu}"
        )


def set_random_seed(configs):
    """Sets random seeds for reproducibility."""
    np.random.seed(configs.random_seed)
    torch.manual_seed(configs.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(configs.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logging.info("Random seed set to %d", configs.random_seed)


def validate_metadata(metadata):
    """Validates metadata for required columns."""
    required_cols = {"sample_name", "label"}
    if not required_cols.issubset(metadata.columns):
        missing = required_cols - set(metadata.columns)
        raise ValueError(f"Metadata missing columns: {missing}")
    return metadata


def load_metadata(file_path):
    """Loads metadata from a CSV file."""
    metadata = pd.read_csv(file_path)
    validate_metadata(metadata)
    logging.info("Metadata loaded with %d samples, cols: %s",
                 len(metadata), list(metadata.columns))
    logging.info("Unique samples: %d, labels: %d",
                 metadata["sample_name"].nunique(),
                 metadata["label"].nunique())
    return metadata


def convert_proportions(proportion_string):
    """Converts a string of split proportions into a list of floats."""
    proportion_list = [float(p) for p in proportion_string.split(",")]
    print(proportion_list)
    if len(proportion_list) == 2:
        proportion_list = [proportion_list[0], 0.0, proportion_list[1]]

    for proportion in proportion_list:
        if proportion < 0 or proportion > 1:
            raise ValueError("Each proportion must be between 0 and 1")

    if abs(sum(proportion_list) - 1.0) > 1e-6:
        raise ValueError("Proportions must sum to approximately 1.0")

    return proportion_list


def calculate_split_counts(total_samples, proportions):
    """Calculates sample counts for each split."""
    counts = [int(p * total_samples) for p in proportions]
    calculated_total = sum(counts)
    if calculated_total < total_samples:
        counts[-1] += total_samples - calculated_total
    elif calculated_total > total_samples:
        counts[0] -= calculated_total - total_samples
    return counts


def assign_split_labels(proportions, sample_count):
    """Assigns split labels based on proportions."""
    proportion_values = convert_proportions(proportions)
    train_fraction, val_fraction, test_fraction = proportion_values

    if val_fraction == 0 and test_fraction == 0:
        labels = np.zeros(sample_count, dtype=int)
    elif val_fraction == 0:
        train_size = int(train_fraction * sample_count)
        test_size = sample_count - train_size
        labels = np.array([0] * train_size + [2] * test_size)
    else:
        split_counts = calculate_split_counts(sample_count, proportion_values)
        labels = np.concatenate([
            np.zeros(split_counts[0], dtype=int),
            np.ones(split_counts[1], dtype=int),
            2 * np.ones(split_counts[2], dtype=int)
        ])
    return labels


def split_dataset(metadata, configs):
    """Splits dataset into train, val, test sets if prevent_leakage is True."""
    if configs.prevent_leakage:
        logging.info("No data leakage allowed")
        unique_samples = metadata["sample_name"].unique()
        sample_count = len(unique_samples)
        split_labels = assign_split_labels(configs.split_proportions,
                                           sample_count)
        shuffled_samples = np.random.permutation(unique_samples)
        label_series = pd.Series(split_labels, index=shuffled_samples)
        metadata["split"] = metadata["sample_name"].map(label_series)
        train_count = (metadata["split"] == 0).sum()
        val_count = (metadata["split"] == 1).sum()
        test_count = (metadata["split"] == 2).sum()
        logging.info("Dataset split: train %d, val %d, test %d",
                     train_count, val_count, test_count)
    else:
        logging.info("Data leakage allowed setup")
    return metadata


def assign_chunk_splits(chunk, split_counts, current_counts):
    """Assigns split labels to a chunk of embeddings."""
    chunk_size = len(chunk)
    remaining = {
        0: split_counts[0] - current_counts[0],
        1: split_counts[1] - current_counts[1],
        2: split_counts[2] - current_counts[2]
    }
    available_splits = [s for s, count in remaining.items() if count > 0]
    if not available_splits:
        return chunk, current_counts

    total_remaining = sum(remaining.values())
    assign_count = min(chunk_size, total_remaining)
    if assign_count == 0:
        return chunk, current_counts

    weights = [remaining[s] / total_remaining for s in available_splits]
    splits = np.random.choice(available_splits, size=assign_count, p=weights)
    chunk["split"] = pd.Series(splits, index=chunk.index[:assign_count])
    chunk["split"] = chunk["split"].fillna(0).astype(int)

    for split in available_splits:
        current_counts[split] += np.sum(splits == split)

    return chunk, current_counts


def setup_temp_files():
    """Sets up temporary Parquet files for splits and bag outputs."""
    splits = [0, 1, 2]
    split_files = {}
    for split in splits:
        fd, path = tempfile.mkstemp(prefix=f"split_{split}_",
                                    suffix=".parquet",
                                    dir=os.getcwd())
        os.close(fd)  # Explicitly close the file descriptor
        split_files[split] = path

    bag_outputs = {}
    for split in splits:
        fd, path = tempfile.mkstemp(prefix=f"MIL_bags_{split}_",
                                    suffix=".parquet",
                                    dir=os.getcwd())
        os.close(fd)  # Explicitly close the file descriptor
        bag_outputs[split] = path

    return split_files, bag_outputs


def distribute_embeddings(configs, metadata, split_files):
    embeddings_path = configs.embeddings_csv
    proportion_string = configs.split_proportions
    prevent_leakage = configs.prevent_leakage

    logging.info("Distributing embeddings from %s to Parquet files",
                 embeddings_path)
    buffer_size = 50000
    merged_header = None
    non_sample_columns = None

    if not prevent_leakage:
        logging.warning(
            "Counting rows in %s; may be slow for large files",
            embeddings_path
        )
        total_rows = sum(1 for _ in open(embeddings_path)) - 1
        proportions = convert_proportions(proportion_string)
        split_counts = calculate_split_counts(total_rows, proportions)
        current_counts = {0: 0, 1: 0, 2: 0}
    else:
        sample_to_split = dict(zip(metadata["sample_name"], metadata["split"]))
        sample_to_label = dict(zip(metadata["sample_name"], metadata["label"]))

    first_write = {split: True for split in split_files}

    try:
        first_header_read = True
        for chunk in pd.read_csv(embeddings_path, chunksize=buffer_size):
            # Modify 'sample_name' to remove part after the last underscore
            chunk['sample_name'] = chunk['sample_name'].apply(lambda x: x.rsplit('_', 1)[0])

            if first_header_read:
                orig_header = list(chunk.columns)
                non_sample_columns = [
                    col for col in orig_header if col != "sample_name"
                ]
                merged_header = ["sample_name", "label"] + non_sample_columns
                logging.info("Merged header: %s", merged_header)
                first_header_read = False

            if prevent_leakage:
                chunk["split"] = chunk["sample_name"].map(sample_to_split)
                chunk["label"] = chunk["sample_name"].map(sample_to_label)
            else:
                chunk, current_counts = assign_chunk_splits(chunk,
                                                            split_counts,
                                                            current_counts)
                chunk = chunk.merge(metadata[["sample_name", "label"]],
                                    on="sample_name",
                                    how="left")

            chunk = chunk.dropna(subset=["split", "label"])
            for split in split_files:
                split_chunk = chunk[chunk["split"] == split]
                if not split_chunk.empty:
                    temp_file = split_files[split]
                    split_chunk[merged_header].to_parquet(
                        temp_file,
                        engine="fastparquet",
                        append=not first_write[split],
                        index=False
                    )
                    first_write[split] = False
            del chunk
            gc.collect()

    except Exception as e:
        logging.error("Error distributing embeddings to Parquet: %s", e)
        raise


def aggregate_embeddings(embeddings, pooling_method, use_gpu=False):
    # Convert embeddings to a float32 array explicitly.
    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    elif embeddings.ndim == 0:
        embeddings = embeddings.reshape(1, 1)

    logging.debug("Aggregating embeddings with shape: %s", embeddings.shape)

    if pooling_method == "max_pooling":
        result = np.max(embeddings, axis=0)
    elif pooling_method == "mean_pooling":
        result = np.mean(embeddings, axis=0)
    elif pooling_method == "sum_pooling":
        result = np.sum(embeddings, axis=0)
    elif pooling_method == "min_pooling":
        result = np.min(embeddings, axis=0)
    elif pooling_method == "median_pooling":
        result = np.median(embeddings, axis=0)
    elif pooling_method == "l2_norm_pooling":
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        if norm.any():
            result = np.mean(embeddings / (norm + 1e-8), axis=0)
        else:
            result = np.mean(embeddings, axis=0)
    elif pooling_method == "geometric_mean_pooling":
        clipped = np.clip(embeddings, 1e-10, None)
        result = np.exp(np.mean(np.log(clipped), axis=0))
    elif pooling_method == "first_embedding":
        result = embeddings[0]
    elif pooling_method == "last_embedding":
        result = embeddings[-1]
    elif pooling_method == "attention_pooling":
        device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)
        with torch.no_grad():
            linear = nn.Linear(tensor.shape[1], 1).to(device)
            weights = nn.Softmax(dim=0)(linear(tensor))
            result = torch.sum(weights * tensor, dim=0).cpu().detach().numpy()
    else:
        raise ValueError(f"Unknown pooling method: {pooling_method}")

    logging.debug("Aggregated embedding shape: %s", result.shape)
    return result


def bag_by_sample(df, split, bag_file, config, batch_size=1000,
                  fixed_target_bags=None):
    """
    Processes the provided DataFrame by grouping rows by sample,
    constructs bags from each sample group using the configured bag_size,
    and writes the bag rows directly to bag_file (a Parquet file) in batches.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        split (str): The split identifier (e.g., 'train', 'val').
        bag_file (str): The path to the Parquet file to write the bags.
        config (object): Configuration object with bag_size, pooling_method...
        batch_size (int, optional): The number of rows to write in each batch.
        fixed_target_bags (tuple, optional): (target_label, num_bags)
        to generate bags only for target_label.

    Output row format:
        sample_name, bag_label, split, bag_size, vector_0, vector_1, vector_N
    """
    log_msg = f"Processing by sample for split: {split}"
    if fixed_target_bags:
        log_msg += f" with fixed target {fixed_target_bags}"
    logging.info(log_msg)

    batch_rows = []
    bag_count = 0
    vector_columns = [
        col for col in df.columns
        if col not in ["sample_name", "label", "split"]
    ]

    if fixed_target_bags is not None:
        target_label, target_needed = fixed_target_bags
        target_samples = list(
            df[df["label"] == target_label]["sample_name"].unique()
        )
        df = df[df["sample_name"].isin(target_samples)]

        if df.empty:
            logging.warning(
                "No samples available for target label %d in split %s",
                target_label,
                split
            )
            return

        available_samples = target_samples.copy()
        np.random.shuffle(available_samples)

        while bag_count < target_needed:
            if len(available_samples) == 0:
                available_samples = target_samples.copy()
                np.random.shuffle(available_samples)
                logging.info(
                    "Reusing samples for target label %d in split %s",
                    target_label,
                    split
                )

            sample_name = available_samples.pop()
            group = df[df["sample_name"] == sample_name]
            embeddings = group[vector_columns].values
            num_instances = len(group)

            current_bag_size = config.bag_size[0] \
                if len(config.bag_size) == 1 else \
                np.random.randint(config.bag_size[0], config.bag_size[1] + 1)
            current_bag_size = min(current_bag_size, num_instances)

            selected = group.sample(n=current_bag_size, replace=True)
            bag_embeddings = selected[vector_columns].values

            aggregated_embedding = aggregate_embeddings(
                bag_embeddings,
                config.pooling_method,
                config.use_gpu
            )

            bag_label = int(any(selected["label"] == 1))
            if bag_label != target_label:
                logging.warning(
                    "Generated bag for target %d but got label %d",
                    target_label, bag_label
                )
                continue

            row = {
                "sample_name": sample_name,
                "bag_label": bag_label,
                "split": split,
                "bag_size": current_bag_size
            }
            for j, val in enumerate(aggregated_embedding):
                row[f"vector_{j}"] = val

            batch_rows.append(row)
            bag_count += 1

            if len(batch_rows) >= batch_size:
                df_batch = pd.DataFrame(batch_rows)
                # Check if the file has data to determine append mode
                append_mode = os.path.getsize(bag_file) > 0
                df_batch.to_parquet(
                    bag_file,
                    engine="fastparquet",
                    append=append_mode,
                    index=False
                )
                logging.debug(
                    "Fixed mode: Wrote batch of %d rows to %s",
                    len(batch_rows),
                    bag_file
                )
                batch_rows = []
                del df_batch
                gc.collect()

    else:
        # Standard mode: process all samples
        groups = df.groupby("sample_name")
        for sample_name, group in groups:
            embeddings = group[vector_columns].values
            labels = group["label"].values
            num_instances = len(group)

            current_bag_size = config.bag_size[0] \
                if len(config.bag_size) == 1 else \
                np.random.randint(
                config.bag_size[0],
                config.bag_size[1] + 1
            )
            num_bags = (
                num_instances + current_bag_size - 1
            ) // current_bag_size
            logging.info(
                "Sample %s: %d instances, creating %d bags (bag size %d)",
                sample_name,
                num_instances,
                num_bags,
                current_bag_size
            )

            for i in range(num_bags):
                start_idx = i * current_bag_size
                end_idx = min(start_idx + current_bag_size, num_instances)
                bag_embeddings = embeddings[start_idx:end_idx]
                bag_labels = labels[start_idx:end_idx]

                aggregated_embedding = aggregate_embeddings(
                    bag_embeddings,
                    config.pooling_method,
                    config.use_gpu
                )
                bag_label = int(any(bag_labels == 1))

                row = {
                    "sample_name": sample_name,
                    "bag_label": bag_label,
                    "split": split,
                    "bag_size": end_idx - start_idx
                }
                for j, val in enumerate(aggregated_embedding):
                    row[f"vector_{j}"] = val

                batch_rows.append(row)
                bag_count += 1

                if len(batch_rows) >= batch_size:
                    df_batch = pd.DataFrame(batch_rows)
                    # Check if the file has data to determine append mode
                    append_mode = os.path.getsize(bag_file) > 0
                    df_batch.to_parquet(
                        bag_file,
                        engine="fastparquet",
                        append=append_mode,
                        index=False
                    )
                    logging.debug(
                        "Wrote batch of %d rows to %s",
                        len(batch_rows),
                        bag_file
                    )
                    batch_rows = []
                    del df_batch
                    gc.collect()

    # Write any remaining rows
    if batch_rows:
        df_batch = pd.DataFrame(batch_rows)
        append_mode = os.path.getsize(bag_file) > 0
        df_batch.to_parquet(
            bag_file,
            engine="fastparquet",
            append=append_mode,
            index=False
        )
        logging.debug(
            "Wrote final batch of %d rows to %s",
            len(batch_rows),
            bag_file
        )
        del df_batch
        gc.collect()

    logging.info("Created %d bags for split: %s", bag_count, split)


def bag_in_turns(df, split, bag_file, config, batch_size=500,
                 fixed_target_bags=None, allow_reuse=True):
    """
    Generate bags of instances from a DataFrame, with optional
    fixed-target mode, data reuse, and enhanced diversity.

    Parameters:
    - df (pd.DataFrame): Input DataFrame with columns including
      'sample_name', 'label', 'split', and embedding vectors.
    - split (str): Dataset split (e.g., 'train', 'test').
    - bag_file (str): Path to save the output Parquet file.
    - config (object): Configuration object with attributes
      'bag_size', 'pooling_method', and 'use_gpu'.
    - batch_size (int): Number of bags to process before writing
      to file (default: 500).
    - fixed_target_bags (tuple): Optional (label, num_bags) to
      generate bags for a specific label (e.g., (0, 100)).
    - allow_reuse (bool): Allow resampling instances with
      replacement if True (default: True).

    Returns:
    - None: Saves bags to the specified Parquet file.
    """
    logging.info(
        "Processing bag in turns for split %s%s",
        split,
        (" with fixed target " + str(fixed_target_bags))
        if fixed_target_bags is not None else ""
    )

    # Identify embedding columns (exclude non-vector columns).
    vector_columns = [
        col for col in df.columns
        if col not in ["sample_name", "label", "split"]
    ]

    # Convert the DataFrame to a NumPy array for faster processing.
    df_np = df.to_numpy()

    # Determine bag size range from config.
    if len(config.bag_size) == 1:
        bag_min = bag_max = config.bag_size[0]
    else:
        bag_min, bag_max = config.bag_size

    batch_rows = []
    bag_count = 0

    if fixed_target_bags is not None:
        # Fixed-target mode: generate bags for a specific label.
        target, target_needed = fixed_target_bags  # e.g., (0, 100)
        if target == 0:
            # Optimize for target label 0: remove all label 1 instances
            indices = np.where(df_np[:, 1] == 0)[0]
            logging.info(
                "Fixed mode: target label 0, using only label 0 instances, \
                total available %d rows",
                len(indices)
            )
        else:
            # For target label 1, use all instances to allow mixing
            indices = np.arange(len(df_np))
            logging.info(
                "Fixed mode: target label 1, using all instances, \
                total available %d rows",
                len(indices)
            )

        total_available = len(indices)

        while bag_count < target_needed:
            current_bag_size = np.random.randint(bag_min, bag_max + 1) \
                if bag_min != bag_max else bag_min

            if total_available < current_bag_size and not allow_reuse:
                logging.warning(
                    "Not enough instances (%d) for bag size %d and \
                    target label %d",
                    total_available, current_bag_size, target
                )
                break

            # Sample instances
            selected = np.random.choice(
                indices,
                size=current_bag_size,
                replace=allow_reuse
            )
            bag_data = df_np[selected]

            if target == 1:
                # For positive bags, ensure at least one instance has label 1
                if not np.any(bag_data[:, 1] == 1):
                    continue  # Skip if no positive instance
                bag_label = 1
            else:
                # For negative bags, all instances are label 0 due to filtering
                bag_label = 0

            # Aggregate embeddings.
            vec_col_indices = [
                df.columns.get_loc(col) for col in vector_columns
            ]
            embeddings = bag_data[:, vec_col_indices].astype(np.float32)
            aggregated_embedding = aggregate_embeddings(
                embeddings,
                config.pooling_method,
                config.use_gpu
            )

            # Set bag metadata.
            bsize = bag_data.shape[0]
            samples = np.unique(bag_data[:, 0])
            merged_sample_name = ",".join(map(str, samples))

            # Create row for the bag.
            row = {
                "sample_name": merged_sample_name,
                "bag_label": bag_label,
                "split": split,
                "bag_size": bsize
            }
            for j, val in enumerate(aggregated_embedding):
                row[f"vector_{j}"] = val

            batch_rows.append(row)
            bag_count += 1

            if len(batch_rows) >= batch_size:
                df_batch = pd.DataFrame(batch_rows)
                df_batch.to_parquet(
                    bag_file,
                    engine="fastparquet",
                    append=True,
                    index=False
                )
                logging.debug(
                    "Fixed mode: Wrote a batch of %d rows to %s",
                    len(batch_rows),
                    bag_file
                )
                batch_rows = []
                del df_batch
                gc.collect()

        # Write any remaining rows.
        if batch_rows:
            df_batch = pd.DataFrame(batch_rows)
            df_batch.to_parquet(
                bag_file,
                engine="fastparquet",
                append=True,
                index=False
            )
            logging.debug(
                "Wrote the final batch of %d rows to %s",
                len(batch_rows),
                bag_file
            )
            del df_batch
            gc.collect()

        logging.info("Created %d bags for split: %s", bag_count, split)

    else:
        # Alternating mode: alternate between labels 0 and 1.
        indices_0 = np.where(df_np[:, 1] == 0)[0]
        indices_1 = np.where(df_np[:, 1] == 1)[0]
        np.random.shuffle(indices_0)
        np.random.shuffle(indices_1)
        turn = 0  # 0: label 0, 1: label 1.

        while len(indices_0) > 0 or len(indices_1) > 0:
            current_bag_size = np.random.randint(bag_min, bag_max + 1) \
                if bag_min != bag_max else bag_min

            if turn == 0:
                if len(indices_0) > 0:
                    num_to_select = min(current_bag_size, len(indices_0))
                    selected = indices_0[:num_to_select]
                    indices_0 = indices_0[num_to_select:]
                else:
                    if len(indices_1) == 0:
                        break
                    num_to_select = min(current_bag_size, len(indices_1))
                    selected = indices_1[:num_to_select]
                    indices_1 = indices_1[num_to_select:]
            else:
                if len(indices_1) > 0:
                    num_to_select = min(current_bag_size, len(indices_1))
                    selected = indices_1[:num_to_select]
                    indices_1 = indices_1[num_to_select:]
                else:
                    if len(indices_0) == 0:
                        break
                    num_to_select = min(current_bag_size, len(indices_0))
                    selected = indices_0[:num_to_select]
                    indices_0 = indices_0[num_to_select:]

            bag_data = df_np[selected]
            if bag_data.shape[0] == 0:
                break

            # Aggregate embeddings.
            vec_col_indices = [
                df.columns.get_loc(col) for col in vector_columns
            ]
            embeddings = bag_data[:, vec_col_indices].astype(np.float32)
            aggregated_embedding = aggregate_embeddings(
                embeddings,
                config.pooling_method,
                config.use_gpu
            )

            # Set bag label and metadata.
            bag_label = int(np.any(bag_data[:, 1] == 1))
            bsize = bag_data.shape[0]
            samples = np.unique(bag_data[:, 0])
            merged_sample_name = ",".join(map(str, samples))

            # Create row for the bag.
            row = {
                "sample_name": merged_sample_name,
                "bag_label": bag_label,
                "split": split,
                "bag_size": bsize
            }
            for j, val in enumerate(aggregated_embedding):
                row[f"vector_{j}"] = val

            batch_rows.append(row)
            bag_count += 1
            turn = 1 - turn

            # Write batch to file if batch_size is reached.
            if len(batch_rows) >= batch_size:
                df_batch = pd.DataFrame(batch_rows)
                df_batch.to_parquet(
                    bag_file,
                    engine="fastparquet",
                    append=(bag_count > len(batch_rows)),
                    index=False
                )
                logging.debug(
                    "Alternating mode: Wrote a batch of %d rows to %s",
                    len(batch_rows),
                    bag_file
                )
                batch_rows = []
                del df_batch
                gc.collect()

        # Write any remaining rows.
        if batch_rows:
            df_batch = pd.DataFrame(batch_rows)
            df_batch.to_parquet(
                bag_file,
                engine="fastparquet",
                append=(bag_count > len(batch_rows)),
                index=False
            )
            logging.debug(
                "Wrote the final batch of %d rows to %s",
                len(batch_rows),
                bag_file
            )
            del df_batch
            gc.collect()

        logging.info("Created %d bags for split: %s", bag_count, split)


def bag_random(df, split, bag_file, configs, batch_size=500):
    """
    Processes the provided DataFrame by randomly selecting instances
    to create bags.
    """
    logging.info("Processing bag randomly for split %s", split)

    # Identify vector columns (exclude non-vector columns).
    vector_columns = [
        col for col in df.columns
        if col not in ["sample_name", "label", "split"]
    ]

    df_np = df.to_numpy()

    # Create an array of all row indices and shuffle them.
    indices = np.arange(df.shape[0])
    np.random.shuffle(indices)

    bag_count = 0
    batch_rows = []

    # Determine bag size parameters.
    if len(configs.bag_size) == 1:
        bag_min = bag_max = configs.bag_size[0]
    else:
        bag_min, bag_max = configs.bag_size

    pos = 0
    total_rows = len(indices)

    # Process until all indices have been used.
    while pos < total_rows:
        # Ensuring we do not exceed remaining rows.
        current_bag_size = (np.random.randint(bag_min, bag_max + 1)
                            if bag_min != bag_max else bag_min)
        current_bag_size = min(current_bag_size, total_rows - pos)

        # Select the indices for this bag.
        selected = indices[pos: pos + current_bag_size]
        pos += current_bag_size

        # Extract the bag data.
        bag_data = df_np[selected]
        if bag_data.shape[0] == 0:
            break

        # Identify the positions of the vector columns using the column names.
        vec_col_indices = [df.columns.get_loc(col) for col in vector_columns]
        embeddings = bag_data[:, vec_col_indices].astype(np.float32)
        aggregated_embedding = aggregate_embeddings(
            embeddings,
            configs.pooling_method,
            configs.use_gpu
        )

        # Determine bag_label: 1 if any instance in this bag has label == 1.
        bag_label = int(np.any(bag_data[:, 1] == 1))

        # Merge all sample names from the bag (unique names, comma-separated).
        samples = np.unique(bag_data[:, 0])
        merged_sample_name = ",".join(map(str, samples))

        # Use the provided split value.
        bag_split = split
        bsize = bag_data.shape[0]

        # Build the output row with header fields:
        # sample_name, bag_label, split, bag_size, then embeddings.
        row = {
            "sample_name": merged_sample_name,
            "bag_label": bag_label,
            "split": bag_split,
            "bag_size": bsize
        }
        for j, val in enumerate(aggregated_embedding):
            row[f"vector_{j}"] = val

        batch_rows.append(row)
        bag_count += 1

        # Write out rows in batches.
        if len(batch_rows) >= batch_size:
            df_batch = pd.DataFrame(batch_rows)
            # For the first batch,
            # append=False (header written),
            # then append=True on subsequent batches.
            df_batch.to_parquet(
                bag_file,
                engine="fastparquet",
                append=(bag_count > len(batch_rows)),
                index=False
            )
            logging.debug(
                "Wrote a batch of %d rows to %s",
                len(batch_rows),
                bag_file
            )
            batch_rows = []
            del df_batch
            gc.collect()

    # Write any remaining rows.
    if batch_rows:
        df_batch = pd.DataFrame(batch_rows)
        df_batch.to_parquet(
            bag_file,
            engine="fastparquet",
            append=(bag_count > len(batch_rows)),
            index=False
        )
        logging.debug(
            "Wrote the final batch of %d rows to %s",
            len(batch_rows),
            bag_file
        )
        del df_batch
        gc.collect()

    logging.info("Created %d bags for split: %s", bag_count, split)


def imbalance_adjustment(bag_file, split, configs, df):
    """
    Verifies if the number of bags per label in bag_file is
    within imbalance_cap.
    If not, generates additional bags for the minority label.

    Args:
        bag_file (str): Path to the Parquet file containing bags.
        split (str): The current split (e.g., 'train', 'val').
        config (object): Configuration with imbalance_cap, by_sample, etc.
        df (pd.DataFrame): Original DataFrame for generating additional bags.
    """
    # Read the bag file and count bags per label
    bags_df = pd.read_parquet(bag_file)
    n0 = (bags_df["bag_label"] == 0).sum()
    n1 = (bags_df["bag_label"] == 1).sum()
    total = n0 + n1

    if total == 0:
        logging.warning("No bags found in %s for split %s", bag_file, split)
        return

    # Calculate imbalance as a percentage
    imbalance = abs(n0 - n1) / total * 100
    logging.info(
        "Split %s: %d bags (label 0: %d, label 1: %d), imbalance %.2f%%",
        split, total, n0, n1, imbalance
    )

    if imbalance > configs.imbalance_cap:
        # Identify minority label
        min_label = 0 if n0 < n1 else 1
        n_min = n0 if min_label == 0 else n1
        n_maj = n1 if min_label == 0 else n0

        # Calculate how many bags are needed to balance (aim for equality)
        num_needed = n_maj - n_min
        logging.info(
            "Imbalance %.2f%% exceeds cap %.2f%% in split %s, \
            need %d bags for label %d",
            imbalance,
            configs.imbalance_cap,
            split,
            num_needed,
            min_label
        )

        # Generate additional bags based on the bag creation method
        if split in configs.by_sample:
            bag_by_sample(
                df,
                split,
                bag_file,
                configs,
                fixed_target_bags=(min_label, num_needed)
            )
        else:
            bag_in_turns(
                df,
                split,
                bag_file,
                configs,
                fixed_target_bags=(min_label, num_needed)
            )

        # Verify the new balance (optional, for logging)
        updated_bags_df = pd.read_parquet(bag_file)
        new_n0 = (updated_bags_df["bag_label"] == 0).sum()
        new_n1 = (updated_bags_df["bag_label"] == 1).sum()
        new_total = new_n0 + new_n1
        new_imbalance = abs(new_n0 - new_n1) / new_total * 100
        logging.info(
            "After adjustment, split %s: %d bags (label 0: %d, label 1: %d), \
            imbalance %.2f%%",
            split,
            new_total,
            new_n0,
            new_n1,
            new_imbalance
        )
    else:
        logging.info(
            "Imbalance %.2f%% within cap %.2f%% for split %s, \
            no adjustment needed",
            imbalance,
            configs.imbalance_cap,
            split
        )


def truncate_bag(bag_file, split):
    """
    Truncates the bags in the bag_file to balance the counts of label 0
    and label 1,
    ensuring that the file is never left empty (at least one bag remains).

    Args:
        bag_file (str): Path to the Parquet file containing the bags.
        split (str): The current split (e.g., 'train', 'val')
        for logging purposes.

    Returns:
        None: Overwrites the bag_file with the truncated bags,
        ensuring at least one bag remains.
    """
    logging.info("Truncating bags for split %s in file: %s", split, bag_file)

    # Step 1: Read the bag file to get the total number of bags
    try:
        bags_df = pd.read_parquet(bag_file)
    except Exception as e:
        logging.error("Failed to read bag file %s: %s", bag_file, e)
        return

    total_bags = len(bags_df)
    if total_bags == 0:
        logging.warning("No bags found in %s for split %s", bag_file, split)
        return

    # Step 2: Count bags with label 0 and label 1
    n0 = (bags_df["bag_label"] == 0).sum()
    n1 = (bags_df["bag_label"] == 1).sum()
    logging.info(
        "Split %s: Total bags %d (label 0: %d, label 1: %d)",
        split,
        total_bags,
        n0,
        n1
    )

    # Determine the minority count and majority label
    min_count = min(n0, n1)
    majority_label = 0 if n0 > n1 else 1

    if n0 == n1:
        logging.info(
            "Bags already balanced for split %s, no truncation needed",
            split
        )
        return

    # Step 3: Adjust min_count to ensure at least one bag remains
    if min_count == 0:
        logging.warning(
            "Minority label has 0 bags in split %s, keeping 1 bag from \
            majority label %d to avoid empty file",
            split,
            majority_label
        )
        min_count = 1  # Ensure at least one bag is kept

    # Step 4: Truncate excess bags from the majority label
    logging.info(
        "Truncating %d bags from label %d to match %d bags per label",
        max(0, (n0 if majority_label == 0 else n1) - min_count),
        majority_label,
        min_count
    )

    # Shuffle the majority label bags to randomly select which to keep
    majority_bags = bags_df[
        bags_df["bag_label"] == majority_label
    ].sample(frac=1, random_state=None)

    minority_bags = bags_df[bags_df["bag_label"] != majority_label]

    # Keep only min_count bags from the majority label
    majority_bags_truncated = majority_bags.iloc[:min_count]

    # Combine the truncated majority and minority bags
    truncated_bags_df = pd.concat(
        [majority_bags_truncated,
         minority_bags],
        ignore_index=True
    )

    # Verify that the resulting DataFrame is not empty
    if len(truncated_bags_df) == 0:
        logging.error(
            "Unexpected empty DataFrame after truncation for split %s, \
            this should not happen",
            split
        )
        return

    # Step 5: Overwrite the bag file with the truncated bags
    try:
        truncated_bags_df.to_parquet(
            bag_file,
            engine="fastparquet",
            index=False
        )
        logging.info(
            "Overwrote %s with %d balanced bags (label 0: %d, label 1: %d)",
            bag_file,
            len(truncated_bags_df),
            (truncated_bags_df["bag_label"] == 0).sum(),
            (truncated_bags_df["bag_label"] == 1).sum()
        )
    except Exception as e:
        logging.error("Failed to overwrite bag file %s: %s", bag_file, e)


def columns_into_string(bag_file):
    """
    Reads the bag file (Parquet) from the given path, identifies
    the vector columns (i.e., columns not among 'sample_name', 'bag_label', 'split',
    and 'bag_size'), concatenates these vector values (as strings) into a single
    whitespace-separated string wrapped in double quotes, stored in a new column
    "embeddings", drops the individual vector columns, and writes the modified
    DataFrame back to the same Parquet file.

    The final output format is:
      "sample_name", "bag_label", "split", "bag_size", "embeddings"
    where "embeddings" is a string like: "0.1 0.2 0.3"
    """
    logging.info(
        "Converting vector columns into string for bag file: %s",
        bag_file
    )

    try:
        df = pd.read_parquet(bag_file, engine="fastparquet")
    except Exception as e:
        logging.error("Error reading bag file %s: %s", bag_file, e)
        return

    # Define non-vector columns.
    non_vector = ["sample_name", "bag_label", "split", "bag_size"]

    # Identify vector columns.
    vector_columns = [col for col in df.columns if col not in non_vector]
    logging.info("Identified vector columns: %s", vector_columns)

    # Create new 'embeddings' column by converting vector columns to str,
    # joining them with whitespace, and wrapping the result in double quotes.
    # Use apply() to ensure the result is a Series with one string per row.
    df["embeddings"] = df[vector_columns].astype(str).apply(
        lambda x: " ".join(x), axis=1
    )
    # Drop the original vector columns.
    df.drop(columns=vector_columns, inplace=True)

    try:
        # Write the modified DataFrame back to the same bag file.
        df.to_parquet(bag_file, engine="fastparquet", index=False)
        logging.info(
            "Conversion complete. Final columns: %s",
            df.columns.tolist()
        )
    except Exception as e:
        logging.error("Error writing updated bag file %s: %s", bag_file, e)


def processing_bag(configs, bag_file, temp_file, split):
    """
    Processes a single split and writes bag results
    directly to the bag output Parquet file.
    """
    logging.info("Processing split %s using file: %s", split, temp_file)
    df = pd.read_parquet(temp_file, engine="fastparquet")

    if configs.by_sample is not None and split in configs.by_sample:
        bag_by_sample(df, split, bag_file, configs)
    elif configs.balance_enforced:
        bag_in_turns(df, split, bag_file, configs)
    else:
        bag_random(df, split, bag_file, configs)

    # Free df if imbalance_adjustment is not needed
    if configs.imbalance_cap is None:
        del df
        gc.collect()

    if configs.imbalance_cap is not None:
        imbalance_adjustment(bag_file, split, configs, df)
        del df
        gc.collect()
    elif configs.truncate_bags:
        truncate_bag(bag_file, split)

    if configs.ludwig_format:
        columns_into_string(bag_file)

    return bag_file


def write_final_csv(output_csv, bag_file_paths):
    """
    Merges all Parquet files into a single CSV file,
    processing one file at a time to minimize memory usage.

    Args:
        output_csv (str): Path to the output CSV file specified
        in config.output_csv.
        bag_file_paths (list): List of paths to the Parquet files
        for each split.

    Returns:
        str: Path to the output CSV file.
    """
    logging.info("Merging Parquet files into final CSV: %s", output_csv)

    first_file = True  # Flag to determine if we need to write the header
    total_rows_written = 0

    # Process each Parquet file sequentially
    for bag_file in bag_file_paths:
        try:
            # Skip empty or invalid files
            if os.path.getsize(bag_file) == 0:
                logging.warning(
                    "Parquet file %s is empty (zero size), skipping",
                    bag_file
                )
                continue

            # Load the Parquet file into a DataFrame
            df = pd.read_parquet(bag_file, engine="fastparquet")
            if df.empty:
                logging.warning("Parquet file %s is empty, skipping", bag_file)
                continue

            logging.info("Loaded %d rows from Parquet file: %s, columns: %s",
                         len(df), bag_file, list(df.columns))

            # Write the DataFrame to the CSV file
            # - For the first file, write with header (mode='w')
            # - For subsequent files, append without header (mode='a')
            mode = 'w' if first_file else 'a'
            header = first_file  # Write header only for the first file
            df.to_csv(output_csv, mode=mode, header=header, index=False)
            total_rows_written += len(df)

            logging.info(
                "Wrote %d rows from %s to CSV, total rows written: %d",
                len(df), bag_file, total_rows_written
            )

            # Clear memory
            del df
            gc.collect()

            first_file = False

        except Exception as e:
            logging.error("Failed to process Parquet file %s: %s", bag_file, e)
            continue

    # Check if any rows were written
    if total_rows_written == 0:
        logging.error(
            "No valid data loaded from Parquet files, cannot create CSV"
        )
        raise ValueError("No data available to write to CSV")

    logging.info(
        "Successfully wrote %d rows to final CSV: %s",
        total_rows_written,
        output_csv
    )
    return output_csv


def process_splits(configs, embedding_files, bag_files):
    """Processes splits in parallel and returns all bags."""
    splits = [0, 1, 2]  # Consistent with setup_temp_files()

    # Filter non-empty split files
    valid_info = []
    for split in splits:
        temp_file = embedding_files[split]
        bag_file = bag_files[split]
        if os.path.getsize(temp_file) > 0:  # Check if file has content
            valid_info.append((configs, bag_file, temp_file, split))
        else:
            logging.info("Skipping empty split file: %s", temp_file)

    if not valid_info:
        logging.warning("No non-empty split files to process")
        return []

    # Process splits in parallel and collect bag file paths
    bag_file_paths = []
    with mp.Pool(processes=mp.cpu_count()) as pool:
        logging.info("Starting multiprocessing")
        bag_file_paths = pool.starmap(processing_bag, valid_info)
        logging.info("Multiprocessing is done")

    # Write the final CSV by merging the Parquet files
    output_file = write_final_csv(configs.output_csv, bag_file_paths)
    return output_file


def cleanup_temp_files(split_files, bag_outputs):
    """Cleans up temporary Parquet files."""
    for temp_file in split_files.values():
        try:
            os.remove(temp_file)
            logging.info("Cleaned up temp file: %s", temp_file)
        except Exception as e:
            logging.error("Error removing %s: %s", temp_file, e)
    for bag_output in bag_outputs.values():
        try:
            os.remove(bag_output)
            logging.info("Cleaned up temp bag file: %s", bag_output)
        except Exception as e:
            logging.error("Error removing %s: %s", bag_output, e)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(
        description="Create bags from embeddings and metadata"
    )
    parser.add_argument(
        "--embeddings_csv", type=str, required=True,
        help="Path to embeddings CSV"
    )
    parser.add_argument(
        "--metadata_csv", type=str, required=True,
        help="Path to metadata CSV"
    )
    parser.add_argument(
        "--split_proportions", type=str, default='0.7,0.1,0.2',
        help="Proportions for train, val, test splits"
    )
    parser.add_argument(
        "--dataleak", action="store_true",
        help="Prevents data leakage"
    )
    parser.add_argument(
        "--balance_enforced", action="store_true",
        help="Enforce balanced bagging"
    )
    parser.add_argument(
        "--bag_size", type=str, required=True,
        help="Bag size (e.g., '4' or '3-5')"
    )
    parser.add_argument(
        "--pooling_method", type=str, required=True,
        help="Pooling method"
    )
    parser.add_argument(
        "--by_sample", type=str, default=None,
        help="Splits to bag by sample"
    )
    parser.add_argument(
        "--repeats", type=int, default=1,
        help="Number of bagging repeats"
    )
    parser.add_argument(
        "--ludwig_format", action="store_true",
        help="Output in Ludwig format"
    )
    parser.add_argument(
        "--output_csv", type=str, required=True,
        help="Path to output CSV"
    )
    parser.add_argument(
        "--random_seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--imbalance_cap", type=int, default=None,
        help="Max imbalance percentage"
    )
    parser.add_argument(
        "--truncate_bags", action="store_true",
        help="Truncate bags for balance"
    )
    parser.add_argument(
        "--use_gpu", action="store_true",
        help="Use GPU for pooling"
    )
    args = parser.parse_args()

    config = BaggingConfig(args)
    logging.info("Starting bagging with args: %s", config)

    set_random_seed(config)

    metadata_csv = load_metadata(config.metadata_csv)
    if config.prevent_leakage:
        metadata_csv = split_dataset(metadata_csv, config)

    split_temp_files, split_bag_outputs = setup_temp_files()

    try:
        logging.info("Writing embeddings to split temp Parquet files")
        distribute_embeddings(config, metadata_csv, split_temp_files)

        logging.info("Processing embeddings for each split")
        bags = process_splits(config, split_temp_files, split_bag_outputs)
        logging.info("Bags processed. File generated: %s", bags)

    finally:
        cleanup_temp_files(split_temp_files, split_bag_outputs)
