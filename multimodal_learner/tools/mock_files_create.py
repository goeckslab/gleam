import argparse
import os
import random
import shutil
import tempfile
import zipfile
from typing import List, Optional

import numpy as np
import pandas as pd
from PIL import Image


def generate_mock_data(
    n_samples: int,
    n_features: int,
    problem_type: str,
    with_images: bool,
    image_dir: str,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate mock data for training or testing.

    Args:
        n_samples: Number of samples to generate.
        n_features: Number of numerical features.
        problem_type: 'binary', 'multiclass', or 'regression'.
        with_images: Whether to include image paths.
        image_dir: Directory to save images if with_images is True.
        random_seed: Seed for reproducibility.

    Returns:
        DataFrame with features, target, and optional image_path.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Generate numerical features
    features = {f"feature_{i}": np.random.randn(n_samples) for i in range(n_features)}

    # Generate target based on problem_type
    if problem_type == "binary":
        target = np.random.randint(0, 2, n_samples)
    elif problem_type == "multiclass":
        target = np.random.randint(0, 3, n_samples)  # 3 classes
    elif problem_type == "regression":
        target = np.random.randn(n_samples) * 10 + 5  # Mean 5, std 10
    else:
        raise ValueError("Invalid problem_type. Choose 'binary', 'multiclass', or 'regression'.")

    df = pd.DataFrame(features)
    df["target"] = target

    if with_images:
        os.makedirs(image_dir, exist_ok=True)
        image_paths = []
        for i in range(n_samples):
            # Create dummy image: solid color based on target for visualization
            if problem_type == "regression":
                color = (int(min(max(target[i] * 5, 0), 255)), 0, 0)  # Red intensity based on value
            else:
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            img = Image.new("RGB", (100, 100), color=color)
            img_path = os.path.join(image_dir, f"image_{i}.png")
            img.save(img_path)
            image_paths.append(f"image_{i}.png")  # Relative path
        df["image_path"] = image_paths

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Generate mock input files for AutoGluon Multimodal Learner tool."
    )
    parser.add_argument(
        "--n_train",
        type=int,
        default=100,
        help="Number of training samples (default: 100).",
    )
    parser.add_argument(
        "--n_test",
        type=int,
        default=50,
        help="Number of test samples (default: 50).",
    )
    parser.add_argument(
        "--n_features",
        type=int,
        default=5,
        help="Number of numerical features (default: 5).",
    )
    parser.add_argument(
        "--problem_type",
        type=str,
        default="binary",
        choices=["binary", "multiclass", "regression"],
        help="Problem type: binary, multiclass, or regression (default: binary).",
    )
    parser.add_argument(
        "--with_images",
        action="store_true",
        help="Include images and image_path column (default: False).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="mock_data",
        help="Output directory for generated files (default: mock_data).",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Temporary directory for images if needed
    image_dir = None
    if args.with_images:
        image_dir = tempfile.mkdtemp(dir=args.output_dir)

    # Generate train and test data
    df_train = generate_mock_data(
        args.n_train,
        args.n_features,
        args.problem_type,
        args.with_images,
        image_dir,
        args.random_seed,
    )
    df_test = generate_mock_data(
        args.n_test,
        args.n_features,
        args.problem_type,
        args.with_images,
        image_dir,
        args.random_seed + 1,  # Different seed for test
    )

    # Save CSVs
    train_csv_path = os.path.join(args.output_dir, "train.csv")
    test_csv_path = os.path.join(args.output_dir, "test.csv")
    df_train.to_csv(train_csv_path, index=False)
    df_test.to_csv(test_csv_path, index=False)
    print(f"Generated train.csv at {train_csv_path}")
    print(f"Generated test.csv at {test_csv_path}")

    if args.with_images:
        # ZIP images
        images_zip_path = os.path.join(args.output_dir, "images.zip")
        with zipfile.ZipFile(images_zip_path, "w") as zipf:
            for root, _, files in os.walk(image_dir):
                for file in files:
                    zipf.write(os.path.join(root, file), arcname=file)  # Relative paths
        print(f"Generated images.zip at {images_zip_path}")

        # Clean up temp image dir
        shutil.rmtree(image_dir)

    print("\nUsage example for the tool:")
    print(f"python multimodal_learner.py --input_csv_train {train_csv_path} --input_csv_test {test_csv_path} --target_column target --output_csv metrics.csv --output_json results.json --output_html report.html --random_seed {args.random_seed}")
    if args.with_images:
        print(f" --image_column image_path --images_zip {images_zip_path}")


if __name__ == "__main__":
    main()
