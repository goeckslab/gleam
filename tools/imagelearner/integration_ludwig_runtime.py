"""Compare cached ensembles with the original execution path in Ludwig 0.10.1.

Run with the Image Learner production container. Uses a tiny CPU CNN and real
automatic batch-size probing, preprocessing, training, predictions and reports.
"""
import argparse
import json
import os
import shutil
import tempfile
import time
import unittest
import zipfile
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import yaml
from image_workflow import ImageLearnerCLI
from ludwig.api import LudwigModel
from ludwig.data.cache.manager import DatasetCache
from ludwig.experiment import experiment_cli
from ludwig.trainers.trainer import Trainer
from ludwig_backend import configure_binary_metrics, LudwigDirectBackend
from PIL import Image
from plotly_plots import _resolve_display_labels


class SmallImageBackend(LudwigDirectBackend):
    def prepare_config(self, config_params, split_config):
        config = yaml.safe_load(super().prepare_config(config_params, split_config))
        config["input_features"][0]["encoder"] = {
            "type": "stacked_cnn",
            "conv_layers": [{"num_filters": 2, "filter_size": 3, "pool_size": 2}],
            "fc_layers": [{"output_size": 4}],
        }
        # A single candidate makes real probing independent of CPU throughput.
        config["trainer"]["max_batch_size"] = 1
        config["trainer"]["eval_batch_size"] = "auto"
        return yaml.safe_dump(config)


class OriginalBackend(SmallImageBackend):
    def run_experiment(self, dataset_path, config_path, output_dir, random_seed=42):
        # Before caching, every member used its own CSV and reloaded its model
        # for validation prediction. Keep this baseline independent of the cache.
        member_csv = dataset_path.with_name(f"{output_dir.name}.csv")
        shutil.copyfile(dataset_path, member_csv)
        output_dir.mkdir(parents=True, exist_ok=True)
        configure_binary_metrics()
        experiment_cli(
            dataset=str(member_csv), config=str(config_path),
            output_directory=str(output_dir), random_seed=random_seed,
            skip_preprocessing=True,
        )


def write_dataset(root, labels, with_split=True, seed=123):
    """Random-noise images (each image scores differently) with the given labels."""
    image_dir = root / "input_images"
    image_dir.mkdir()
    rng = np.random.RandomState(seed)
    rows = []
    for index, label in enumerate(labels):
        name = f"image{index}.png"
        Image.fromarray(rng.randint(0, 256, (12, 12, 3), dtype=np.uint8)).save(image_dir / name)
        row = {"image_path": name, "label": label}
        if with_split:
            # Pairs of rows alternate between train and validation, so both see every class.
            row["split"] = 2 if index >= len(labels) - 6 else (index // 2) % 2
        rows.append(row)
    csv_path = root / "metadata.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    image_zip = root / "images.zip"
    with zipfile.ZipFile(image_zip, "w") as archive:
        for image in image_dir.iterdir():
            archive.write(image, image.name)
    return csv_path, image_zip, image_dir


def make_args(csv_path, image_zip, output, **overrides):
    values = dict(
        csv_file=csv_path, image_zip=image_zip, model_name="resnet18",
        use_pretrained=False, fine_tune=False, epochs=2, early_stop=5,
        batch_size=None, output_dir=output, validation_size=0.15,
        preprocessing_num_processes=1, split_probabilities=[0.7, 0.1, 0.2],
        random_seed=42, learning_rate=0.001, augmentation=None,
        image_resize="original", threshold=None, threshold_mode="auto",
        threshold_metric="f1", validation_metric=None,
        target_column=None, image_column=None, sample_id_column=None,
        torch_home=None, cross_validation=True, num_folds=2,
        fold_role="validation", num_seeds=2,
        # The tests reload and compare every member's weights.
        keep_member_weights=True,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def run_cli(args, backend):
    previous_dir = Path.cwd()
    try:
        os.chdir(args.output_dir)
        ImageLearnerCLI(args, backend).run()
    finally:
        os.chdir(previous_dir)
    return (args.output_dir / "image_classification_results_report.html").read_text()


class RuntimeEquivalenceTest(unittest.TestCase):
    def assert_values_equal(self, expected, actual, path="root"):
        if isinstance(expected, dict):
            self.assertEqual(expected.keys(), actual.keys())
            for key in expected:
                self.assert_values_equal(expected[key], actual[key], f"{path}.{key}")
        elif isinstance(expected, list):
            self.assertEqual(len(expected), len(actual))
            for index, (left, right) in enumerate(zip(expected, actual)):
                self.assert_values_equal(left, right, f"{path}[{index}]")
        elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
            np.testing.assert_allclose(expected, actual, rtol=1e-6, atol=1e-7, equal_nan=True, err_msg=path)
        else:
            self.assertEqual(expected, actual)

    def compare_ensemble(self, task="binary", fold_role="validation", num_seeds=2, cross_validation=True, stateful=False):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix="imagelearner_runtime_") as tmp:
            root = Path(tmp).resolve()
            labels = []
            for index in range(30):
                label = index % 2
                if task == "category":
                    label = f"class{index % 3}"
                elif task == "regression":
                    label = index / 3.0
                labels.append(label)
            # Test folds refuse a test split the user supplied; let the tool generate one.
            csv_path, image_zip, _ = write_dataset(root, labels, with_split=fold_role != "test")

            outputs = []
            counts = []
            durations = []
            for backend_class in (OriginalBackend, SmallImageBackend):
                output = root / backend_class.__name__
                output.mkdir()
                args = make_args(
                    csv_path, image_zip, output,
                    threshold=0.6 if fold_role == "test" else None,
                    threshold_mode="manual" if fold_role == "test" else "auto",
                    cross_validation=cross_validation, fold_role=fold_role, num_seeds=num_seeds,
                )
                counter = Counter()
                original_put = DatasetCache.put
                original_tune = Trainer.tune_batch_size

                def put(cache, *args, **kwargs):
                    counter["preprocessing"] += 1
                    return original_put(cache, *args, **kwargs)

                def tune(trainer, *args, **kwargs):
                    counter["batch_probes"] += 1
                    return original_tune(trainer, *args, **kwargs)

                previous_dir = Path.cwd()
                started = time.monotonic()
                try:
                    os.chdir(output)
                    backend = backend_class()
                    if stateful:
                        prepare_config = backend.prepare_config

                        def stateful_config(*args, **kwargs):
                            config = yaml.safe_load(prepare_config(*args, **kwargs))
                            config["input_features"][0]["encoder"]["conv_layers"][0]["norm"] = "batch"
                            return yaml.safe_dump(config)

                        backend.prepare_config = stateful_config
                    with patch.object(DatasetCache, "put", put), patch.object(Trainer, "tune_batch_size", tune):
                        ImageLearnerCLI(args, backend).run()
                finally:
                    os.chdir(previous_dir)
                durations.append(time.monotonic() - started)
                self.assertIn("Ensemble Summary", (output / "image_classification_results_report.html").read_text())
                outputs.append(output / "experiment_run")
                counts.append(counter)

            before, after = outputs
            before_summary = json.loads((before / "ensemble_summary.json").read_text())
            after_summary = json.loads((after / "ensemble_summary.json").read_text())
            self.assertEqual(after_summary["members_trained"], (2 if cross_validation else 1) * num_seeds)
            for relative in ("predictions.csv", "validation_predictions.csv", *[
                f"members/{member['name']}/{filename}"
                for member in after_summary["members"]
                for filename in ("predictions.csv", "validation_predictions.csv")
            ]):
                left = pd.read_csv(before / relative)
                right = pd.read_csv(after / relative)
                pd.testing.assert_frame_equal(left, right, check_exact=False, rtol=1e-6, atol=1e-7, obj=relative)
            for member in after_summary["members"]:
                relative = Path("members") / member["name"]
                for filename in ("training_statistics.json", "test_statistics.json"):
                    self.assert_values_equal(
                        json.loads((before / relative / filename).read_text()),
                        json.loads((after / relative / filename).read_text()), str(relative / filename),
                    )
                weights_before = torch.load(before / relative / "model/model_weights", map_location="cpu")
                weights_after = torch.load(after / relative / "model/model_weights", map_location="cpu")
                self.assertEqual(weights_before.keys(), weights_after.keys())
                for name in weights_before:
                    self.assertTrue(torch.equal(weights_before[name], weights_after[name]), f"{relative}: {name}")
            self.assert_values_equal(before_summary, after_summary)
            folds = 2 if cross_validation else 1
            self.assertEqual(counts[0]["preprocessing"], folds * num_seeds)
            self.assertEqual(counts[1]["preprocessing"], folds)
            self.assertEqual(counts[0]["batch_probes"], 2 * folds * num_seeds)
            self.assertEqual(counts[1]["batch_probes"], 2 * folds * (num_seeds if stateful else 1))
            print(json.dumps({
                "task": task, "fold_role": fold_role, "cross_validation": cross_validation, "stateful": stateful,
                "before": dict(counts[0]), "after": dict(counts[1]),
                "seconds_before": durations[0], "seconds_after": durations[1],
            }), flush=True)

    def test_single_model_run_matches_the_original_path(self):
        """Without ensemble options the cached backend must match one that calls Ludwig the way main does.

        Same predictions, statistics and weights as OriginalBackend, which runs Ludwig from the config file
        with skip_preprocessing and reloads the saved model for validation predictions.
        """
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix="imagelearner_single_") as tmp:
            root = Path(tmp).resolve()
            csv_path, image_zip, _ = write_dataset(root, [index % 2 for index in range(30)])
            runs = []
            for backend_class in (OriginalBackend, SmallImageBackend):
                output = root / backend_class.__name__
                output.mkdir()
                args = make_args(csv_path, image_zip, output, cross_validation=False, num_seeds=1)
                report = run_cli(args, backend_class())
                self.assertTrue("Ensemble Summary" not in report, "single-model run rendered an ensemble")
                self.assertTrue("Test Performance Summary" in report, "single-model report is incomplete")
                runs.append(output / "experiment_run")
            before, after = runs
            self.assertFalse((after / "ensemble_summary.json").exists())
            for filename in ("predictions.csv", "validation_predictions.csv"):
                pd.testing.assert_frame_equal(
                    pd.read_csv(before / filename), pd.read_csv(after / filename),
                    check_exact=False, rtol=1e-6, atol=1e-7, obj=filename,
                )
            for filename in ("training_statistics.json", "test_statistics.json"):
                self.assert_values_equal(
                    json.loads((before / filename).read_text()), json.loads((after / filename).read_text()), filename,
                )
            weights_before = torch.load(before / "model/model_weights", map_location="cpu")
            weights_after = torch.load(after / "model/model_weights", map_location="cpu")
            self.assertEqual(weights_before.keys(), weights_after.keys())
            for name in weights_before:
                self.assertTrue(torch.equal(weights_before[name], weights_after[name]), name)

    def test_binary_validation_folds(self):
        self.compare_ensemble()

    def test_binary_test_folds_manual_threshold(self):
        self.compare_ensemble(fold_role="test")

    def test_category_validation_folds(self):
        self.compare_ensemble(task="category")

    def test_stateful_encoder_preserves_probe_side_effects(self):
        self.compare_ensemble(stateful=True)

    def test_regression_seeds_only(self):
        self.compare_ensemble(task="regression", cross_validation=False)


class EnsembleRowAlignmentTest(unittest.TestCase):
    """Averaged predictions must belong to the image and label of their row.

    Member predictions are matched to dataset rows by position. Each member
    model is reloaded and scores the raw images directly, so a reordered split
    would show up as probabilities attached to the wrong image.
    """

    def run_ensemble(self, root, labels, fold_role="validation", num_seeds=1):
        csv_path, image_zip, image_dir = write_dataset(root, labels, seed=7)
        output = root / "out"
        output.mkdir()
        args = make_args(csv_path, image_zip, output, fold_role=fold_role, num_seeds=num_seeds)
        report = run_cli(args, SmallImageBackend())
        self.assertTrue("Ensemble Summary" in report, "the ensemble fell back to the minimal report")
        return image_dir, output / "experiment_run"

    def direct_probabilities(self, run_dir, image_dir):
        """Each member's class probabilities for every image, scored image by image order."""
        summary = json.loads((run_dir / "ensemble_summary.json").read_text())
        frame = pd.DataFrame({"image_path": sorted(str(p) for p in image_dir.iterdir())})
        scored = {}
        for member in summary["members"]:
            model = LudwigModel.load(str(run_dir / "members" / member["name"] / "model"))
            predictions, _ = model.predict(dataset=frame)
            columns = [c for c in predictions.columns if c.startswith("label_probabilities_")]
            scored[member["name"]] = predictions[columns].set_axis(
                [Path(p).name for p in frame["image_path"]], axis=0
            )
        return summary, scored

    @staticmethod
    def class_name(column):
        """A probability column's class, normalized independently of the code under test."""
        name = column[len("label_probabilities_"):].strip()
        name = {"False": "0", "True": "1"}.get(name, name)
        try:
            number = float(name)
        except ValueError:
            return name
        return str(int(number)) if number.is_integer() else str(number)

    def assert_rows_match_member_models(self, run_dir, image_dir):
        summary, scored = self.direct_probabilities(run_dir, image_dir)
        scored = {name: frame.rename(columns=self.class_name) for name, frame in scored.items()}
        classes = sorted(next(iter(scored.values())).columns)
        # Without image-specific scores a reordered split could not be detected.
        self.assertGreater(float(next(iter(scored.values()))[classes[0]].std()), 1e-4)
        mean = sum(frame[classes] for frame in scored.values()) / len(scored)
        for filename, split in (("predictions.csv", "test"), ("validation_predictions.csv", "validation")):
            reported = pd.read_csv(run_dir / filename)
            self.assertEqual(len(reported), summary["rows"][split])
            columns = {self.class_name(c): c for c in reported.columns if c.startswith("label_probabilities_")}
            self.assertEqual(sorted(columns), classes, filename)
            for _, row in reported.iterrows():
                name = Path(row["image_path"]).name
                values = row[[columns[c] for c in classes]].to_numpy(dtype=float)
                if split == "test":
                    # Every member scores the fixed test split; the ensemble is their mean.
                    expected = [mean.loc[name].to_numpy(dtype=float)]
                else:
                    # A validation row is scored by the member of its own fold only.
                    expected = [frame.loc[name, classes].to_numpy(dtype=float) for frame in scored.values()]
                gap = min(float(np.abs(values - candidate).max()) for candidate in expected)
                self.assertLess(gap, 1e-5, f"{filename}: row {name} does not carry its own image's scores")

    def test_binary_rows_keep_their_own_predictions(self):
        with tempfile.TemporaryDirectory(prefix="imagelearner_alignment_") as tmp:
            root = Path(tmp).resolve()
            image_dir, run_dir = self.run_ensemble(root, [index % 2 for index in range(30)])
            self.assert_rows_match_member_models(run_dir, image_dir)

    def test_float_and_spaced_category_labels_are_scored(self):
        # Ludwig names float classes "1.0" and strips spaces; every test row must still count.
        for labels in ([1.0, 2.0, 3.0], [" cat ", "dog", " bird"]):
            with tempfile.TemporaryDirectory(prefix="imagelearner_labels_") as tmp:
                root = Path(tmp).resolve()
                image_dir, run_dir = self.run_ensemble(root, [labels[index % 3] for index in range(30)])
                statistics = json.loads((run_dir / "test_statistics.json").read_text())["label"]
                self.assertEqual(int(np.sum(statistics["confusion_matrix"])), 6, labels)
                self.assert_report_labels_name_the_right_classes(run_dir, statistics)
                self.assert_rows_match_member_models(run_dir, image_dir)

    def assert_report_labels_name_the_right_classes(self, run_dir, statistics):
        """The per-class plots label each statistics key with the class it belongs to."""
        meta = json.loads((run_dir / "model" / "training_set_metadata.json").read_text())
        keys = list(statistics["per_class_stats"])
        shown = _resolve_display_labels(keys, meta["label"]["idx2str"])

        def same(value):
            return self.class_name(f"label_probabilities_{value}")

        self.assertEqual([same(label) for label in shown], [same(key) for key in keys])

    def test_single_model_report_labels_integer_classes_correctly(self):
        # idx2str is frequency-ordered ('2', '0', '1'); the statistics are keyed by class name.
        with tempfile.TemporaryDirectory(prefix="imagelearner_intlabels_") as tmp:
            root = Path(tmp).resolve()
            csv_path, image_zip, _ = write_dataset(root, [2] * 14 + [0] * 10 + [1] * 6)
            output = root / "out"
            output.mkdir()
            run_cli(make_args(csv_path, image_zip, output, cross_validation=False, num_seeds=1), SmallImageBackend())
            run_dir = output / "experiment_run"
            statistics = json.loads((run_dir / "test_statistics.json").read_text())["label"]
            support = {
                key: stats["true_positives"] + stats["false_negatives"]
                for key, stats in statistics["per_class_stats"].items()
            }
            self.assertEqual(support["1"], 6)  # every test row is label 1
            self.assert_report_labels_name_the_right_classes(run_dir, statistics)


if __name__ == "__main__":
    unittest.main()
