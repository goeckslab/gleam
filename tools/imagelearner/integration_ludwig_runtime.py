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
from ludwig.data.cache.manager import DatasetCache
from ludwig.experiment import experiment_cli
from ludwig.trainers.trainer import Trainer
from ludwig_backend import LudwigDirectBackend
from PIL import Image


class SmallImageBackend(LudwigDirectBackend):
    def prepare_config(self, config_params, split_config):
        config = yaml.safe_load(super().prepare_config(config_params, split_config))
        config["input_features"][0]["encoder"] = {
            "type": "stacked_cnn",
            "conv_layers": [{"num_filters": 2, "filter_size": 3, "pool_size": 2}],
            "fc_layers": [{"output_size": 4}],
        }
        # Only batch size 2 fits the dataset-fraction limit. Real probing then
        # selects the same size independently of CPU throughput fluctuations.
        config["trainer"]["max_batch_size"] = 2
        config["trainer"]["eval_batch_size"] = "auto"
        return yaml.safe_dump(config)


class OriginalBackend(SmallImageBackend):
    def run_experiment(self, dataset_path, config_path, output_dir, random_seed=42):
        # Before caching, every member used its own CSV and reloaded its model
        # for validation prediction. Keep this baseline independent of the cache.
        member_csv = dataset_path.with_name(f"{output_dir.name}.csv")
        shutil.copyfile(dataset_path, member_csv)
        output_dir.mkdir(parents=True, exist_ok=True)
        experiment_cli(
            dataset=str(member_csv), config=str(config_path),
            output_directory=str(output_dir), random_seed=random_seed,
            skip_preprocessing=True,
        )


class RuntimeEquivalenceTest(unittest.TestCase):
    def assert_values_equal(self, expected, actual):
        if isinstance(expected, dict):
            self.assertEqual(expected.keys(), actual.keys())
            for key in expected:
                self.assert_values_equal(expected[key], actual[key])
        elif isinstance(expected, list):
            self.assertEqual(len(expected), len(actual))
            for left, right in zip(expected, actual):
                self.assert_values_equal(left, right)
        elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
            np.testing.assert_allclose(expected, actual, rtol=1e-6, atol=1e-7, equal_nan=True)
        else:
            self.assertEqual(expected, actual)

    def compare_ensemble(self, task="binary", fold_role="validation", num_seeds=2, cross_validation=True):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix="imagelearner_runtime_") as tmp:
            root = Path(tmp).resolve()
            image_dir = root / "input_images"
            image_dir.mkdir()
            rows = []
            rng = np.random.RandomState(123)
            for index in range(30):
                name = f"image{index}.png"
                pixels = rng.randint(0, 256, (12, 12, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(image_dir / name)
                label = index % 2
                if task == "category":
                    label = f"class{index % 3}"
                elif task == "regression":
                    label = index / 3.0
                rows.append({"image_path": name, "label": label, "split": 2 if index >= 24 else index % 2})
            csv_path = root / "metadata.csv"
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            image_zip = root / "images.zip"
            with zipfile.ZipFile(image_zip, "w") as archive:
                for image in image_dir.iterdir():
                    archive.write(image, image.name)

            outputs = []
            counts = []
            durations = []
            for backend_class in (OriginalBackend, SmallImageBackend):
                output = root / backend_class.__name__
                output.mkdir()
                args = argparse.Namespace(
                    csv_file=csv_path, image_zip=image_zip, model_name="resnet18",
                    use_pretrained=False, fine_tune=False, epochs=2, early_stop=5,
                    batch_size=None, output_dir=output, validation_size=0.15,
                    preprocessing_num_processes=1, split_probabilities=[0.7, 0.1, 0.2],
                    random_seed=42, learning_rate=0.001, augmentation=None,
                    image_resize="original", threshold=0.6 if fold_role == "test" else None,
                    threshold_mode="manual" if fold_role == "test" else "auto",
                    threshold_metric="f1", validation_metric=None,
                    target_column=None, image_column=None, sample_id_column=None,
                    torch_home=None, cross_validation=cross_validation,
                    num_folds=2, fold_role=fold_role, num_seeds=num_seeds,
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
                    with patch.object(DatasetCache, "put", put), patch.object(Trainer, "tune_batch_size", tune):
                        ImageLearnerCLI(args, backend_class()).run()
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
            self.assert_values_equal(before_summary, after_summary)
            for relative in ("predictions.csv", "validation_predictions.csv", *[
                f"members/{member['name']}/{filename}"
                for member in after_summary["members"]
                for filename in ("predictions.csv", "validation_predictions.csv")
            ]):
                left = pd.read_csv(before / relative)
                right = pd.read_csv(after / relative)
                pd.testing.assert_frame_equal(left, right, check_exact=False, rtol=1e-6, atol=1e-7)
            for member in after_summary["members"]:
                relative = Path("members") / member["name"]
                for filename in ("training_statistics.json", "test_statistics.json"):
                    self.assert_values_equal(
                        json.loads((before / relative / filename).read_text()),
                        json.loads((after / relative / filename).read_text()),
                    )
                weights_before = torch.load(before / relative / "model/model_weights", map_location="cpu")
                weights_after = torch.load(after / relative / "model/model_weights", map_location="cpu")
                self.assertEqual(weights_before.keys(), weights_after.keys())
                for name in weights_before:
                    self.assertTrue(torch.equal(weights_before[name], weights_after[name]), name)
            folds = 2 if cross_validation else 1
            self.assertEqual(counts[0]["preprocessing"], folds * num_seeds)
            self.assertEqual(counts[1]["preprocessing"], folds)
            self.assertEqual(counts[0]["batch_probes"], 2 * folds * num_seeds)
            self.assertEqual(counts[1]["batch_probes"], 2 * folds)
            print(json.dumps({
                "task": task, "fold_role": fold_role, "cross_validation": cross_validation,
                "before": dict(counts[0]), "after": dict(counts[1]),
                "seconds_before": durations[0], "seconds_after": durations[1],
            }), flush=True)

    def test_binary_validation_folds(self):
        self.compare_ensemble()

    def test_binary_test_folds_manual_threshold(self):
        self.compare_ensemble(fold_role="test")

    def test_category_validation_folds(self):
        self.compare_ensemble(task="category")

    def test_regression_seeds_only(self):
        self.compare_ensemble(task="regression", cross_validation=False)


if __name__ == "__main__":
    unittest.main()
