"""Run separately in the Image Learner container: python test_binary_metrics_integration.py."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from ludwig.api import LudwigModel
from ludwig.backend import initialize_backend
from ludwig.constants import BINARY, LOGITS, MAXIMIZE, PREDICTIONS, PROBABILITIES
from ludwig.distributed import init_dist_strategy
from ludwig.features.binary_feature import BinaryOutputFeature
from ludwig.modules.metric_registry import get_metric_cls, get_metric_tensor_input, register_metric
from ludwig.schema.features.binary_feature import ECDBinaryOutputFeatureConfig
from ludwig.utils.output_feature_utils import get_feature_concat_name
from ludwig_backend import configure_binary_metrics, LudwigDirectBackend
from PIL import Image
from plotly_plots import _resolve_positive_label_from_probability
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score


class BinaryMetricsIntegrationTest(unittest.TestCase):
    def setUp(self):
        init_dist_strategy("local")
        self.original_inputs = {
            name: get_metric_tensor_input(name)
            for name in ("precision", "recall", "specificity")
        }

    def tearDown(self):
        for name, tensor_input in self.original_inputs.items():
            register_metric(name, [BINARY], MAXIMIZE, tensor_input)(get_metric_cls(BINARY, name))

    def test_metrics_follow_model_predictions_at_threshold_boundaries(self):
        configure_binary_metrics()
        configure_binary_metrics()  # Repeated experiments must be safe in one process.
        probabilities = torch.tensor([0.9, 0.55, 0.57, 0.2, 0.6, 0.0, 1.0, 0.5])
        targets = torch.tensor([1, 0, 1, 0, 1, 0, 1, 1], dtype=torch.bool)
        logits = torch.logit(probabilities).clamp(-100, 100)
        reference_loss = None
        for threshold in (0.0, 0.5, 0.6, 1.0):
            with self.subTest(threshold=threshold):
                config = ECDBinaryOutputFeatureConfig(name="label", threshold=threshold)
                config.input_size = 1
                feature = BinaryOutputFeature(config, {})
                predictions = feature.predictions(
                    {get_feature_concat_name("label", LOGITS): logits}, "label"
                )
                decisions = predictions[PREDICTIONS].numpy()
                np.testing.assert_array_equal(decisions, probabilities.numpy() >= threshold)
                feature.update_metrics(targets, predictions)
                metrics = feature.get_metrics()
                expected = self.expected_metrics(targets.numpy(), decisions)
                for name, value in expected.items():
                    self.assertAlmostEqual(metrics[name], value, places=6)
                self.assertEqual(get_metric_tensor_input("roc_auc"), PROBABILITIES)
                self.assertEqual(get_metric_tensor_input("loss"), LOGITS)
                self.assertAlmostEqual(metrics["roc_auc"], roc_auc_score(targets.numpy(), probabilities), places=6)
                if reference_loss is None:
                    reference_loss = metrics["loss"]
                self.assertAlmostEqual(metrics["loss"], reference_loss, places=6)

    @staticmethod
    def expected_metrics(targets, decisions):
        tn, fp, fn, tp = confusion_matrix(targets, decisions, labels=[0, 1]).ravel()
        return {
            "accuracy": accuracy_score(targets, decisions),
            "precision": precision_score(targets, decisions, zero_division=0),
            "recall": recall_score(targets, decisions, zero_division=0),
            "specificity": tn / (tn + fp),
        }

    def test_numeric_label_mapping_matches_ludwig_preprocessing(self):
        backend = initialize_backend("local")
        for values in ([0, 2], [0, -1], ["0", "2"], ["0", "-1"]):
            for labels in (values, values[::-1]):
                with self.subTest(labels=labels):
                    targets = pd.Series(labels)
                    converted = BinaryOutputFeature.cast_column(targets, backend)
                    positive = _resolve_positive_label_from_probability(targets.to_numpy(), "True")
                    np.testing.assert_array_equal(targets.eq(positive), converted)
        targets = pd.Series([1, 2])
        self.assertTrue(BinaryOutputFeature.cast_column(targets, backend).all())
        self.assertIsNone(_resolve_positive_label_from_probability(targets.to_numpy(), "True"))

    def test_training_validation_and_test_statistics_use_manual_threshold(self):
        # Start from Ludwig's original routing so this also tests the training-entry hook.
        for name in self.original_inputs:
            register_metric(name, [BINARY], MAXIMIZE, PROBABILITIES)(get_metric_cls(BINARY, name))
        with tempfile.TemporaryDirectory(prefix="binary-metrics-") as directory:
            root = Path(directory)
            rng = np.random.RandomState(42)
            rows = []
            for split, count in ((0, 16), (1, 8), (2, 8)):
                for index in range(count):
                    image_path = root / f"image-{split}-{index}.png"
                    Image.fromarray(rng.randint(0, 256, (32, 32, 3), dtype=np.uint8)).save(image_path)
                    rows.append({"image_path": str(image_path), "label": index % 2, "split": split})
            dataset = pd.DataFrame(rows)
            dataset_path = root / "images.csv"
            dataset.to_csv(dataset_path, index=False)
            config = {
                "input_features": [{
                    "name": "image_path", "type": "image",
                    "encoder": {
                        "type": "stacked_cnn", "num_conv_layers": 1, "num_filters": 4,
                        "num_fc_layers": 1, "fc_size": 8,
                    },
                    "preprocessing": {"height": 32, "width": 32, "num_channels": 3, "in_memory": True},
                }],
                "output_features": [{"name": "label", "type": "binary", "threshold": 0.6}],
                "combiner": {"type": "concat", "num_fc_layers": 1, "fc_size": 8},
                "preprocessing": {"num_processes": 1, "split": {"type": "fixed", "column": "split"}},
                "trainer": {
                    "epochs": 1, "batch_size": 8, "eval_batch_size": 8, "learning_rate": 0.001,
                    "validation_field": "label", "validation_metric": "precision",
                    # Evaluate the final checkpoint on training data as well, so all saved
                    # split metrics can be checked against that checkpoint's predictions.
                    "evaluate_training_set": True,
                },
            }
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config))
            LudwigDirectBackend().run_experiment(dataset_path, config_path, root, random_seed=42)
            for metric in self.original_inputs:
                self.assertEqual(get_metric_tensor_input(metric), PREDICTIONS)
            experiment = root / "experiment_run"
            training_stats = json.loads((experiment / "training_statistics.json").read_text())
            test_stats = json.loads((experiment / "test_statistics.json").read_text())["label"]
            model = LudwigModel.load(str(experiment / "model"))
            for split, name in ((0, "training"), (1, "validation"), (2, "test")):
                with self.subTest(split=name):
                    subset = dataset[dataset["split"] == split].reset_index(drop=True)
                    stats, predictions, _ = model.evaluate(
                        subset, collect_predictions=True, collect_overall_stats=True,
                        output_directory=str(root / f"evaluate-{name}"),
                    )
                    scores = predictions["label_probabilities_True"].to_numpy()
                    decisions = scores >= 0.6
                    np.testing.assert_array_equal(predictions["label_predictions"], decisions)
                    expected = self.expected_metrics(subset["label"], decisions)
                    for metric, value in expected.items():
                        self.assertAlmostEqual(stats["label"][metric], value, places=6)
                        self.assertAlmostEqual(training_stats[name]["label"][metric][-1], value, places=6)
                        if split == 2:
                            self.assertAlmostEqual(test_stats[metric], value, places=6)
                    np.testing.assert_array_equal(
                        stats["label"]["confusion_matrix"],
                        confusion_matrix(subset["label"], decisions, labels=[0, 1]),
                    )
                    print(f"Verified {name} at threshold 0.6: {expected}", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
