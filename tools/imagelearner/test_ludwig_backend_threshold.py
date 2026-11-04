import importlib
import io
import json
import sys
import types
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pandas as pd
import pytest
import yaml
from PIL import Image


TOOL_DIR = Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

ludwig = types.ModuleType("ludwig")
ludwig.__path__ = []
ludwig_globals = types.ModuleType("ludwig.globals")
ludwig_globals.DESCRIPTION_FILE_NAME = "description.json"
ludwig_globals.PREDICTIONS_PARQUET_FILE_NAME = "predictions.parquet"
ludwig_globals.TEST_STATISTICS_FILE_NAME = "test_statistics.json"
ludwig_globals.TRAIN_SET_METADATA_FILE_NAME = "training_set_metadata.json"
ludwig_utils = types.ModuleType("ludwig.utils")
ludwig_data_utils = types.ModuleType("ludwig.utils.data_utils")
ludwig_data_utils.get_split_path = lambda dataset: f"{dataset}.split"
ludwig.globals = ludwig_globals
ludwig.utils = ludwig_utils
sys.modules["ludwig"] = ludwig
sys.modules["ludwig.globals"] = ludwig_globals
sys.modules["ludwig.utils"] = ludwig_utils
sys.modules["ludwig.utils.data_utils"] = ludwig_data_utils

sys.modules.pop("ludwig_backend", None)


def _load_backend_test_objects():
    constants = importlib.import_module("constants")
    ludwig_backend = importlib.import_module("ludwig_backend")
    return (
        constants.IMAGE_PATH_COLUMN_NAME,
        constants.LABEL_COLUMN_NAME,
        constants.SPLIT_COLUMN_NAME,
        ludwig_backend.LudwigDirectBackend,
    )


def _write_image_zip(tmp_path, sizes, suffixes=None):
    zip_path = tmp_path / "images.zip"
    suffixes = suffixes or [".png"] * len(sizes)
    assert len(suffixes) == len(sizes)
    with zipfile.ZipFile(zip_path, "w") as zf:
        for index, (size, suffix) in enumerate(zip(sizes, suffixes)):
            image = Image.new("RGB", size, color="white")
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            zf.writestr(f"image_{index}{suffix}", buffer.getvalue())
    return zip_path


def test_generate_validation_predictions_writes_split_specific_file(tmp_path):
    (
        image_path_col,
        label_col,
        split_col,
        LudwigDirectBackend,
    ) = _load_backend_test_objects()
    exp_dir = tmp_path / "experiment_run"
    (exp_dir / "model").mkdir(parents=True)
    data_dir = tmp_path / "prepared"
    data_dir.mkdir()
    prepared_csv = data_dir / "prepared_data.csv"
    pd.DataFrame(
        {
            image_path_col: [
                "images/train.jpg",
                "images/val_a.jpg",
                "images/val_b.jpg",
                "images/test.jpg",
            ],
            label_col: [0, 1, 0, 1],
            split_col: [0, 1, 1, 2],
        }
    ).to_csv(prepared_csv, index=False)

    class StubLudwigModel:
        @classmethod
        def load(cls, model_dir):
            assert model_dir == str(exp_dir / "model")
            return cls()

        def predict(self, dataset=None, **_kwargs):
            assert len(dataset) == 2
            assert all(Path(path).is_absolute() for path in dataset[image_path_col])
            return pd.DataFrame(
                {
                    "label_probabilities_0": [0.2, 0.7],
                    "label_probabilities_1": [0.8, 0.3],
                }
            )

    ludwig_api = types.ModuleType("ludwig.api")
    ludwig_api.LudwigModel = StubLudwigModel
    sys.modules["ludwig.api"] = ludwig_api

    output_path = LudwigDirectBackend().generate_split_predictions(
        tmp_path,
        prepared_csv,
        split_value=1,
        output_filename="validation_predictions.csv",
    )

    assert output_path == exp_dir / "validation_predictions.csv"
    df = pd.read_csv(output_path)
    assert df[split_col].tolist() == [1, 1]
    assert df[label_col].tolist() == [1, 0]
    assert df["label_probabilities_1"].tolist() == [0.8, 0.3]


@pytest.fixture
def runtime_backend(tmp_path, monkeypatch):
    _, _, _, backend_class = _load_backend_test_objects()
    monkeypatch.setattr(backend_class, "_patch_torchvision_download", lambda self: None)
    monkeypatch.setattr(importlib.import_module("ludwig_backend"), "configure_binary_metrics", lambda: None)
    dataset = tmp_path / "fold1.csv"
    pd.DataFrame({
        "image_path": ["images/a.png"] * 4,
        "label": [0, 1, 0, 1],
        "split": [0, 1, 1, 2],
    }).to_csv(dataset, index=False)
    config = tmp_path / "config.yaml"
    config.write_text("trainer:\n  batch_size: auto\n  learning_rate: 0.001\n")
    calls = []
    prepared = (["train"], ["val1", "val2"], ["test"], {})
    model = types.SimpleNamespace(config_obj=types.SimpleNamespace(
        trainer=types.SimpleNamespace(batch_size=8, eval_batch_size=16),
    ), model=types.SimpleNamespace(named_parameters=lambda: [], state_dict=lambda: {}))

    def experiment_cli(**kwargs):
        calls.append(kwargs)
        output = Path(kwargs["output_directory"]) / "experiment_run"
        (output / "model").mkdir(parents=True, exist_ok=True)
        (output / "description.json").write_text("{}")
        return model, {}, {}, prepared, str(output)

    experiment_module = types.ModuleType("ludwig.experiment")
    experiment_module.experiment_cli = experiment_cli
    monkeypatch.setitem(sys.modules, "ludwig.experiment", experiment_module)
    return backend_class(), dataset, config, calls, model, prepared, experiment_module


def test_runtime_reuses_batch_sizes_only_for_unchanged_split_and_config(runtime_backend, tmp_path):
    backend, dataset, config, calls, _, prepared, _ = runtime_backend
    backend.run_experiment(dataset, config, tmp_path / "seed42", 42)
    backend.run_experiment(dataset, config, tmp_path / "seed43", 43)
    assert calls[0]["config"]["trainer"]["batch_size"] == "auto"
    assert calls[1]["config"]["trainer"] == {
        "batch_size": 8, "eval_batch_size": 16, "learning_rate": 0.001,
    }
    assert [call["random_seed"] for call in calls] == [42, 43]
    assert calls[1]["training_set"] is prepared[0]
    assert calls[1]["training_set_metadata"] is not backend._prepared_context[1][3]

    dataset.write_text(dataset.read_text().replace("0,0", "1,0"))
    backend.run_experiment(dataset, config, tmp_path / "changed_data", 44)
    assert calls[-1]["config"]["trainer"]["batch_size"] == "auto"
    assert "training_set" not in calls[-1]

    config.write_text(config.read_text().replace("0.001", "0.002"))
    backend.run_experiment(dataset, config, tmp_path / "changed_config", 45)
    assert calls[-1]["config"]["trainer"]["batch_size"] == "auto"

    other_fold = tmp_path / "fold2.csv"
    other_fold.write_bytes(dataset.read_bytes())
    backend.run_experiment(other_fold, config, tmp_path / "other_fold", 46)
    assert calls[-1]["config"]["trainer"]["batch_size"] == "auto"


def test_runtime_keeps_explicit_batch_size(runtime_backend, tmp_path):
    backend, dataset, config, calls, model, _, _ = runtime_backend
    config.write_text("trainer:\n  batch_size: 4\n  eval_batch_size: 12\n")
    model.config_obj.trainer.batch_size = 4
    model.config_obj.trainer.eval_batch_size = 12
    for seed in (42, 43):
        backend.run_experiment(dataset, config, tmp_path / str(seed), seed)
    assert all(call["config"]["trainer"] == {"batch_size": 4, "eval_batch_size": 12} for call in calls)


def test_runtime_keeps_probes_for_models_with_mutable_buffers(runtime_backend, tmp_path):
    backend, dataset, config, calls, model, prepared, _ = runtime_backend
    model.model.state_dict = lambda: {"encoder.running_mean": object()}
    for seed in (42, 43):
        backend.run_experiment(dataset, config, tmp_path / str(seed), seed)
    assert all(call["config"]["trainer"]["batch_size"] == "auto" for call in calls)
    assert calls[1]["training_set"] is prepared[0]


def test_runtime_can_reuse_probes_with_constant_loss_weights(runtime_backend, tmp_path):
    backend, dataset, config, calls, model, _, _ = runtime_backend
    model.model.state_dict = lambda: {"output_features.label.loss.loss_fn.pos_weight": object()}
    for seed in (42, 43):
        backend.run_experiment(dataset, config, tmp_path / str(seed), seed)
    assert calls[1]["config"]["trainer"]["batch_size"] == 8


def test_runtime_reuses_live_model_and_prepared_validation_data(runtime_backend, tmp_path, monkeypatch):
    backend, dataset, config, _, model, prepared, _ = runtime_backend
    output = tmp_path / "seed42"
    backend.run_experiment(dataset, config, output, 42)

    def predict(dataset, **kwargs):
        assert dataset is prepared[1]
        return pd.DataFrame({"label_probabilities_1": [0.8, 0.3]}), None

    model.predict = predict
    monkeypatch.delitem(sys.modules, "ludwig.api", raising=False)
    prediction_path = backend.generate_split_predictions(output, dataset, 1, "validation_predictions.csv")
    predictions = pd.read_csv(prediction_path)
    assert predictions["label"].tolist() == [1, 0]
    assert predictions["label_probabilities_1"].tolist() == [0.8, 0.3]
    assert backend._prediction_context is None


def test_runtime_falls_back_to_raw_data_when_prepared_prediction_fails(runtime_backend, tmp_path):
    backend, dataset, config, _, model, prepared, _ = runtime_backend
    output = tmp_path / "seed42"
    backend.run_experiment(dataset, config, output, 42)
    calls = []

    def predict(dataset, **kwargs):
        calls.append(dataset)
        if dataset is prepared[1]:
            raise ValueError("prepared data unavailable")
        assert isinstance(dataset, pd.DataFrame)
        return pd.DataFrame({"label_probabilities_1": [0.8, 0.3]})

    model.predict = predict
    assert backend.generate_split_predictions(output, dataset, 1, "validation_predictions.csv").exists()
    assert len(calls) == 2


def test_runtime_does_not_use_another_runs_model(runtime_backend, tmp_path, monkeypatch):
    backend, dataset, config, _, live_model, _, _ = runtime_backend
    backend.run_experiment(dataset, config, tmp_path / "first", 42)
    other_output = tmp_path / "second"
    model_dir = other_output / "experiment_run/model"
    model_dir.mkdir(parents=True)

    class SavedModel:
        @classmethod
        def load(cls, path):
            assert path == str(model_dir)
            return cls()

        def predict(self, dataset, **kwargs):
            assert isinstance(dataset, pd.DataFrame)
            return pd.DataFrame({"label_probabilities_1": [0.6, 0.4]})

    monkeypatch.setitem(sys.modules, "ludwig.api", types.SimpleNamespace(LudwigModel=SavedModel))
    live_model.predict = lambda **kwargs: pytest.fail("reused the wrong member")
    path = backend.generate_split_predictions(other_output, dataset, 1, "validation_predictions.csv")
    assert pd.read_csv(path)["label_probabilities_1"].tolist() == [0.6, 0.4]
    assert backend._prediction_context is None


def test_runtime_retries_oom_with_original_config(runtime_backend, tmp_path, monkeypatch):
    backend, dataset, config, calls, _, _, module = runtime_backend
    backend.run_experiment(dataset, config, tmp_path / "seed42", 42)
    original = module.experiment_cli

    def experiment_cli(**kwargs):
        if isinstance(kwargs["config"], dict):
            assert kwargs["config"]["trainer"]["batch_size"] == 8
            raise RuntimeError("CUDA out of memory")
        assert kwargs["config"] == str(config)
        return original(**kwargs)

    monkeypatch.setattr(module, "experiment_cli", experiment_cli)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
    ))
    backend.run_experiment(dataset, config, tmp_path / "seed43", 43)
    assert calls[-1]["random_seed"] == 43
    assert backend._prediction_context is not None


def test_runtime_propagates_non_oom_and_releases_previous_model(runtime_backend, tmp_path, monkeypatch):
    backend, dataset, config, _, _, _, module = runtime_backend
    backend.run_experiment(dataset, config, tmp_path / "seed42", 42)

    def fail(**kwargs):
        raise RuntimeError("training error")

    monkeypatch.setattr(module, "experiment_cli", fail)
    with pytest.raises(RuntimeError, match="training error"):
        backend.run_experiment(dataset, config, tmp_path / "seed43", 43)
    assert backend._prediction_context is None


def test_prepare_config_records_category_top_k():
    (
        _image_path_col,
        _label_col,
        _split_col,
        LudwigDirectBackend,
    ) = _load_backend_test_objects()

    yaml_str = LudwigDirectBackend().prepare_config(
        {
            "model_name": "resnet18",
            "use_pretrained": False,
            "epochs": 1,
            "label_metadata": {"num_unique": 5},
        },
        {"type": "random", "probabilities": [0.7, 0.1, 0.2]},
    )
    config = yaml.safe_load(yaml_str)

    assert config["output_features"][0]["type"] == "category"
    assert config["output_features"][0]["top_k"] == 3


def test_prepare_config_records_explicit_resize_image_size_summary(tmp_path):
    (
        _image_path_col,
        _label_col,
        _split_col,
        LudwigDirectBackend,
    ) = _load_backend_test_objects()
    image_zip = _write_image_zip(tmp_path, [(96, 96), (96, 96)])
    params = {
        "model_name": "resnet18",
        "use_pretrained": False,
        "epochs": 1,
        "image_resize": "384x384",
        "image_zip": str(image_zip),
        "label_metadata": {"num_unique": 2},
    }

    yaml_str = LudwigDirectBackend().prepare_config(
        params,
        {"type": "random", "probabilities": [0.7, 0.1, 0.2]},
    )
    config = yaml.safe_load(yaml_str)
    preprocessing = config["input_features"][0]["preprocessing"]

    assert preprocessing["height"] == 384
    assert preprocessing["width"] == 384
    assert params["image_size"] == "384x384"
    assert params["image_size_adaptation"]["original_size"] == "96x96"
    assert params["image_size_adaptation"]["requested_resize"] == "384x384"
    assert params["image_size_adaptation"]["training_size"] == "384x384"
    assert params["image_size_adaptation"]["final_training_size"] == "384x384"
    assert "model_adaptation_size" not in params["image_size_adaptation"]


def test_detect_image_dimension_summary_uses_supported_suffixes(tmp_path):
    (
        _image_path_col,
        _label_col,
        _split_col,
        LudwigDirectBackend,
    ) = _load_backend_test_objects()
    constants = importlib.import_module("constants")
    suffixes = sorted(constants.IMAGE_FILE_SUFFIXES)
    image_zip = _write_image_zip(
        tmp_path,
        [(17, 23)] * len(suffixes),
        suffixes=suffixes,
    )

    summary = LudwigDirectBackend()._detect_image_dimension_summary(str(image_zip))

    assert summary["is_fallback"] is False
    assert summary["original_size"] == "23x17"
    assert summary["first_image_size"] == "23x17"
    assert summary["image_count"] == len(suffixes)
    assert summary["readable_image_count"] == len(suffixes)


def test_detect_image_dimension_summary_limits_inspection_count(tmp_path, monkeypatch):
    (
        _image_path_col,
        _label_col,
        _split_col,
        LudwigDirectBackend,
    ) = _load_backend_test_objects()
    ludwig_backend = importlib.import_module("ludwig_backend")
    monkeypatch.setattr(ludwig_backend, "MAX_IMAGE_DIMENSION_INSPECTION_COUNT", 2)
    image_zip = _write_image_zip(
        tmp_path,
        [(16, 16), (16, 16), (32, 32)],
    )

    summary = LudwigDirectBackend()._detect_image_dimension_summary(str(image_zip))

    assert summary["is_fallback"] is False
    assert summary["image_count"] == 3
    assert summary["inspected_image_count"] == 2
    assert summary["uninspected_image_count"] == 1
    assert summary["readable_image_count"] == 2
    assert summary["is_sampled"] is True


def test_prepare_config_records_metaformer_model_size_adaptation(tmp_path):
    (
        _image_path_col,
        _label_col,
        _split_col,
        LudwigDirectBackend,
    ) = _load_backend_test_objects()
    ludwig_backend = importlib.import_module("ludwig_backend")
    original_meta_default_cfgs = ludwig_backend.META_DEFAULT_CFGS
    ludwig_backend.META_DEFAULT_CFGS = {
        **original_meta_default_cfgs,
        "caformer_s18_384": {
            "input_size": (3, 384, 384),
            "url": "https://example.invalid/caformer_s18_384.pth",
        },
    }
    try:
        image_zip = _write_image_zip(tmp_path, [(96, 96)])
        params = {
            "model_name": "caformer_s18_384",
            "use_pretrained": True,
            "epochs": 1,
            "image_resize": "original",
            "image_zip": str(image_zip),
            "label_metadata": {"num_unique": 3},
        }
        yaml_str = LudwigDirectBackend().prepare_config(
            params,
            {"type": "random", "probabilities": [0.7, 0.1, 0.2]},
        )
    finally:
        ludwig_backend.META_DEFAULT_CFGS = original_meta_default_cfgs
    config = yaml.safe_load(yaml_str)
    input_feature = config["input_features"][0]

    assert input_feature["preprocessing"]["height"] == 96
    assert input_feature["preprocessing"]["width"] == 96
    assert input_feature["encoder"]["height"] == 96
    assert input_feature["encoder"]["width"] == 96
    assert params["image_size_adaptation"]["original_size"] == "96x96"
    assert params["image_size_adaptation"]["training_size"] == "96x96"
    assert params["image_size_adaptation"]["final_training_size"] == "96x96"
    assert params["image_size_adaptation"]["model_configured_size"] == "384x384"
    assert params["image_size_adaptation"]["model_adaptation_size"] == "224x224"
    assert params["image_size_adaptation"]["model_adaptation_from_size"] == "96x96"
    assert params["image_size_adaptation"]["model_adaptation_to_size"] == "224x224"


@pytest.mark.parametrize("mode, include_test_predictions, evaluation_threshold, selected_threshold, positive_label, reverse_rows", [
    ("auto", True, 0.5, 0.7, 1, False),
    ("manual", True, 0.3, 0.3, 1, False),
    ("manual", True, 0.6, 0.6, 1, False),
    ("auto", False, 0.5, 0.7, 1, False),
    ("manual", True, 0.0, 0.0, 1, False),
    ("auto", True, 0.5, 0.7, 2, False),
    ("auto", True, 0.5, 0.7, 2, True),
    ("auto", True, 0.5, 0.7, -1, False),
    ("auto", True, 0.5, 0.7, -1, True),
])
def test_report_labels_each_metrics_table_with_its_actual_threshold(
    tmp_path, monkeypatch, mode, include_test_predictions, evaluation_threshold, selected_threshold,
    positive_label, reverse_rows
):
    *_, backend_class = _load_backend_test_objects()
    backend_module = importlib.import_module("ludwig_backend")
    exp_dir = tmp_path / "experiment_run"
    exp_dir.mkdir()
    (exp_dir / "description.json").write_text(json.dumps({"config": {
        "output_features": [{"type": "binary", "threshold": evaluation_threshold}],
    }}))
    label_stats = {
        "accuracy": 0.75, "precision": 0.8, "recall": 0.5, "specificity": 1.0,
        "roc_auc": 0.9, "loss": 0.3, "confusion_matrix": [[2, 0], [1, 1]],
        "per_class_stats": {"0": {}, "1": {}},
    }
    (exp_dir / "test_statistics.json").write_text(json.dumps({"label": label_stats}))
    (exp_dir / "training_statistics.json").write_text(json.dumps({
        split: {"label": {key: [value] for key, value in label_stats.items()
                          if isinstance(value, float)}}
        for split in ("training", "validation")
    }))
    for name, split in [("validation_predictions.csv", 1), ("predictions.csv", 2)]:
        if split == 2 and not include_test_predictions:
            continue
        frame = pd.DataFrame({
            "split": [split] * 4, "label": [positive_label, 0, positive_label, 0],
            "label_probabilities_True": [0.9, 0.6, 0.7, 0.2],
            "label_probabilities_False": [0.1, 0.4, 0.3, 0.8],
        })
        if reverse_rows:
            frame = frame.iloc[::-1]
        frame.to_csv(exp_dir / name, index=False)
    fallback_thresholds = []

    def capture_fallback(*args, **kwargs):
        fallback_thresholds.append(kwargs["threshold"])
        return []

    monkeypatch.setattr(backend_module, "build_classification_plots", capture_fallback)
    for name in ("build_train_validation_plots", "build_prediction_diagnostics",
                 "build_binary_threshold_classification_plots_from_predictions"):
        monkeypatch.setattr(backend_module, name, lambda *args, **kwargs: [])
    monkeypatch.setattr(backend_module, "build_binary_threshold_plot", lambda *args, **kwargs: None)
    monkeypatch.setattr(backend_class, "_generate_gradcam_heatmaps", lambda *args: {})
    monkeypatch.setattr(backend_class, "get_training_process", lambda *args: {})
    tabs = {}
    render_tabs = backend_module.build_tabbed_html

    def capture_tabs(first, second, third):
        tabs.update(first=first, second=second, third=third)
        return render_tabs(first, second, third)

    monkeypatch.setattr(backend_module, "build_tabbed_html", capture_tabs)
    monkeypatch.chdir(tmp_path)
    config = {"threshold_mode": mode, "threshold_metric": "balanced_accuracy"}
    if mode == "manual":
        config["threshold"] = evaluation_threshold
    report_path = backend_class().generate_html_report("Threshold test", str(tmp_path), config, "Fixed split")
    assert report_path.exists()
    for tab in (tabs["first"], tabs["second"]):
        assert f"Decision threshold: {evaluation_threshold:.6f}" in tab
        assert "Source: Ludwig model evaluation." in tab
        assert "ROC-AUC and loss are independent of the decision threshold" in tab
    actual_test_threshold = selected_threshold if include_test_predictions else evaluation_threshold
    assert f"Decision threshold: {actual_test_threshold:.6f}" in tabs["third"]
    if not include_test_predictions:
        assert "selected-threshold test metrics unavailable" in tabs["third"]
    elif mode == "auto":
        assert "Source: Optimized on validation split." in tabs["third"]
        assert "Optimization metric: Balanced Accuracy." in tabs["third"]
        assert "validation probabilities unavailable" not in report_path.read_text()
        assert "selected-threshold test metrics unavailable" not in tabs["third"]
    else:
        assert "Source: Manual." in tabs["third"]
        assert "Optimization metric:" not in tabs["third"]
    assert "Evaluation Threshold (Experiment and Validation Summaries)" in tabs["first"]
    assert "Selected Threshold (Test Summary)" in tabs["first"]
    assert "Decision Threshold (Test)" not in tabs["first"]
    assert fallback_thresholds == [evaluation_threshold]

    if mode == "manual" and evaluation_threshold == 0.6:
        # Exercise the actual Galaxy report assertions locally without retraining.
        # This catches stale wrapper expectations when report labels change.
        wrapper = ET.parse(TOOL_DIR / "image_learner.xml").getroot()
        manual_test = next(
            test for test in wrapper.findall("./tests/test")
            if any(param.get("name", "").endswith("binary_threshold_policy|threshold")
                   and param.get("value") == "0.6" for param in test.findall("param"))
        )
        for assertion in manual_test.findall("./output[@name='output_report']/assert_contents/has_text"):
            assert assertion.get("text") in report_path.read_text()
