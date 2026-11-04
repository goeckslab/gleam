import importlib
import io
import sys
import types
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
    ))

    def experiment_cli(**kwargs):
        calls.append(kwargs)
        output = Path(kwargs["output_directory"]) / "experiment_run"
        (output / "model").mkdir(parents=True, exist_ok=True)
        return model, {}, {}, prepared, str(output)

    experiment_module = types.ModuleType("ludwig.experiment")
    experiment_module.experiment_cli = experiment_cli
    monkeypatch.setitem(sys.modules, "ludwig.experiment", experiment_module)
    return backend_class(), dataset, config, calls, model, prepared, experiment_module


def test_runtime_reuses_batch_sizes_only_for_unchanged_split_and_config(runtime_backend, tmp_path):
    backend, dataset, config, calls, _, _, _ = runtime_backend
    backend.run_experiment(dataset, config, tmp_path / "seed42", 42)
    backend.run_experiment(dataset, config, tmp_path / "seed43", 43)
    assert calls[0]["config"]["trainer"]["batch_size"] == "auto"
    assert calls[1]["config"]["trainer"] == {
        "batch_size": 8, "eval_batch_size": 16, "learning_rate": 0.001,
    }
    assert [call["random_seed"] for call in calls] == [42, 43]

    dataset.write_text(dataset.read_text().replace("0,0", "1,0"))
    backend.run_experiment(dataset, config, tmp_path / "changed_data", 44)
    assert calls[-1]["config"]["trainer"]["batch_size"] == "auto"

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
