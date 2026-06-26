import importlib
import io
import sys
import types
import zipfile
from pathlib import Path

import pandas as pd
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


def _write_image_zip(tmp_path, sizes):
    zip_path = tmp_path / "images.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for index, size in enumerate(sizes):
            image = Image.new("RGB", size, color="white")
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            zf.writestr(f"image_{index}.png", buffer.getvalue())
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
    assert "model_adaptation_size" not in params["image_size_adaptation"]


def test_prepare_config_records_metaformer_model_size_adaptation(tmp_path):
    (
        _image_path_col,
        _label_col,
        _split_col,
        LudwigDirectBackend,
    ) = _load_backend_test_objects()
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
    config = yaml.safe_load(yaml_str)
    input_feature = config["input_features"][0]

    assert input_feature["preprocessing"]["height"] == 96
    assert input_feature["preprocessing"]["width"] == 96
    assert input_feature["encoder"]["height"] == 96
    assert input_feature["encoder"]["width"] == 96
    assert params["image_size_adaptation"]["original_size"] == "96x96"
    assert params["image_size_adaptation"]["training_size"] == "96x96"
    assert params["image_size_adaptation"]["model_configured_size"] == "384x384"
    assert params["image_size_adaptation"]["model_adaptation_size"] == "224x224"
