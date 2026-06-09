import sys
import types
from pathlib import Path

import pandas as pd


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

from constants import IMAGE_PATH_COLUMN_NAME, LABEL_COLUMN_NAME, SPLIT_COLUMN_NAME
from ludwig_backend import LudwigDirectBackend


def test_generate_validation_predictions_writes_split_specific_file(tmp_path):
    exp_dir = tmp_path / "experiment_run"
    (exp_dir / "model").mkdir(parents=True)
    data_dir = tmp_path / "prepared"
    data_dir.mkdir()
    prepared_csv = data_dir / "prepared_data.csv"
    pd.DataFrame(
        {
            IMAGE_PATH_COLUMN_NAME: [
                "images/train.jpg",
                "images/val_a.jpg",
                "images/val_b.jpg",
                "images/test.jpg",
            ],
            LABEL_COLUMN_NAME: [0, 1, 0, 1],
            SPLIT_COLUMN_NAME: [0, 1, 1, 2],
        }
    ).to_csv(prepared_csv, index=False)

    class StubLudwigModel:
        @classmethod
        def load(cls, model_dir):
            assert model_dir == str(exp_dir / "model")
            return cls()

        def predict(self, dataset=None, **_kwargs):
            assert len(dataset) == 2
            assert all(Path(path).is_absolute() for path in dataset[IMAGE_PATH_COLUMN_NAME])
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
    assert df[SPLIT_COLUMN_NAME].tolist() == [1, 1]
    assert df[LABEL_COLUMN_NAME].tolist() == [1, 0]
    assert df["label_probabilities_1"].tolist() == [0.8, 0.3]
