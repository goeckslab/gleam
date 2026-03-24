from types import SimpleNamespace

from report_utils import _build_runtime_metadata_rows


def test_build_runtime_metadata_rows_includes_seed_flags_and_resources():
    args = SimpleNamespace(random_seed=42, deterministic=True)
    ag_config = {
        "fit": {"seed": 42},
        "hyperparameters": {
            "env": {
                "seed": 42,
                "num_gpus": 2,
                "num_workers": 6,
                "num_workers_inference": 3,
            }
        },
    }

    rows = dict(
        _build_runtime_metadata_rows(
            cfg={},
            args=args,
            ag_config=ag_config,
        )
    )

    assert rows["Random seed"] == 42
    assert rows["Deterministic mode"] is True
    assert rows["GPU count"] == 2
    assert rows["Training worker count"] == 6
    assert rows["Evaluation worker count"] == 3
