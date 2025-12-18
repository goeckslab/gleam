# Multimodal Learner (AutoGluon)

Train and evaluate AutoGluon’s multimodal models inside Galaxy, combining tabular features with optional text and image columns. The tool targets both classification and regression tasks and outputs reproducible reports and configurations.

## Capabilities
- Ingests CSV/TSV data with a required target column; numeric, categorical, and free-text columns are detected automatically.
- Adds image modality support: supply one or more ZIP archives containing the images referenced in your table, and choose a vision backbone.
- Offers quality presets, time limits, deterministic mode, and cross-validation or custom splits when no external test set is provided.
- Lets you pick text backbones, adjust epochs/learning rate/batch size, and pass additional AutoGluon hyperparameters.
- Handles missing images via configurable strategies and produces transparent metrics plus plots in an interactive HTML report.

## Inputs
- `Training dataset (CSV/TSV)`: includes the label column and any feature columns; image columns should contain file paths that exist in the provided ZIP archives (or absolute paths).
- Optional `Test dataset (CSV/TSV)`: if omitted, the tool performs train/validation/test splitting or k-fold CV.
- Optional `Image archive(s) (ZIP)`: one or more archives containing the image files referenced in the table.
- Optional overrides: text and image backbones, evaluation metric, quality preset, threshold for binary tasks, and extra hyperparameters (JSON/YAML string or file path).

## Outputs
- `output_html`: interactive training/evaluation report with metrics and visualizations across the available splits.
- `output_json`: machine-readable summary of metrics (train/val/test or cross-validation folds).
- `output_config`: YAML config capturing the effective AutoGluon settings used for the run.

## Typical Galaxy usage
1) Upload your training CSV/TSV (and optional test split) plus any ZIP archives that hold referenced images.
2) Select the target/label column, choose text and image backbones as needed, and pick a quality preset or time limit.
3) Run the tool to obtain an HTML report, metrics JSON, and reproducible config for downstream prediction or auditing.
