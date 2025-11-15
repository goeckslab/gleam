import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Tuple

import pandas as pd
import pandas.api.types as ptypes
import yaml

from constants import (
    IMAGE_PATH_COLUMN_NAME,
    LABEL_COLUMN_NAME,
    MODEL_ENCODER_TEMPLATES,
)
from html_structure import (
    build_tabbed_html,
    encode_image_to_base64,
    format_config_table_html,
    format_stats_table_html,
    format_test_merged_stats_table_html,
    format_train_val_stats_table_html,
    get_html_closing,
    get_html_template,
    get_metrics_help_modal,
)
from ludwig.globals import (
    DESCRIPTION_FILE_NAME,
    PREDICTIONS_PARQUET_FILE_NAME,
    TEST_STATISTICS_FILE_NAME,
    TRAIN_SET_METADATA_FILE_NAME,
)
from ludwig.utils.data_utils import get_split_path
from metaformer_setup import META_DEFAULT_CFGS, get_visualizations_registry
from plotly_plots import build_classification_plots
from utils import detect_output_type, extract_metrics_from_json

logger = logging.getLogger("ImageLearner")


class Backend(Protocol):
    """Interface for a machine learning backend."""

    def prepare_config(
        self,
        config_params: Dict[str, Any],
        split_config: Dict[str, Any],
    ) -> str:
        ...

    def run_experiment(
        self,
        dataset_path: Path,
        config_path: Path,
        output_dir: Path,
        random_seed: int,
    ) -> None:
        ...

    def generate_plots(self, output_dir: Path) -> None:
        ...

    def generate_html_report(
        self,
        title: str,
        output_dir: str,
        config: Dict[str, Any],
        split_info: str,
    ) -> Path:
        ...


class LudwigDirectBackend:
    """Backend for running Ludwig experiments directly via the internal experiment_cli function."""

    def _detect_image_dimensions(self, image_zip_path: str) -> Tuple[int, int]:
        """Detect image dimensions from the first image in the dataset."""
        try:
            import zipfile
            from PIL import Image
            import io

            # Check if image_zip is provided
            if not image_zip_path:
                logger.warning("No image zip provided, using default 224x224")
                return 224, 224

            # Extract first image to detect dimensions
            with zipfile.ZipFile(image_zip_path, 'r') as z:
                image_files = [f for f in z.namelist() if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if not image_files:
                    logger.warning("No image files found in zip, using default 224x224")
                    return 224, 224

                # Check first image
                with z.open(image_files[0]) as f:
                    img = Image.open(io.BytesIO(f.read()))
                    width, height = img.size
                    logger.info(f"Detected image dimensions: {width}x{height}")
                    return height, width  # Return as (height, width) to match encoder config

        except Exception as e:
            logger.warning(f"Error detecting image dimensions: {e}, using default 224x224")
            return 224, 224

    def prepare_config(
        self,
        config_params: Dict[str, Any],
        split_config: Dict[str, Any],
    ) -> str:
        logger.info("LudwigDirectBackend: Preparing YAML configuration.")

        model_name = config_params.get("model_name", "resnet18")
        use_pretrained = config_params.get("use_pretrained", False)
        fine_tune = config_params.get("fine_tune", False)
        if use_pretrained:
            trainable = bool(fine_tune)
        else:
            trainable = True
        epochs = config_params.get("epochs", 10)
        batch_size = config_params.get("batch_size")
        num_processes = config_params.get("preprocessing_num_processes", 1)
        early_stop = config_params.get("early_stop", None)
        learning_rate = config_params.get("learning_rate")
        learning_rate = "auto" if learning_rate is None else float(learning_rate)
        raw_encoder = MODEL_ENCODER_TEMPLATES.get(model_name, model_name)

        # --- MetaFormer detection and config logic ---
        def _is_metaformer(name: str) -> bool:
            return isinstance(name, str) and name.startswith(
                (
                    "identityformer_",
                    "randformer_",
                    "poolformerv2_",
                    "convformer_",
                    "caformer_",
                )
            )

        # Check if this is a MetaFormer model (either direct name or in custom_model)
        is_metaformer = (
            _is_metaformer(model_name)
            or (isinstance(raw_encoder, dict) and "custom_model" in raw_encoder and _is_metaformer(raw_encoder["custom_model"]))
        )

        metaformer_resize: Optional[Tuple[int, int]] = None
        metaformer_channels = 3

        if is_metaformer:
            # Handle MetaFormer models
            custom_model = None
            if isinstance(raw_encoder, dict) and "custom_model" in raw_encoder:
                custom_model = raw_encoder["custom_model"]
            else:
                custom_model = model_name

            logger.info(f"DETECTED MetaFormer model: {custom_model}")
            cfg_channels, cfg_height, cfg_width = 3, 224, 224
            if META_DEFAULT_CFGS:
                model_cfg = META_DEFAULT_CFGS.get(custom_model, {})
                input_size = model_cfg.get("input_size")
                if isinstance(input_size, (list, tuple)) and len(input_size) == 3:
                    cfg_channels, cfg_height, cfg_width = (
                        int(input_size[0]),
                        int(input_size[1]),
                        int(input_size[2]),
                    )

            target_height, target_width = cfg_height, cfg_width
            resize_value = config_params.get("image_resize")
            if resize_value and resize_value != "original":
                try:
                    dimensions = resize_value.split("x")
                    if len(dimensions) == 2:
                        target_height, target_width = int(dimensions[0]), int(dimensions[1])
                        if target_height <= 0 or target_width <= 0:
                            raise ValueError(
                                f"Image resize must be positive integers, received {resize_value}."
                            )
                        logger.info(f"MetaFormer explicit resize: {target_height}x{target_width}")
                    else:
                        raise ValueError(resize_value)
                except (ValueError, IndexError):
                    logger.warning(
                        "Invalid image resize format '%s'; falling back to model default %sx%s",
                        resize_value,
                        cfg_height,
                        cfg_width,
                    )
                    target_height, target_width = cfg_height, cfg_width
            else:
                image_zip_path = config_params.get("image_zip", "")
                detected_height, detected_width = self._detect_image_dimensions(image_zip_path)
                if use_pretrained:
                    if (detected_height, detected_width) != (cfg_height, cfg_width):
                        logger.info(
                            "MetaFormer pretrained weights expect %sx%s; resizing from detected %sx%s",
                            cfg_height,
                            cfg_width,
                            detected_height,
                            detected_width,
                        )
                else:
                    target_height, target_width = detected_height, detected_width
                if target_height <= 0 or target_width <= 0:
                    raise ValueError(
                        f"Invalid detected image dimensions for MetaFormer: {target_height}x{target_width}."
                    )

            metaformer_channels = cfg_channels
            metaformer_resize = (target_height, target_width)

            encoder_config = {
                "type": "stacked_cnn",
                "height": target_height,
                "width": target_width,
                "num_channels": metaformer_channels,
                "output_size": 128,
                "use_pretrained": use_pretrained,
                "trainable": trainable,
                "custom_model": custom_model,
            }

        elif isinstance(raw_encoder, dict):
            # Handle image resize for regular encoders
            # Note: Standard encoders like ResNet don't support height/width parameters
            # Resize will be handled at the preprocessing level by Ludwig
            if config_params.get("image_resize") and config_params["image_resize"] != "original":
                logger.info(f"Resize requested: {config_params['image_resize']} for standard encoder. Resize will be handled at preprocessing level.")

            encoder_config = {
                **raw_encoder,
                "use_pretrained": use_pretrained,
                "trainable": trainable,
            }
        else:
            encoder_config = {"type": raw_encoder}

        batch_size_cfg = batch_size or "auto"

        label_column_path = config_params.get("label_column_data_path")
        label_series = None
        num_unique_labels = 2
        numeric_binary_labels = False
        likely_regression = False
        if label_column_path is not None and Path(label_column_path).exists():
            try:
                label_series = pd.read_csv(label_column_path)[LABEL_COLUMN_NAME]
                non_na = label_series.dropna()
                if not non_na.empty:
                    num_unique_labels = non_na.nunique()
                is_numeric = ptypes.is_numeric_dtype(label_series.dtype)
                numeric_binary_labels = is_numeric and num_unique_labels == 2
                likely_regression = (
                    is_numeric and not numeric_binary_labels and num_unique_labels > 10
                )
                if numeric_binary_labels:
                    logger.info(
                        "Detected numeric binary labels in '%s'; configuring Ludwig for binary classification.",
                        LABEL_COLUMN_NAME,
                    )
            except Exception as e:
                logger.warning(f"Could not read label column for task detection: {e}")

        if numeric_binary_labels:
            task_type = "classification"
        elif likely_regression:
            task_type = "regression"
        else:
            task_type = "classification"

        if task_type == "regression" and numeric_binary_labels:
            logger.warning(
                "Numeric binary labels detected but regression task chosen; forcing classification to avoid invalid Ludwig config."
            )
            task_type = "classification"

        config_params["task_type"] = task_type

        image_feat: Dict[str, Any] = {
            "name": IMAGE_PATH_COLUMN_NAME,
            "type": "image",
        }
        # Set preprocessing dimensions FIRST for MetaFormer models
        if is_metaformer:
            if metaformer_resize is None:
                metaformer_resize = (224, 224)
            height, width = metaformer_resize

            # CRITICAL: Set preprocessing dimensions FIRST for MetaFormer models
            # This is essential for MetaFormer models to work properly
            if "preprocessing" not in image_feat:
                image_feat["preprocessing"] = {}
            image_feat["preprocessing"]["height"] = height
            image_feat["preprocessing"]["width"] = width
            # Use infer_image_dimensions=True to allow Ludwig to read images for validation
            # but set explicit max dimensions to control the output size
            image_feat["preprocessing"]["infer_image_dimensions"] = True
            image_feat["preprocessing"]["infer_image_max_height"] = height
            image_feat["preprocessing"]["infer_image_max_width"] = width
            image_feat["preprocessing"]["num_channels"] = metaformer_channels
            image_feat["preprocessing"]["resize_method"] = "interpolate"  # Use interpolation for better quality
            image_feat["preprocessing"]["standardize_image"] = "imagenet1k"  # Use ImageNet standardization
            # Force Ludwig to respect our dimensions by setting additional parameters
            image_feat["preprocessing"]["requires_equal_dimensions"] = False
            logger.info(f"Set preprocessing dimensions for MetaFormer: {height}x{width} (infer_dimensions=True with max dimensions to allow validation)")
        # Now set the encoder configuration
        image_feat["encoder"] = encoder_config

        if config_params.get("augmentation") is not None:
            image_feat["augmentation"] = config_params["augmentation"]

        # Add resize configuration for standard encoders (ResNet, etc.)
        # FIXED: MetaFormer models now respect user dimensions completely
        # Previously there was a double resize issue where MetaFormer would force 224x224
        # Now both MetaFormer and standard encoders respect user's resize choice
        if (not is_metaformer) and config_params.get("image_resize") and config_params["image_resize"] != "original":
            try:
                dimensions = config_params["image_resize"].split("x")
                if len(dimensions) == 2:
                    height, width = int(dimensions[0]), int(dimensions[1])
                    if height <= 0 or width <= 0:
                        raise ValueError(
                            f"Image resize must be positive integers, received {config_params['image_resize']}."
                        )

                    # Add resize to preprocessing for standard encoders
                    if "preprocessing" not in image_feat:
                        image_feat["preprocessing"] = {}
                    image_feat["preprocessing"]["height"] = height
                    image_feat["preprocessing"]["width"] = width
                    # Use infer_image_dimensions=True to allow Ludwig to read images for validation
                    # but set explicit max dimensions to control the output size
                    image_feat["preprocessing"]["infer_image_dimensions"] = True
                    image_feat["preprocessing"]["infer_image_max_height"] = height
                    image_feat["preprocessing"]["infer_image_max_width"] = width
                    logger.info(f"Added resize preprocessing: {height}x{width} for standard encoder with infer_image_dimensions=True and max dimensions")
            except (ValueError, IndexError):
                logger.warning(f"Invalid image resize format: {config_params['image_resize']}, skipping resize preprocessing")
        if task_type == "regression":
            output_feat = {
                "name": LABEL_COLUMN_NAME,
                "type": "number",
                "decoder": {"type": "regressor"},
                "loss": {"type": "mean_squared_error"},
            }
            val_metric = config_params.get("validation_metric", "mean_squared_error")

        else:
            output_type = "binary" if num_unique_labels == 2 else "category"
            if num_unique_labels == 2:
                output_feat = {
                    "name": LABEL_COLUMN_NAME,
                    "type": "binary",
                    "decoder": {"type": "classifier"},
                    "loss": {"type": "sigmoid_cross_entropy"},
                }
            else:
                output_feat = {
                    "name": LABEL_COLUMN_NAME,
                    "type": "category",
                    "decoder": {"type": "classifier"},
                    "loss": {"type": "softmax_cross_entropy"},
                }
            if output_type == "binary" and config_params.get("threshold") is not None:
                output_feat["threshold"] = float(config_params["threshold"])
            val_metric = None

        conf: Dict[str, Any] = {
            "model_type": "ecd",
            "input_features": [image_feat],
            "output_features": [output_feat],
            "combiner": {"type": "concat"},
            "trainer": {
                "epochs": epochs,
                "early_stop": early_stop,
                "batch_size": batch_size_cfg,
                "learning_rate": learning_rate,
                # only set validation_metric for regression
                **({"validation_metric": val_metric} if val_metric else {}),
            },
            "preprocessing": {
                "split": split_config,
                "num_processes": num_processes,
                "in_memory": False,
            },
        }

        logger.debug("LudwigDirectBackend: Config dict built.")
        try:
            yaml_str = yaml.dump(conf, sort_keys=False, indent=2)
            logger.info("LudwigDirectBackend: YAML config generated.")
            return yaml_str
        except Exception:
            logger.error(
                "LudwigDirectBackend: Failed to serialize YAML.",
                exc_info=True,
            )
            raise

    def run_experiment(
        self,
        dataset_path: Path,
        config_path: Path,
        output_dir: Path,
        random_seed: int = 42,
    ) -> None:
        """Invoke Ludwig's internal experiment_cli function to run the experiment."""
        logger.info("LudwigDirectBackend: Starting experiment execution.")

        try:
            from ludwig.experiment import experiment_cli
        except ImportError as e:
            logger.error(
                "LudwigDirectBackend: Could not import experiment_cli.",
                exc_info=True,
            )
            raise RuntimeError("Ludwig import failed.") from e

        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            experiment_cli(
                dataset=str(dataset_path),
                config=str(config_path),
                output_directory=str(output_dir),
                random_seed=random_seed,
                skip_preprocessing=True,
            )
            logger.info(
                f"LudwigDirectBackend: Experiment completed. Results in {output_dir}"
            )
        except TypeError as e:
            logger.error(
                "LudwigDirectBackend: Argument mismatch in experiment_cli call.",
                exc_info=True,
            )
            raise RuntimeError("Ludwig argument error.") from e
        except Exception:
            logger.error(
                "LudwigDirectBackend: Experiment execution error.",
                exc_info=True,
            )
            raise

    def get_training_process(self, output_dir) -> Optional[Dict[str, Any]]:
        """Retrieve the learning rate used in the most recent Ludwig run."""
        output_dir = Path(output_dir)
        exp_dirs = sorted(
            output_dir.glob("experiment_run*"),
            key=lambda p: p.stat().st_mtime,
        )

        if not exp_dirs:
            logger.warning(f"No experiment run directories found in {output_dir}")
            return None

        progress_file = exp_dirs[-1] / "model" / "training_progress.json"
        if not progress_file.exists():
            logger.warning(f"No training_progress.json found in {progress_file}")
            return None

        try:
            with progress_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return {
                "learning_rate": data.get("learning_rate"),
                "batch_size": data.get("batch_size"),
                "epoch": data.get("epoch"),
            }
        except Exception as e:
            logger.warning(f"Failed to read training progress info: {e}")
            return {}

    def convert_parquet_to_csv(self, output_dir: Path):
        """Convert the predictions Parquet file to CSV."""
        output_dir = Path(output_dir)
        exp_dirs = sorted(
            output_dir.glob("experiment_run*"),
            key=lambda p: p.stat().st_mtime,
        )
        if not exp_dirs:
            logger.warning(f"No experiment run dirs found in {output_dir}")
            return
        exp_dir = exp_dirs[-1]
        parquet_path = exp_dir / PREDICTIONS_PARQUET_FILE_NAME
        csv_path = exp_dir / "predictions.csv"

        # Check if parquet file exists before trying to convert
        if not parquet_path.exists():
            logger.info(f"Predictions parquet file not found at {parquet_path}, skipping conversion")
            return

        try:
            df = pd.read_parquet(parquet_path)
            df.to_csv(csv_path, index=False)
            logger.info(f"Converted Parquet to CSV: {csv_path}")
        except Exception as e:
            logger.error(f"Error converting Parquet to CSV: {e}")

    def generate_plots(self, output_dir: Path) -> None:
        """Generate all registered Ludwig visualizations for the latest experiment run."""
        logger.info("Generating all Ludwig visualizations…")

        test_plots = {
            "compare_performance",
            "compare_classifiers_performance_from_prob",
            "compare_classifiers_performance_from_pred",
            "compare_classifiers_performance_changing_k",
            "compare_classifiers_multiclass_multimetric",
            "compare_classifiers_predictions",
            "confidence_thresholding_2thresholds_2d",
            "confidence_thresholding_2thresholds_3d",
            "confidence_thresholding",
            "confidence_thresholding_data_vs_acc",
            "binary_threshold_vs_metric",
            "roc_curves",
            "roc_curves_from_test_statistics",
            "calibration_1_vs_all",
            "calibration_multiclass",
            "confusion_matrix",
            "frequency_vs_f1",
        }
        train_plots = {
            "learning_curves",
            "compare_classifiers_performance_subset",
        }

        output_dir = Path(output_dir)
        exp_dirs = sorted(
            output_dir.glob("experiment_run*"),
            key=lambda p: p.stat().st_mtime,
        )
        if not exp_dirs:
            logger.warning(f"No experiment run dirs found in {output_dir}")
            return
        exp_dir = exp_dirs[-1]

        viz_dir = exp_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        train_viz = viz_dir / "train"
        test_viz = viz_dir / "test"
        train_viz.mkdir(parents=True, exist_ok=True)
        test_viz.mkdir(parents=True, exist_ok=True)

        def _check(p: Path) -> Optional[str]:
            return str(p) if p.exists() else None

        training_stats = _check(exp_dir / "training_statistics.json")
        test_stats = _check(exp_dir / TEST_STATISTICS_FILE_NAME)
        probs_path = _check(exp_dir / PREDICTIONS_PARQUET_FILE_NAME)
        gt_metadata = _check(exp_dir / "model" / TRAIN_SET_METADATA_FILE_NAME)

        dataset_path = None
        split_file = None
        desc = exp_dir / DESCRIPTION_FILE_NAME
        if desc.exists():
            with open(desc, "r") as f:
                cfg = json.load(f)
            dataset_path = _check(Path(cfg.get("dataset", "")))
            split_file = _check(Path(get_split_path(cfg.get("dataset", ""))))

        output_feature = ""
        if desc.exists():
            try:
                output_feature = cfg["config"]["output_features"][0]["name"]
            except Exception:
                pass
        if not output_feature and test_stats:
            with open(test_stats, "r") as f:
                stats = json.load(f)
            output_feature = next(iter(stats.keys()), "")

        viz_registry = get_visualizations_registry()
        for viz_name, viz_func in viz_registry.items():
            if viz_name in train_plots:
                viz_dir_plot = train_viz
            elif viz_name in test_plots:
                viz_dir_plot = test_viz
            else:
                continue

            try:
                viz_func(
                    training_statistics=[training_stats] if training_stats else [],
                    test_statistics=[test_stats] if test_stats else [],
                    probabilities=[probs_path] if probs_path else [],
                    output_feature_name=output_feature,
                    ground_truth_split=2,
                    top_n_classes=[0],
                    top_k=3,
                    ground_truth_metadata=gt_metadata,
                    ground_truth=dataset_path,
                    split_file=split_file,
                    output_directory=str(viz_dir_plot),
                    normalize=False,
                    file_format="png",
                )
                logger.info(f"✔ Generated {viz_name}")
            except Exception as e:
                logger.warning(f"✘ Skipped {viz_name}: {e}")

        logger.info(f"All visualizations written to {viz_dir}")

    def generate_html_report(
        self,
        title: str,
        output_dir: str,
        config: dict,
        split_info: str,
    ) -> Path:
        """Assemble an HTML report from visualizations under train_val/ and test/ folders."""
        cwd = Path.cwd()
        report_name = title.lower().replace(" ", "_") + "_report.html"
        report_path = cwd / report_name
        output_dir = Path(output_dir)
        output_type = None

        exp_dirs = sorted(
            output_dir.glob("experiment_run*"),
            key=lambda p: p.stat().st_mtime,
        )
        if not exp_dirs:
            raise RuntimeError(f"No 'experiment*' dirs found in {output_dir}")
        exp_dir = exp_dirs[-1]

        base_viz_dir = exp_dir / "visualizations"
        train_viz_dir = base_viz_dir / "train"
        test_viz_dir = base_viz_dir / "test"

        html = get_html_template()

        # Extra CSS & JS: center Plotly and enable CSV download for predictions table
        html += """
<style>
  /* Center Plotly figures (both wrapper and native classes) */
  .plotly-center { display: flex; justify-content: center; }
  .plotly-center .plotly-graph-div, .plotly-center .js-plotly-plot { margin: 0 auto !important; }
  .js-plotly-plot, .plotly-graph-div { margin-left: auto !important; margin-right: auto !important; }

  /* Download button for predictions table */
  .download-btn {
    padding: 8px 12px;
    border: 1px solid #4CAF50;
    background: #4CAF50;
    color: white;
    border-radius: 6px;
    cursor: pointer;
  }
  .download-btn:hover { filter: brightness(0.95); }
  .preds-controls {
    display: flex;
    justify-content: flex-end;
    gap: 8px;
    margin: 8px 0;
  }
</style>
<script>
  function tableToCSV(table){
    const rows = Array.from(table.querySelectorAll('tr'));
    return rows.map(row =>
      Array.from(row.querySelectorAll('th,td')).map(cell => {
        let text = cell.innerText.replace(/\\r?\\n|\\r/g,' ').trim();
        if (text.includes('"') || text.includes(',')) {
          text = '"' + text.replace(/"/g,'""') + '"';
        }
        return text;
      }).join(',')
    ).join('\\n');
  }
  document.addEventListener('DOMContentLoaded', function(){
    const btn = document.getElementById('downloadPredsCsv');
    if(btn){
      btn.addEventListener('click', function(){
        const tbl = document.querySelector('.predictions-table');
        if(!tbl){ alert('Predictions table not found.'); return; }
        const csv = tableToCSV(tbl);
        const blob = new Blob([csv], {type: 'text/csv;charset=utf-8;'});
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'ground_truth_vs_predictions.csv';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      });
    }
  });
</script>
"""
        html += f"<h1>{title}</h1>"

        metrics_html = ""
        train_val_metrics_html = ""
        test_metrics_html = ""
        try:
            train_stats_path = exp_dir / "training_statistics.json"
            test_stats_path = exp_dir / TEST_STATISTICS_FILE_NAME
            if train_stats_path.exists() and test_stats_path.exists():
                with open(train_stats_path) as f:
                    train_stats = json.load(f)
                with open(test_stats_path) as f:
                    test_stats = json.load(f)
                output_type = detect_output_type(test_stats)
                metrics_html = format_stats_table_html(train_stats, test_stats, output_type)
                train_val_metrics_html = format_train_val_stats_table_html(
                    train_stats, test_stats
                )
                test_metrics_html = format_test_merged_stats_table_html(
                    extract_metrics_from_json(train_stats, test_stats, output_type)[
                        "test"
                    ], output_type
                )
        except Exception as e:
            logger.warning(
                f"Could not load stats for HTML report: {type(e).__name__}: {e}"
            )

        config_html = ""
        training_progress = self.get_training_process(output_dir)
        try:
            config_html = format_config_table_html(
                config, split_info, training_progress, output_type
            )
        except Exception as e:
            logger.warning(f"Could not load config for HTML report: {e}")

        # ---------- image rendering with exclusions ----------
        def render_img_section(
            title: str,
            dir_path: Path,
            output_type: str = None,
            exclude_names: Optional[set] = None,
        ) -> str:
            if not dir_path.exists():
                return f"<h2>{title}</h2><p><em>Directory not found.</em></p>"

            exclude_names = exclude_names or set()

            imgs = list(dir_path.glob("*.png"))

            # Exclude ROC curves and standard confusion matrices (keep only entropy version)
            default_exclude = {
                # "roc_curves.png",  # Remove ROC curves from test tab
                "confusion_matrix__label_top5.png",  # Remove standard confusion matrix
                "confusion_matrix__label_top10.png",  # Remove duplicate
                "confusion_matrix__label_top6.png",   # Remove duplicate
                "confusion_matrix_entropy__label_top10.png",  # Keep only top5
                "confusion_matrix_entropy__label_top6.png",   # Keep only top5
            }

            imgs = [
                img
                for img in imgs
                if img.name not in default_exclude
                and img.name not in exclude_names
            ]

            if not imgs:
                return f"<h2>{title}</h2><p><em>No plots found.</em></p>"

            # Sort images by name for consistent ordering (works with string and numeric labels)
            imgs = sorted(imgs, key=lambda x: x.name)

            html_section = ""
            for img in imgs:
                b64 = encode_image_to_base64(str(img))
                img_title = img.stem.replace("_", " ").title()
                html_section += (
                    f"<h2 style='text-align: center;'>{img_title}</h2>"
                    f'<div class="plot" style="margin-bottom:20px;text-align:center;">'
                    f'<img src="data:image/png;base64,{b64}" '
                    f'style="max-width:90%;max-height:600px;border:1px solid #ddd;" />'
                    f"</div>"
                )
            return html_section

        tab1_content = config_html + metrics_html

        tab2_content = train_val_metrics_html + render_img_section(
            "Training and Validation Visualizations",
            train_viz_dir,
            output_type,
            exclude_names={
                "compare_classifiers_performance_from_prob.png",
                "roc_curves_from_prediction_statistics.png",
                "precision_recall_curves_from_prediction_statistics.png",
                "precision_recall_curve.png",
            },
        )

        # --- Predictions vs Ground Truth table (REGRESSION ONLY) ---
        preds_section = ""
        parquet_path = exp_dir / PREDICTIONS_PARQUET_FILE_NAME
        if output_type == "regression" and parquet_path.exists():
            try:
                # 1) load predictions from Parquet
                df_preds = pd.read_parquet(parquet_path).reset_index(drop=True)
                # assume the column containing your model's prediction is named "prediction"
                # or contains that substring:
                pred_col = next(
                    (c for c in df_preds.columns if "prediction" in c.lower()),
                    None,
                )
                if pred_col is None:
                    raise ValueError("No prediction column found in Parquet output")
                df_pred = df_preds[[pred_col]].rename(columns={pred_col: "prediction"})

                # 2) load ground truth for the test split from prepared CSV
                df_all = pd.read_csv(config["label_column_data_path"])
                df_gt = df_all[df_all[SPLIT_COLUMN_NAME] == 2][
                    LABEL_COLUMN_NAME
                ].reset_index(drop=True)
                # 3) concatenate side-by-side
                df_table = pd.concat([df_gt, df_pred], axis=1)
                df_table.columns = [LABEL_COLUMN_NAME, "prediction"]

                # 4) render as HTML
                preds_html = df_table.to_html(index=False, classes="predictions-table")
                preds_section = (
                    "<h2 style='text-align: center;'>Ground Truth vs. Predictions</h2>"
                    "<div class='preds-controls'>"
                    "<button id='downloadPredsCsv' class='download-btn'>Download CSV</button>"
                    "</div>"
                    "<div class='scroll-rows-30' style='overflow-x:auto; overflow-y:auto; max-height:900px; margin-bottom:20px;'>"
                    + preds_html
                    + "</div>"
                )
            except Exception as e:
                logger.warning(f"Could not build Predictions vs GT table: {e}")

        tab3_content = test_metrics_html + preds_section

        if output_type in ("binary", "category") and test_stats_path.exists():
            try:
                interactive_plots = build_classification_plots(
                    str(test_stats_path),
                    str(train_stats_path) if train_stats_path.exists() else None,
                )
                for plot in interactive_plots:
                    tab3_content += (
                        f"<h2 style='text-align: center;'>{plot['title']}</h2>"
                        f"<div class='plotly-center'>{plot['html']}</div>"
                    )
                logger.info(f"Generated {len(interactive_plots)} interactive Plotly plots")
            except Exception as e:
                logger.warning(f"Could not generate Plotly plots: {e}")

        # Add static TEST PNGs (with default dedupe/exclusions)
        tab3_content += render_img_section(
            "Test Visualizations", test_viz_dir, output_type
        )

        tabbed_html = build_tabbed_html(tab1_content, tab2_content, tab3_content)
        modal_html = get_metrics_help_modal()
        html += tabbed_html + modal_html + get_html_closing()

        try:
            with open(report_path, "w") as f:
                f.write(html)
            logger.info(f"HTML report generated at: {report_path}")
        except Exception as e:
            logger.error(f"Failed to write HTML report: {e}")
            raise

        return report_path
