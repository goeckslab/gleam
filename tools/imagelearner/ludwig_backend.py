import inspect
import json
import logging
import os
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

import pandas as pd
import pandas.api.types as ptypes
import yaml
from constants import (
    IMAGE_PATH_COLUMN_NAME,
    LABEL_COLUMN_NAME,
    MODEL_ENCODER_TEMPLATES,
    SPLIT_COLUMN_NAME,
)
from html_structure import (
    build_tabbed_html,
    encode_image_to_base64,
    format_config_table_html,
    format_dataset_overview_table,
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
from metaformer_setup import get_visualizations_registry, META_DEFAULT_CFGS
from plotly_plots import (
    build_binary_threshold_plot,
    build_classification_plots,
    build_multiclass_metric_plots,
    build_prediction_diagnostics,
    build_regression_test_plots,
    build_regression_train_val_plots,
    build_train_validation_plots,
)
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

    _torchvision_patched = False

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
            # Stash the model name for patched Stacked2DCNN in case Ludwig drops custom_model from kwargs
            try:
                from MetaFormer.metaformer_stacked_cnn import set_current_metaformer_model

                set_current_metaformer_model(custom_model)
            except Exception:
                logger.debug("Could not set current MetaFormer model hint; proceeding without global override")
            # Also pass via environment to survive process boundaries (e.g., Ray workers)
            os.environ["GLEAM_META_FORMER_MODEL"] = custom_model
            cfg_channels, cfg_height, cfg_width = 3, 224, 224
            model_cfg = {}
            if META_DEFAULT_CFGS:
                model_cfg = META_DEFAULT_CFGS.get(custom_model, {})
                input_size = model_cfg.get("input_size")
                if isinstance(input_size, (list, tuple)) and len(input_size) == 3:
                    cfg_channels, cfg_height, cfg_width = (
                        int(input_size[0]),
                        int(input_size[1]),
                        int(input_size[2]),
                    )

            weights_url = None
            if isinstance(model_cfg, dict):
                weights_url = model_cfg.get("url")
            logger.info(
                "MetaFormer cfg lookup: model=%s has_cfg=%s url=%s use_pretrained=%s",
                custom_model,
                bool(model_cfg),
                weights_url,
                use_pretrained,
            )
            if use_pretrained and not weights_url:
                logger.warning(
                    "MetaFormer pretrained requested for %s but no URL found in default cfgs; model will be randomly initialized",
                    custom_model,
                )

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
                target_height, target_width = detected_height, detected_width
                if use_pretrained and (detected_height, detected_width) != (cfg_height, cfg_width):
                    logger.info(
                        "MetaFormer pretrained weights expect %sx%s; proceeding with detected %sx%s",
                        cfg_height,
                        cfg_width,
                        detected_height,
                        detected_width,
                    )
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

        # Set a human-friendly architecture string for reporting
        arch_display = None
        if is_metaformer and custom_model:
            arch_display = str(custom_model)
        elif isinstance(raw_encoder, dict):
            enc_type = raw_encoder.get("type")
            enc_variant = raw_encoder.get("model_variant")
            if enc_type:
                base = str(enc_type).replace("_", " ").title()
                arch_display = f"{base} {enc_variant}" if enc_variant is not None else base
        else:
            arch_display = str(raw_encoder).replace("_", " ").title()

        if not arch_display:
            arch_display = str(model_name)
        config_params["architecture"] = arch_display

        batch_size_cfg = batch_size or "auto"

        label_column_path = config_params.get("label_column_data_path")
        label_series = None
        label_metadata_hint = config_params.get("label_metadata") or {}
        output_type_hint = config_params.get("output_type_hint")
        num_unique_labels = int(label_metadata_hint.get("num_unique", 2))
        numeric_binary_labels = bool(label_metadata_hint.get("is_numeric_binary", False))
        likely_regression = bool(label_metadata_hint.get("likely_regression", False))
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

        if output_type_hint == "binary":
            num_unique_labels = 2
            numeric_binary_labels = numeric_binary_labels or bool(
                label_metadata_hint.get("is_numeric", False)
            )

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
            config_params["image_size"] = f"{height}x{width}"
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
                    config_params["image_size"] = f"{height}x{width}"
            except (ValueError, IndexError):
                logger.warning(f"Invalid image resize format: {config_params['image_resize']}, skipping resize preprocessing")
        elif not is_metaformer:
            # No explicit resize provided; keep for reporting purposes
            config_params.setdefault("image_size", "original")

        def _resolve_validation_metric(
            task: str, requested: Optional[str], output_feature: Dict[str, Any]
        ) -> Optional[str]:
            """
            Pick a validation metric that Ludwig will accept for the resolved task/output.
            If the requested metric is invalid, fall back to a safe option or omit it entirely.
            """
            default_map = {
                "regression": "mean_squared_error",
                "binary": "roc_auc",
                "category": "accuracy",
            }
            # Safe defaults when Ludwig's registry isn't available.
            safe_allowed_map = {
                "regression": {
                    "mean_absolute_error",
                    "mean_squared_error",
                    "root_mean_squared_error",
                    "root_mean_squared_percentage_error",
                    "loss",
                },
                "binary": {
                    "roc_auc",
                    "accuracy",
                    "precision",
                    "recall",
                    "loss",
                },
                "category": {
                    "accuracy",
                    "loss",
                },
            }
            alias_map = {
                "regression": {
                    "mae": "mean_absolute_error",
                    "mse": "mean_squared_error",
                    "rmse": "root_mean_squared_error",
                    "rmspe": "root_mean_squared_percentage_error",
                },
                "category": {},
                "binary": {
                    "roc_auc": "roc_auc",
                },
            }

            default_metric = default_map.get(task)
            metric = requested or default_metric
            if metric is None:
                return None

            metric = alias_map.get(task, {}).get(metric, metric)

            # Prefer Ludwig's own metric registry when available; intersect with known-safe sets.
            registry_metrics = None
            try:
                from ludwig.features.feature_registries import output_type_registry

                feature_cls = output_type_registry.get(output_feature.get("type"))
                if feature_cls:
                    feature_obj = feature_cls(feature=output_feature)
                    metrics_attr = getattr(feature_obj, "metric_functions", None) or getattr(
                        feature_obj, "metrics", None
                    )
                    if isinstance(metrics_attr, dict):
                        registry_metrics = set(metrics_attr.keys())
                    elif isinstance(metrics_attr, (list, tuple, set)):
                        extracted = set()
                        for item in metrics_attr:
                            if isinstance(item, str):
                                extracted.add(item)
                            elif hasattr(item, "name"):
                                extracted.add(str(item.name))
                            elif hasattr(item, "__name__"):
                                extracted.add(str(item.__name__))
                        registry_metrics = extracted or None
            except Exception as exc:
                logger.debug(
                    "Could not inspect Ludwig metrics for output type %s: %s",
                    output_feature.get("type"),
                    exc,
                )

            allowed = set(safe_allowed_map.get(task, set()))
            if registry_metrics is not None:
                # Use Ludwig's registry when available; fall back to safe defaults if it's empty.
                allowed = registry_metrics or allowed

            if allowed and metric not in allowed:
                fallback_candidates = [
                    default_metric if default_metric in allowed else None,
                    "loss" if "loss" in allowed else None,
                    next(iter(allowed), None),
                ]
                fallback = next((m for m in fallback_candidates if m in allowed), None)
                if requested:
                    logger.warning(
                        "Validation metric '%s' is not supported for %s outputs; %s",
                        requested,
                        task,
                        (f"using '{fallback}' instead." if fallback else "omitting validation_metric."),
                    )
                metric = fallback

            return metric

        if task_type == "regression":
            output_feat = {
                "name": LABEL_COLUMN_NAME,
                "type": "number",
                "decoder": {"type": "regressor"},
                "loss": {"type": "mean_squared_error"},
            }
            val_metric = _resolve_validation_metric(
                "regression",
                config_params.get("validation_metric"),
                output_feat,
            )

        else:
            if num_unique_labels == 2:
                output_feat = {
                    "name": LABEL_COLUMN_NAME,
                    "type": "binary",
                    "loss": {"type": "binary_weighted_cross_entropy"},
                }
                if config_params.get("threshold") is not None:
                    output_feat["threshold"] = float(config_params["threshold"])
            else:
                output_feat = {
                    "name": LABEL_COLUMN_NAME,
                    "type": "category",
                    "loss": {"type": "softmax_cross_entropy"},
                }
            val_metric = _resolve_validation_metric(
                "binary" if num_unique_labels == 2 else "category",
                config_params.get("validation_metric"),
                output_feat,
            )

        # Propagate the resolved validation metric (including any task-based fallback or alias normalization)
        config_params["validation_metric"] = val_metric

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
                # set validation_metric when provided
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

    def _patch_torchvision_download(self) -> None:
        """
        Torchvision weight downloads sometimes fail checksum validation behind
        corporate proxies that rewrite binaries. Skip hash checking to allow
        pre-trained weights to load in those environments.
        """
        if LudwigDirectBackend._torchvision_patched:
            return
        try:
            import torch.hub as torch_hub

            original = torch_hub.load_state_dict_from_url
            original_download = torch_hub.download_url_to_file

            def _no_hash(url, model_dir=None, map_location=None, progress=True, check_hash=False, file_name=None):
                return original(
                    url,
                    model_dir=model_dir,
                    map_location=map_location,
                    progress=progress,
                    check_hash=False,
                    file_name=file_name,
                )

            def _download_no_hash(url, dst, hash_prefix=None, progress=True):
                # Torchvision's download_url_to_file signature does not accept check_hash in older versions.
                return original_download(url, dst, hash_prefix=None, progress=progress)

            torch_hub.load_state_dict_from_url = _no_hash  # type: ignore[assignment]
            torch_hub.download_url_to_file = _download_no_hash  # type: ignore[assignment]
            LudwigDirectBackend._torchvision_patched = True
            logger.info("Disabled torchvision weight hash verification to avoid proxy-corrupted downloads.")
        except Exception as exc:
            logger.warning(f"Could not patch torchvision download hash check: {exc}")

    def run_experiment(
        self,
        dataset_path: Path,
        config_path: Path,
        output_dir: Path,
        random_seed: int = 42,
    ) -> None:
        """Invoke Ludwig's internal experiment_cli function to run the experiment."""
        logger.info("LudwigDirectBackend: Starting experiment execution.")

        # Avoid strict hash validation for torchvision weights (common in proxied environments)
        self._patch_torchvision_download()

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
                "LudwigDirectBackend: Experiment execution error. "
                "If this relates to validation_metric, confirm the XML task selection "
                "passes a metric that matches the inferred task type.",
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

    def _get_latest_experiment_dir(self, output_dir: Path) -> Optional[Path]:
        """Return the most recent experiment_run* directory, if present."""
        output_dir = Path(output_dir)
        exp_dirs = sorted(
            output_dir.glob("experiment_run*"),
            key=lambda p: p.stat().st_mtime,
        )
        return exp_dirs[-1] if exp_dirs else None

    def _extract_preprocessing_config(
        self, exp_dir: Path, config: Dict[str, Any]
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Path]]:
        """Parse Ludwig preprocessing settings from train_set_metadata or description."""
        image_meta: Dict[str, Any] = {}
        meta_path = exp_dir / "model" / TRAIN_SET_METADATA_FILE_NAME
        if meta_path.exists():
            try:
                with meta_path.open("r", encoding="utf-8") as f:
                    meta_json = json.load(f)
                image_list = meta_json.get("input_features") or []
                if image_list:
                    image_meta = image_list[0] or {}
            except Exception as exc:
                logger.warning("Unable to read train_set_metadata: %s", exc)

        # Fallback to description config for preprocessing hints
        desc_cfg: Dict[str, Any] = {}
        desc_path = exp_dir / DESCRIPTION_FILE_NAME
        if desc_path.exists():
            try:
                with desc_path.open("r", encoding="utf-8") as f:
                    desc_json = json.load(f)
                desc_cfg = desc_json.get("config", {}) if isinstance(desc_json, dict) else {}
            except Exception as exc:
                logger.warning("Unable to read description.json for preprocessing: %s", exc)

        preprocessing = {}
        if isinstance(image_meta, dict):
            preprocessing = image_meta.get("preprocessing") or {}
        if not preprocessing and desc_cfg:
            try:
                preprocessing = (
                    desc_cfg.get("input_features", [{}])[0].get("preprocessing") or {}
                )
            except Exception:
                preprocessing = {}

        # If height/width are missing but max inferred dimensions exist, use them as fallback
        if isinstance(preprocessing, dict):
            if not preprocessing.get("height") and preprocessing.get("infer_image_max_height"):
                preprocessing["height"] = preprocessing.get("infer_image_max_height")
            if not preprocessing.get("width") and preprocessing.get("infer_image_max_width"):
                preprocessing["width"] = preprocessing.get("infer_image_max_width")

        # Keep label path for downstream sampling
        label_path = None
        try:
            label_path_cfg = config.get("label_column_data_path")
            if label_path_cfg:
                label_path = Path(label_path_cfg)
        except Exception:
            label_path = None

        return preprocessing if isinstance(preprocessing, dict) else {}, label_path

    def _find_last_conv_layer(self, encoder_obj: Any) -> Optional[Any]:
        """Identify the last Conv2d layer within the encoder."""
        try:
            import torch.nn as nn
        except Exception:
            return None

        target_model = encoder_obj
        if hasattr(encoder_obj, "model"):
            target_model = encoder_obj.model

        try:
            modules = list(target_model.named_modules())
        except Exception:
            return None

        for _, module in reversed(modules):
            if isinstance(module, nn.Conv2d):
                return module
        return None

    def _generate_gradcam_heatmaps(
        self,
        exp_dir: Path,
        config: Dict[str, Any],
        output_type: Optional[str],
    ) -> Dict[str, Any]:
        """Compute Grad-CAM overlays for convolutional encoders, when possible."""
        result: Dict[str, Any] = {
            "status": "skipped",
            "reason": "",
            "preview_paths": [],
            "zip_path": None,
            "dir_path": None,
        }

        try:
            import numpy as np
            import torch
            import torch.nn.functional as F
            from matplotlib import cm
            from PIL import Image
            from ludwig.api import LudwigModel
        except Exception as exc:
            result["reason"] = f"Missing dependency for Grad-CAM: {exc}"
            return result

        exp_dir = Path(exp_dir)
        model_dir = exp_dir / "model"
        if not model_dir.exists():
            result["reason"] = "Model directory not found; skipping Grad-CAM."
            return result

        preprocessing, label_path = self._extract_preprocessing_config(exp_dir, config)
        height = preprocessing.get("height")
        width = preprocessing.get("width")
        if not height or not width:
            result["reason"] = "Image resize/height not found in Ludwig preprocessing."
            return result

        label_csv = label_path if label_path and label_path.exists() else None
        if not label_csv:
            result["reason"] = "Prepared label CSV not available for Grad-CAM sampling."
            return result

        try:
            df_all = pd.read_csv(label_csv)
        except Exception as exc:
            result["reason"] = f"Could not read prepared CSV: {exc}"
            return result

        if IMAGE_PATH_COLUMN_NAME not in df_all.columns:
            result["reason"] = "Image column missing from prepared CSV; cannot build Grad-CAM inputs."
            return result

        # Prefer test split; otherwise fall back to the full dataset
        df_candidates = df_all
        if SPLIT_COLUMN_NAME in df_all.columns:
            try:
                df_candidates = df_all[df_all[SPLIT_COLUMN_NAME] == 2]
                if df_candidates.empty:
                    df_candidates = df_all
            except Exception:
                df_candidates = df_all

        # Cap the number of samples
        df_candidates = df_candidates.head(12)
        if df_candidates.empty:
            result["reason"] = "No samples available for Grad-CAM generation."
            return result

        try:
            ludwig_model = LudwigModel.load(str(model_dir))
        except Exception as exc:
            result["reason"] = f"Unable to load LudwigModel for Grad-CAM: {exc}"
            return result

        base_model = getattr(ludwig_model, "model", None)
        if base_model is None:
            result["reason"] = "Ludwig model missing underlying torch model."
            return result

        image_feature_name = None
        image_feature = None
        try:
            for name, feat in getattr(base_model, "input_features", {}).items():
                if hasattr(feat, "encoder_obj"):
                    image_feature_name = name
                    image_feature = feat
                    break
        except Exception:
            image_feature_name = None

        if not image_feature or not image_feature_name:
            result["reason"] = "Image input feature not found; skipping Grad-CAM."
            return result

        target_layer = self._find_last_conv_layer(getattr(image_feature, "encoder_obj", None))
        if target_layer is None:
            result["reason"] = "No convolutional layer detected in the encoder (heatmaps unsupported)."
            return result

        standardize = preprocessing.get("standardize_image")
        mean = preprocessing.get("mean") or preprocessing.get("img_mean")
        std = preprocessing.get("std") or preprocessing.get("img_std")
        encoder_obj = getattr(image_feature, "encoder_obj", None)
        if hasattr(encoder_obj, "normalize_mean") and encoder_obj.normalize_mean:
            mean = encoder_obj.normalize_mean
        if hasattr(encoder_obj, "normalize_std") and encoder_obj.normalize_std:
            std = encoder_obj.normalize_std
        if isinstance(standardize, str) and standardize.lower() == "imagenet1k":
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        if mean is None or std is None:
            result["reason"] = "Normalization parameters (mean/std) not found in the saved encoder; skipping heatmaps to avoid mismatch."
            return result

        output_feature_name = LABEL_COLUMN_NAME
        try:
            if getattr(base_model, "output_features", None):
                output_feature_name = next(iter(base_model.output_features.keys()))
        except Exception:
            output_feature_name = LABEL_COLUMN_NAME

        device = torch.device("cpu")
        try:
            base_model.to(device)
            base_model.eval()
        except Exception:
            logger.debug("Could not move model to CPU for Grad-CAM; continuing on default device.")

        heatmap_dir = exp_dir / "feature_importance_examples"
        heatmap_dir.mkdir(parents=True, exist_ok=True)

        def _load_tensor(image_path: Path) -> Tuple[Optional[torch.Tensor], Optional[Image.Image]]:
            try:
                img = Image.open(image_path).convert("RGB")
            except Exception:
                return None, None
            resized = img.resize((int(width), int(height)))
            arr = np.asarray(resized).astype("float32") / 255.0
            arr = np.transpose(arr, (2, 0, 1))
            tensor = torch.from_numpy(arr)
            try:
                mean_tensor = torch.tensor(mean).view(-1, 1, 1)
                std_tensor = torch.tensor(std).view(-1, 1, 1)
                tensor = (tensor - mean_tensor) / std_tensor
            except Exception:
                return None, None
            return tensor.unsqueeze(0).to(device), resized

        generated: List[Path] = []
        pairs: List[Tuple[Path, Path]] = []
        image_root = label_csv.parent

        for _, row in df_candidates.iterrows():
            raw_path = row.get(IMAGE_PATH_COLUMN_NAME)
            if not isinstance(raw_path, str):
                continue
            abs_path = (image_root / raw_path).resolve()
            if not abs_path.exists():
                continue

            tensor, resized_img = _load_tensor(abs_path)
            if tensor is None or resized_img is None:
                continue

            activations: List[torch.Tensor] = []
            gradients: List[torch.Tensor] = []

            def _fwd_hook(_module, _inp, output):
                activations.append(output)

            def _bwd_hook(_module, _grad_in, grad_out):
                if grad_out and isinstance(grad_out[0], torch.Tensor):
                    gradients.append(grad_out[0])

            handle_fwd = target_layer.register_forward_hook(_fwd_hook)
            try:
                handle_bwd = target_layer.register_full_backward_hook(_bwd_hook)
            except Exception:
                handle_bwd = target_layer.register_backward_hook(_bwd_hook)

            try:
                base_model.zero_grad(set_to_none=True)
                with torch.enable_grad():
                    outputs = base_model({image_feature_name: tensor})

                logits = None
                if isinstance(outputs, dict):
                    feature_out = outputs.get(output_feature_name)
                    if isinstance(feature_out, dict):
                        logits = feature_out.get("logits") or feature_out.get("logit")
                    elif isinstance(feature_out, torch.Tensor):
                        logits = feature_out

                    # Ludwig 0.10+ uses namespaced keys: "<feature>::logits"
                    if logits is None:
                        ns_key = f"{output_feature_name}::logits"
                        if isinstance(outputs.get(ns_key), torch.Tensor):
                            logits = outputs[ns_key]

                    # Fallback: a top-level logits tensor
                    if logits is None and isinstance(outputs.get("logits"), torch.Tensor):
                        logits = outputs.get("logits")

                if logits is None:
                    raise ValueError("Could not locate logits for Grad-CAM.")

                if logits.dim() == 1:
                    target_logit = logits.unsqueeze(0)
                else:
                    target_class = 0
                    if output_type != "regression" and logits.shape[-1] > 1:
                        target_class = int(torch.argmax(logits, dim=-1).item())
                    target_logit = logits[:, target_class]

                target_logit.sum().backward()

                if not activations or not gradients:
                    raise ValueError("Missing activations or gradients for Grad-CAM.")

                act = activations[-1]
                grad = gradients[-1]
                weights = grad.mean(dim=(2, 3), keepdim=True)
                cam = (weights * act).sum(dim=1)
                cam = torch.relu(cam)
                cam = cam.squeeze(0)
                cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=(int(height), int(width)), mode="bilinear", align_corners=False)
                cam = cam.squeeze().detach().cpu().numpy()
                if cam.max() > 0:
                    cam = cam / cam.max()
                heatmap_rgba = np.uint8(cm.get_cmap("jet")(cam) * 255)
                heatmap_img = Image.fromarray(heatmap_rgba).convert("RGBA").resize(resized_img.size)
                overlay = Image.blend(resized_img.convert("RGBA"), heatmap_img, alpha=0.45)

                stem = Path(raw_path).stem
                out_path = heatmap_dir / f"{stem}_gradcam.png"
                overlay.save(out_path)
                orig_path = heatmap_dir / f"{stem}_original.png"
                try:
                    resized_img.save(orig_path)
                except Exception:
                    orig_path = None

                generated.append(out_path)
                if orig_path:
                    pairs.append((orig_path, out_path))
            except Exception as exc:
                logger.warning("Grad-CAM failed for %s: %s", raw_path, exc)
            finally:
                try:
                    handle_fwd.remove()
                    handle_bwd.remove()
                except Exception:
                    pass

        if not generated:
            result["reason"] = "No heatmaps were generated (model may be non-convolutional or preprocessing missing)."
            return result

        zip_path = exp_dir / "feature_importance_examples.zip"
        try:
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for png in generated:
                    zf.write(png, png.name)
        except Exception as exc:
            logger.warning("Failed to create Grad-CAM zip: %s", exc)

        result.update(
            {
                "status": "generated",
                "preview_paths": generated[:6],
                "pairs": pairs[:6],
                "zip_path": zip_path if zip_path.exists() else None,
                "dir_path": heatmap_dir,
            }
        )
        return result

    @staticmethod
    def _extract_metric_series(stats: Dict[str, Any], split: str, prefer: Optional[str] = None) -> Tuple[Optional[str], Optional[List[float]]]:
        """Pull the first numeric metric list we can find for the requested split."""
        if not isinstance(stats, dict):
            return None, None

        split_stats = stats.get(split, {})
        ordered_metrics: List[Tuple[str, List[float]]] = []

        def _append_metrics(metric_map: Dict[str, Any]) -> None:
            for metric_name, values in metric_map.items():
                if isinstance(values, list) and any(isinstance(v, (int, float)) for v in values):
                    ordered_metrics.append((metric_name, values))

        if isinstance(split_stats, dict):
            combined = split_stats.get("combined")
            if isinstance(combined, dict):
                _append_metrics(combined)

            for feature_name, feature_metrics in split_stats.items():
                if feature_name == "combined" or not isinstance(feature_metrics, dict):
                    continue
                _append_metrics(feature_metrics)

        if prefer:
            for metric_name, values in ordered_metrics:
                if metric_name == prefer:
                    return metric_name, values

        return ordered_metrics[0] if ordered_metrics else (None, None)

    def generate_plots(self, output_dir: Path) -> None:
        """Generate Ludwig visualizations (train/val + test) for the latest experiment run."""
        logger.info("Generating Ludwig visualizations (train/val + test)…")

        # Train/validation visualizations
        train_plots = {
            "learning_curves",
        }

        # Test visualizations (multi-class transparency)
        test_plots = {
            "confusion_matrix",
            "compare_performance",
            "compare_classifiers_multiclass_multimetric",
            "frequency_vs_f1",
            "confidence_thresholding",
            "confidence_thresholding_data_vs_acc",
            "confidence_thresholding_data_vs_acc_subset",
            "confidence_thresholding_data_vs_acc_subset_per_class",
            # Binary-only visualizations will still be attempted; multi-class replacements handled elsewhere
            "binary_threshold_vs_metric",
            "roc_curves",
            "precision_recall_curves",
            "calibration_1_vs_all",
            "calibration_multiclass",
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
        gt_metadata = _check(exp_dir / "model" / TRAIN_SET_METADATA_FILE_NAME)

        dataset_path = None
        split_file = None
        desc = exp_dir / DESCRIPTION_FILE_NAME
        if desc.exists():
            with open(desc, "r") as f:
                cfg = json.load(f)
            dataset_path = _check(Path(cfg.get("dataset", "")))
            split_file = _check(Path(get_split_path(cfg.get("dataset", ""))))
            model_name = cfg.get("model_name", "model")
        else:
            model_name = "model"

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

        probs_path = None
        prob_candidates = [
            exp_dir / f"{LABEL_COLUMN_NAME}_probabilities.csv",
            exp_dir / f"{output_feature}_probabilities.csv" if output_feature else None,
            exp_dir / "probabilities.csv",
            exp_dir / "predictions.csv",
            exp_dir / PREDICTIONS_PARQUET_FILE_NAME,
        ]
        for cand in prob_candidates:
            if cand and Path(cand).exists():
                probs_path = str(cand)
                break

        viz_registry = get_visualizations_registry()
        if not viz_registry:
            logger.warning(
                "Ludwig visualizations registry not available; train/test PNGs will be skipped."
            )
            return

        base_kwargs = {
            "training_statistics": [training_stats] if training_stats else [],
            "test_statistics": [test_stats] if test_stats else [],
            "probabilities": [probs_path] if probs_path else [],
            "output_feature_name": output_feature,
            "ground_truth_split": 2,
            "top_n_classes": [20],
            "top_k": 3,
            "metrics": ["f1", "precision", "recall", "accuracy"],
            "positive_label": 0,
            "ground_truth_metadata": gt_metadata,
            "ground_truth": dataset_path,
            "split_file": split_file,
            "output_directory": None,  # set per plot below
            "normalize": False,
            "file_format": "png",
            "model_names": [model_name],
        }
        for viz_name, viz_func in viz_registry.items():
            if viz_name in train_plots:
                viz_dir_plot = train_viz
            elif viz_name in test_plots:
                viz_dir_plot = test_viz
            else:
                continue

            try:
                # Build per-viz kwargs based on the function signature to avoid unexpected args
                sig_params = set(inspect.signature(viz_func).parameters.keys())
                call_kwargs = {
                    k: v
                    for k, v in base_kwargs.items()
                    if k in sig_params and v is not None
                }
                if "output_directory" in sig_params:
                    call_kwargs["output_directory"] = str(viz_dir_plot)

                viz_func(
                    **call_kwargs,
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
        train_set_metadata_path = exp_dir / "model" / TRAIN_SET_METADATA_FILE_NAME
        label_metadata_path = config.get("label_column_data_path")
        if label_metadata_path:
            label_metadata_path = Path(label_metadata_path)
        dataset_path_from_desc: Optional[Path] = None

        # Pull additional config details from description.json if available
        config_for_summary = dict(config)
        if "target_column" not in config_for_summary or not config_for_summary.get("target_column"):
            config_for_summary["target_column"] = LABEL_COLUMN_NAME
        desc_path = exp_dir / DESCRIPTION_FILE_NAME
        if desc_path.exists():
            try:
                with open(desc_path, "r") as f:
                    desc_json = json.load(f)
                desc_cfg = desc_json.get("config", {}) if isinstance(desc_json, dict) else {}
                encoder_cfg = (
                    desc_cfg.get("input_features", [{}])[0].get("encoder", {})
                    if isinstance(desc_cfg.get("input_features", [{}]), list)
                    else {}
                )
                output_cfg = (
                    desc_cfg.get("output_features", [{}])[0]
                    if isinstance(desc_cfg.get("output_features", [{}]), list)
                    else {}
                )
                trainer_cfg = desc_cfg.get("trainer", {}) if isinstance(desc_cfg, dict) else {}
                loss_cfg = output_cfg.get("loss", {}) if isinstance(output_cfg, dict) else {}
                opt_cfg = trainer_cfg.get("optimizer", {}) if isinstance(trainer_cfg, dict) else {}
                clip_cfg = trainer_cfg.get("gradient_clipping", {}) if isinstance(trainer_cfg, dict) else {}

                arch_type = encoder_cfg.get("type")
                arch_variant = encoder_cfg.get("model_variant")
                arch_custom = encoder_cfg.get("custom_model")
                arch_name = None
                if arch_custom:
                    arch_name = str(arch_custom)
                if arch_type:
                    arch_base = str(arch_type).replace("_", " ").title()
                    arch_type_name = (
                        f"{arch_base} {arch_variant}" if arch_variant is not None else arch_base
                    )
                    # Prefer explicit custom model names (e.g., MetaFormer) but fall back to encoder type
                    arch_name = arch_name or arch_type_name
                if not arch_name and config.get("model_name"):
                    # As a last resort, show the user-selected model name (handles custom/MetaFormer cases)
                    arch_name = str(config.get("model_name"))

                summary_fields = {
                    "architecture": arch_name,
                    "model_variant": arch_variant,
                    "pretrained": encoder_cfg.get("use_pretrained"),
                    "trainable": encoder_cfg.get("trainable"),
                    "target_column": output_cfg.get("column"),
                    "task_type": output_cfg.get("type"),
                    "validation_metric": trainer_cfg.get("validation_metric"),
                    "loss_function": loss_cfg.get("type"),
                    "threshold": output_cfg.get("threshold"),
                    "total_epochs": trainer_cfg.get("epochs"),
                    "early_stop": trainer_cfg.get("early_stop"),
                    "batch_size": trainer_cfg.get("batch_size"),
                    "optimizer": opt_cfg.get("type"),
                    "learning_rate": trainer_cfg.get("learning_rate"),
                    "random_seed": desc_cfg.get("random_seed") or config.get("random_seed"),
                    "use_mixed_precision": trainer_cfg.get("use_mixed_precision"),
                    "gradient_clipping": clip_cfg.get("clipglobalnorm"),
                }
                for k, v in summary_fields.items():
                    if v is None:
                        continue
                    # Do not override user-passed target/image column names in config
                    if k in {"target_column", "image_column"} and config_for_summary.get(k):
                        continue
                    config_for_summary.setdefault(k, v)

                dataset_field = None
                if isinstance(desc_json, dict):
                    dataset_field = desc_json.get("dataset") or desc_cfg.get("dataset")
                if dataset_field:
                    try:
                        dataset_path_from_desc = Path(dataset_field)
                    except TypeError:
                        dataset_path_from_desc = None
                if dataset_path_from_desc and (not label_metadata_path or not label_metadata_path.exists()):
                    label_metadata_path = dataset_path_from_desc
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(f"Could not merge description.json into config summary: {e}")

        base_viz_dir = exp_dir / "visualizations"
        train_viz_dir = base_viz_dir / "train"

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

        def append_plot_blocks(tab_html: str, plots: List[Dict[str, str]], title_suffix: str = "") -> str:
            """Append Plotly blocks to a tab with consistent markup."""
            if not plots:
                return tab_html
            suffix = title_suffix or ""
            for plot in plots:
                tab_html += (
                    f"<h2 style='text-align: center;'>{plot['title']}{suffix}</h2>"
                    f"<div class='plotly-center'>{plot['html']}</div>"
                )
            return tab_html

        def build_dataset_overview(
            label_metadata: Optional[Path],
            output_type: Optional[str],
            split_probabilities: Optional[List[float]],
            label_split_counts: Optional[List[Dict[str, int]]] = None,
            split_counts: Optional[Dict[int, int]] = None,
            fallback_dataset: Optional[Path] = None,
        ) -> str:
            """Summarize dataset distribution across splits using the actual split config."""
            if label_split_counts:
                # Use the actual counts captured during data prep instead of heuristics.
                return format_dataset_overview_table(label_split_counts, regression_mode=False)

            if output_type == "regression" and split_counts:
                rows = [
                    {"split": "train", "count": int(split_counts.get(0, 0))},
                    {"split": "validation", "count": int(split_counts.get(1, 0))},
                    {"split": "test", "count": int(split_counts.get(2, 0))},
                ]
                return format_dataset_overview_table(rows, regression_mode=True)

            candidate_paths: List[Path] = []
            if label_metadata and label_metadata.exists():
                candidate_paths.append(label_metadata)
            if fallback_dataset and fallback_dataset.exists():
                candidate_paths.append(fallback_dataset)
            if not candidate_paths:
                return format_dataset_overview_table([])

            def _normalize_split_probabilities(
                probs: Optional[List[float]],
            ) -> Optional[List[float]]:
                if not probs or len(probs) != 3:
                    return None
                try:
                    probs = [float(p) for p in probs]
                except (TypeError, ValueError):
                    return None
                total = sum(probs)
                if total <= 0:
                    return None
                return [p / total for p in probs]

            def _split_counts_from_column(df: pd.DataFrame) -> Dict[int, int]:
                if SPLIT_COLUMN_NAME not in df.columns:
                    return {}
                split_series = pd.to_numeric(
                    df[SPLIT_COLUMN_NAME], errors="coerce"
                ).dropna()
                if split_series.empty:
                    return {}
                split_series = split_series.astype(int)
                return split_series.value_counts().to_dict()

            def _split_counts_from_probs(total: int, probs: List[float]) -> Dict[int, int]:
                train_n = int(total * probs[0])
                val_n = int(total * probs[1])
                test_n = max(0, total - train_n - val_n)
                return {0: train_n, 1: val_n, 2: test_n}

            fallback_rows: Optional[List[Dict[str, int]]] = None
            for meta_path in candidate_paths:
                try:
                    df_labels = pd.read_csv(meta_path)
                    probs = _normalize_split_probabilities(split_probabilities)

                    # Regression (or missing label column): only need split counts
                    if output_type == "regression" or LABEL_COLUMN_NAME not in df_labels.columns:
                        split_counts_found = _split_counts_from_column(df_labels)
                        if split_counts_found:
                            rows = [
                                {"split": "train", "count": int(split_counts_found.get(0, 0))},
                                {"split": "validation", "count": int(split_counts_found.get(1, 0))},
                                {"split": "test", "count": int(split_counts_found.get(2, 0))},
                            ]
                            return format_dataset_overview_table(rows, regression_mode=True)
                        if probs and fallback_rows is None:
                            split_counts_found = _split_counts_from_probs(len(df_labels), probs)
                            fallback_rows = [
                                {"split": "train", "count": int(split_counts_found.get(0, 0))},
                                {"split": "validation", "count": int(split_counts_found.get(1, 0))},
                                {"split": "test", "count": int(split_counts_found.get(2, 0))},
                            ]
                        continue

                    # Classification: prefer actual split assignments; fall back to configured probabilities
                    if SPLIT_COLUMN_NAME in df_labels.columns:
                        df_counts = df_labels[[LABEL_COLUMN_NAME, SPLIT_COLUMN_NAME]].copy()
                        df_counts[SPLIT_COLUMN_NAME] = pd.to_numeric(
                            df_counts[SPLIT_COLUMN_NAME], errors="coerce"
                        )
                        df_counts = df_counts.dropna(subset=[SPLIT_COLUMN_NAME])
                        if df_counts.empty:
                            continue

                        df_counts[SPLIT_COLUMN_NAME] = df_counts[SPLIT_COLUMN_NAME].astype(int)
                        df_counts = df_counts.dropna(subset=[LABEL_COLUMN_NAME])
                        counts = (
                            df_counts.groupby([LABEL_COLUMN_NAME, SPLIT_COLUMN_NAME])
                            .size()
                            .unstack(fill_value=0)
                            .sort_index()
                        )
                        rows = []
                        for lbl, row in counts.iterrows():
                            rows.append(
                                {
                                    "label": str(lbl),
                                    "train": int(row.get(0, 0)),
                                    "validation": int(row.get(1, 0)),
                                    "test": int(row.get(2, 0)),
                                }
                            )
                        return format_dataset_overview_table(rows)

                    if probs:
                        label_series = df_labels[LABEL_COLUMN_NAME].dropna()
                        label_counts = label_series.value_counts().sort_index()
                        if label_counts.empty:
                            continue
                        rows = []
                        for lbl, count in label_counts.items():
                            train_n = int(count * probs[0])
                            val_n = int(count * probs[1])
                            test_n = max(0, count - train_n - val_n)
                            rows.append(
                                {
                                    "label": str(lbl),
                                    "train": train_n,
                                    "validation": val_n,
                                    "test": test_n,
                                }
                            )
                        fallback_rows = fallback_rows or rows
                except Exception as exc:
                    logger.warning("Failed to build dataset overview from %s: %s", meta_path, exc)
                    continue

            if fallback_rows:
                return format_dataset_overview_table(fallback_rows, regression_mode=output_type == "regression")
            return format_dataset_overview_table([])

        metrics_html = ""
        train_val_metrics_html = ""
        test_metrics_html = ""
        output_type = None
        train_stats_path = exp_dir / "training_statistics.json"
        test_stats_path = exp_dir / TEST_STATISTICS_FILE_NAME
        try:
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

        if not output_type:
            # Fallback to configured task type when stats are unavailable (e.g., failed run).
            output_type = (
                str(config_for_summary.get("task_type")).lower()
                if config_for_summary.get("task_type")
                else None
            )

        dataset_overview_html = build_dataset_overview(
            label_metadata_path,
            output_type,
            config.get("split_probabilities"),
            config.get("label_split_counts"),
            config.get("split_counts"),
            dataset_path_from_desc,
        )

        config_html = ""
        training_progress = self.get_training_process(output_dir)
        try:
            config_html = format_config_table_html(
                config_for_summary, split_info, training_progress, output_type
            )
        except Exception as e:
            logger.warning(f"Could not load config for HTML report: {e}")
            config_html = (
                "<h2 style='text-align: center;'>Model and Training Summary</h2>"
                "<p style='text-align:center; color:#666;'>Configuration details unavailable.</p>"
            )
        if not config_html:
            config_html = (
                "<h2 style='text-align: center;'>Model and Training Summary</h2>"
                "<p style='text-align:center; color:#666;'>No configuration details found.</p>"
            )

        # ---------- image rendering with exclusions ----------
        def render_img_section(
            title: str,
            dir_path: Path,
            output_type: str = None,
            exclude_names: Optional[set] = None,
        ) -> str:
            if not dir_path.exists():
                return ""

            exclude_names = exclude_names or set()

            # Search recursively because Ludwig can nest figures under per-feature folders
            imgs = list(dir_path.rglob("*.png"))

            # Exclude ROC curves and standard confusion matrices (keep only entropy version)
            default_exclude = {
                # "roc_curves.png",  # Remove ROC curves from test tab
                "confusion_matrix__label_top5.png",  # Remove standard confusion matrix
                "confusion_matrix__label_top10.png",  # Remove duplicate
                "confusion_matrix__label_top6.png",   # Remove duplicate
                "confusion_matrix_entropy__label_top10.png",  # Keep only top5
                "confusion_matrix_entropy__label_top6.png",   # Keep only top5
            }
            title_is_test = title.lower().startswith("test")
            if title_is_test and output_type == "binary":
                default_exclude.update(
                    {
                        "confusion_matrix__label_top2.png",
                        "confusion_matrix_entropy__label_top2.png",
                        "roc_curves_from_prediction_statistics.png",
                    }
                )
            elif title_is_test and output_type == "category":
                default_exclude.update(
                    {
                        "compare_classifiers_multiclass_multimetric__label_best10.png",
                        "compare_classifiers_multiclass_multimetric__label_sorted.png",
                        "compare_classifiers_multiclass_multimetric__label_worst10.png",
                    }
                )

            imgs = [
                img
                for img in imgs
                if img.name not in default_exclude
                and img.name not in exclude_names
                and not (
                    "learning_curves" in img.stem
                    and "loss" in img.stem
                    and "label" in img.stem
                )
            ]

            if not imgs:
                return ""

            # Sort images by name for consistent ordering (works with string and numeric labels)
            imgs = sorted(imgs, key=lambda x: x.name)

            html_section = ""
            custom_titles = {
                "compare_classifiers_multiclass_multimetric__label_top10": "Metric Comparison by Label",
                "compare_classifiers_performance_from_prob": "Label Metric Comparison by Probability",
            }
            for img in imgs:
                b64 = encode_image_to_base64(str(img))
                default_title = img.stem.replace("_", " ").title()
                img_title = custom_titles.get(img.stem, default_title)
                html_section += (
                    f"<h2 style='text-align: center;'>{img_title}</h2>"
                    f'<div class="plot" style="margin-bottom:20px;text-align:center;">'
                    f'<img src="data:image/png;base64,{b64}" '
                    f'style="max-width:90%;max-height:600px;border:1px solid #ddd;" />'
                    f"</div>"
                )
            return html_section

        # Show dataset overview, performance first, then config
        predictions_csv_path = exp_dir / "predictions.csv"

        tab1_content = dataset_overview_html + metrics_html + config_html

        tab2_content = train_val_metrics_html
        # Preload binary threshold plot so it appears first in Train/Val tab
        threshold_plot = None
        threshold_value = (
            config_for_summary.get("threshold")
            if config_for_summary.get("threshold") is not None
            else config.get("threshold")
        )
        if threshold_value is None and output_type == "binary":
            threshold_value = 0.5
        if output_type == "binary" and predictions_csv_path.exists():
            try:
                threshold_plot = build_binary_threshold_plot(
                    str(predictions_csv_path),
                    label_data_path=str(config.get("label_column_data_path"))
                    if config.get("label_column_data_path")
                    else None,
                    split_value=1,
                )
            except Exception as e:
                logger.warning(f"Could not generate validation threshold plot: {e}")

        if train_stats_path.exists():
            try:
                if output_type == "regression":
                    tv_plots = build_regression_train_val_plots(str(train_stats_path))
                    tab2_content = append_plot_blocks(tab2_content, tv_plots)
                else:
                    tv_plots = build_train_validation_plots(str(train_stats_path))
                    # Add threshold plot first, then other train/val plots
                    if threshold_plot:
                        tab2_content = append_plot_blocks(tab2_content, [threshold_plot])
                        # Only append once; avoid duplicates if added elsewhere
                        threshold_plot = None
                    tab2_content = append_plot_blocks(tab2_content, tv_plots)
                    if threshold_plot or tv_plots:
                        logger.info(
                            f"Added {len(tv_plots) + (1 if threshold_plot else 0)} train/val diagnostic plots"
                        )
            except Exception as e:
                logger.warning(f"Could not generate train/val plots: {e}")

        # Only include training PNGs for regression; classification is handled by filtered Plotly plots
        if output_type == "regression":
            tab2_content += render_img_section(
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

        # Validation diagnostics (calibration/threshold) from predictions.csv, using split=1
        if output_type in ("binary", "category") and predictions_csv_path.exists():
            try:
                val_diag_plots = build_prediction_diagnostics(
                    str(predictions_csv_path),
                    label_data_path=str(config.get("label_column_data_path"))
                    if config.get("label_column_data_path")
                    else None,
                    split_value=1,
                )
                val_conf_plots = [
                    p for p in val_diag_plots if "Prediction Confidence Distribution" in p.get("title", "")
                ]
                tab2_content = append_plot_blocks(
                    tab2_content, val_conf_plots, " (Validation)"
                )
            except Exception as e:
                logger.warning(f"Could not generate validation diagnostics: {e}")

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
                    "<div class='scroll-rows-30' style='overflow-x:auto; overflow-y:auto; max-height:350px; margin-bottom:20px;'>"
                    + preds_html
                    + "</div>"
                )
            except Exception as e:
                logger.warning(f"Could not build Predictions vs GT table: {e}")

        tab3_content = test_metrics_html + preds_section

        gradcam_info = self._generate_gradcam_heatmaps(exp_dir, config, output_type)

        if output_type == "regression" and train_stats_path.exists():
            try:
                test_plots = build_regression_test_plots(str(train_stats_path))
                tab3_content = append_plot_blocks(tab3_content, test_plots)
                if test_plots:
                    logger.info(f"Generated {len(test_plots)} regression test plots")
            except Exception as e:
                logger.warning(f"Could not generate regression test plots: {e}")

        if output_type in ("binary", "category") and test_stats_path.exists():
            try:
                interactive_plots = build_classification_plots(
                    str(test_stats_path),
                    str(train_stats_path) if train_stats_path.exists() else None,
                    metadata_csv_path=str(label_metadata_path)
                    if label_metadata_path and label_metadata_path.exists()
                    else None,
                    train_set_metadata_path=str(train_set_metadata_path)
                    if train_set_metadata_path.exists()
                    else None,
                    threshold=threshold_value,
                )
                tab3_content = append_plot_blocks(tab3_content, interactive_plots)
                if interactive_plots:
                    logger.info(f"Generated {len(interactive_plots)} interactive Plotly plots")
            except Exception as e:
                logger.warning(f"Could not generate Plotly plots: {e}")

            # Multi-class transparency plots from test stats (replace ROC/PR for multi-class)
            if output_type == "category" and test_stats_path.exists():
                try:
                    multi_curves = build_multiclass_metric_plots(
                        str(test_stats_path),
                        metadata_csv_path=str(label_metadata_path)
                        if label_metadata_path and label_metadata_path.exists()
                        else None,
                        train_set_metadata_path=str(train_set_metadata_path)
                        if train_set_metadata_path.exists()
                        else None,
                    )
                    tab3_content = append_plot_blocks(tab3_content, multi_curves)
                    if multi_curves:
                        logger.info("Added multi-class per-class metric plots to test tab")
                except Exception as e:
                    logger.warning(f"Could not generate multi-class metric plots: {e}")

            # Test diagnostics (confidence histogram) from predictions.csv, using split=2
            if predictions_csv_path.exists():
                try:
                    test_diag_plots = build_prediction_diagnostics(
                        str(predictions_csv_path),
                        label_data_path=str(config.get("label_column_data_path"))
                        if config.get("label_column_data_path")
                        else None,
                        split_value=2,
                    )
                    test_conf_plots = [
                        p for p in test_diag_plots if "Prediction Confidence Distribution" in p.get("title", "")
                    ]
                    if test_conf_plots:
                        tab3_content = append_plot_blocks(tab3_content, test_conf_plots)
                        logger.info("Added test prediction confidence plot")
                except Exception as e:
                    logger.warning(f"Could not generate test diagnostics: {e}")

        if gradcam_info.get("status") == "generated":
            tab3_content += "<h2 style='text-align: center;'>Grad-CAM Heatmaps</h2>"
            for orig_path, heat_path in gradcam_info.get("pairs", [])[:4]:
                try:
                    display_name = Path(str(orig_path)).name
                    if display_name.endswith("_original.png"):
                        display_name = display_name[: -len("_original.png")]
                    b64_orig = encode_image_to_base64(str(orig_path))
                    b64_heat = encode_image_to_base64(str(heat_path))
                    tab3_content += (
                        "<div class='plot' style='margin-bottom:15px;text-align:center;display:flex;gap:12px;justify-content:center;flex-wrap:wrap;'>"
                        f"<div><div style='font-weight:600;margin-bottom:4px;'>{display_name}</div>"
                        f"<img src='data:image/png;base64,{b64_orig}' style='max-width:320px;max-height:320px;border:1px solid #ddd;' /></div>"
                        f"<div><div style='font-weight:600;margin-bottom:4px;'>Grad-CAM</div>"
                        f"<img src='data:image/png;base64,{b64_heat}' style='max-width:320px;max-height:320px;border:1px solid #ddd;' /></div>"
                        "</div>"
                    )
                except Exception as exc:
                    logger.debug("Could not embed Grad-CAM pair %s / %s: %s", orig_path, heat_path, exc)

        # Add static TEST PNGs (with default dedupe/exclusions)
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
