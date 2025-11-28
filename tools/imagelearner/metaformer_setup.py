import logging
from typing import Any, Dict

logger = logging.getLogger("ImageLearner")

# Optional MetaFormer configuration registry
META_DEFAULT_CFGS: Dict[str, Any] = {}
try:
    from MetaFormer import default_cfgs as META_DEFAULT_CFGS  # type: ignore[attr-defined]
except Exception as exc:  # pragma: no cover - optional dependency
    logger.debug("MetaFormer default configs unavailable: %s", exc)
    META_DEFAULT_CFGS = {}

# Try to import Ludwig visualization registry (may fail due to optional dependencies)
_ludwig_viz_available = False
try:
    from ludwig.visualize import get_visualizations_registry as _raw_get_visualizations_registry
    _ludwig_viz_available = True
    logger.info("Ludwig visualizations available")
except ImportError as exc:  # pragma: no cover - optional dependency
    logger.warning(
        "Ludwig visualizations not available: %s. Will use fallback plots only.",
        exc,
    )
    _raw_get_visualizations_registry = None
except Exception as exc:  # pragma: no cover - defensive
    logger.warning(
        "Ludwig visualizations not available due to dependency issues: %s. Will use fallback plots only.",
        exc,
    )
    _raw_get_visualizations_registry = None


def get_visualizations_registry():
    """Return the Ludwig visualizations registry or an empty dict if unavailable."""
    if not _raw_get_visualizations_registry:
        return {}
    try:
        return _raw_get_visualizations_registry()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to load Ludwig visualizations registry: %s", exc)
        return {}


# --- MetaFormer patching integration ---
_metaformer_patch_ok = False
try:
    from MetaFormer.metaformer_stacked_cnn import patch_ludwig_stacked_cnn as _mf_patch

    if _mf_patch():
        _metaformer_patch_ok = True
        logger.info("MetaFormer patching applied for Ludwig stacked_cnn encoder.")
except Exception as exc:  # pragma: no cover - optional dependency
    logger.warning("MetaFormer stacked CNN not available: %s", exc)
    _metaformer_patch_ok = False

# Note: CAFormer models are now handled through MetaFormer framework
