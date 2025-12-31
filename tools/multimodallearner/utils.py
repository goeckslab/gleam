import json
import logging
import os
import random
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

LOG = logging.getLogger(__name__)
_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
    ".webp",
    ".svs",
}
_MAX_PATH_COMPONENT = 240


def str2bool(val) -> bool:
    """Parse common truthy strings to bool."""
    return str(val).strip().lower() in ("1", "true", "yes", "y")


def load_user_hparams(hp_arg: Optional[str]) -> dict:
    """Parse --hyperparameters (inline JSON or path to .json)."""
    if not hp_arg:
        return {}
    try:
        s = hp_arg.strip()
        if s.startswith("{"):
            return json.loads(s)
        with open(s, "r") as f:
            return json.load(f)
    except Exception as e:
        LOG.warning(f"Could not parse --hyperparameters: {e}. Ignoring.")
        return {}


def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_local_tmp():
    os.makedirs("/tmp", exist_ok=True)


def enable_tensor_cores_if_available():
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")


def enable_deterministic_mode(seed: Optional[int] = None):
    """
    Force deterministic algorithms where possible to reduce run-to-run variance.
    """
    if seed is not None:
        set_seeds(seed)
        os.environ.setdefault("PYTHONHASHSEED", str(int(seed)))
    # cuBLAS determinism
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        torch.use_deterministic_algorithms(True)
    except Exception as e:
        LOG.warning(f"Could not enable torch deterministic algorithms: {e}")
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception as e:
        LOG.warning(f"Could not enforce deterministic cuDNN settings: {e}")
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
    except Exception:
        pass
    try:
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass


def load_file(path: str) -> pd.DataFrame:
    if not path:
        return None
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path, sep=None, engine="python")


def _normalize_path_value(val: object) -> Optional[str]:
    if val is None:
        return None
    s = str(val).strip().strip('"').strip("'")
    return s if s else None


def _has_long_component(path_str: str) -> bool:
    for part in path_str.replace("\\", "/").split("/"):
        if len(part) > _MAX_PATH_COMPONENT:
            return True
    return False


def _build_extracted_index(extracted_root: Optional[Path]) -> set:
    if extracted_root is None:
        return set()
    index = set()
    for root, _dirs, files in os.walk(extracted_root):
        rel_root = os.path.relpath(root, extracted_root)
        for fname in files:
            if _has_long_component(fname):
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext not in _IMAGE_EXTENSIONS:
                continue
            rel_path = fname if rel_root == "." else os.path.join(rel_root, fname)
            index.add(rel_path.replace("\\", "/"))
            index.add(fname)
    return index


def prepare_image_search_dirs(args) -> Optional[Path]:
    if not args.images_zip:
        return None

    root = Path(tempfile.mkdtemp(prefix="autogluon_images_"))
    LOG.info(f"Extracting {len(args.images_zip)} image ZIP(s) to {root}")

    for zip_path in args.images_zip:
        path = Path(zip_path)
        if not path.exists():
            raise FileNotFoundError(f"Image ZIP not found: {zip_path}")
        with zipfile.ZipFile(path, 'r') as z:
            z.extractall(root)
        LOG.info(f"Extracted {path.name}")

    return root


def absolute_path_expander(df: pd.DataFrame, extracted_root: Optional[Path], image_columns: Optional[List[str]]) -> List[str]:
    """
    Resolve image paths to absolute paths. If no image_columns are provided,
    infers candidate columns whose values resolve to existing files (checking
    absolute paths first, then paths relative to the extracted_root).
    """
    if df is None or df.empty:
        return []

    image_columns = [c for c in (image_columns or []) if c in df.columns]
    extracted_index = _build_extracted_index(extracted_root)

    def resolve(p):
        if pd.isna(p):
            return None
        raw = _normalize_path_value(p)
        if not raw or _has_long_component(raw):
            return None
        orig = Path(raw)
        candidates = []
        if orig.is_absolute():
            candidates.append(orig)
        if extracted_root is not None:
            candidates.extend([extracted_root / orig, extracted_root / orig.name])
        for cand in candidates:
            try:
                if cand.exists():
                    return str(cand.resolve())
            except OSError:
                continue
        return None

    def matches_extracted(p) -> bool:
        if not extracted_index or pd.isna(p):
            return False
        raw = _normalize_path_value(p)
        if not raw or _has_long_component(raw):
            return False
        norm = raw.replace("\\", "/").lstrip("./")
        return norm in extracted_index

    # Infer image columns if none were provided
    if not image_columns:
        obj_cols = [c for c in df.columns if str(df[c].dtype) == "object"]
        inferred = []
        for col in obj_cols:
            sample = df[col].dropna().head(50)
            if sample.empty:
                continue
            if extracted_index:
                matched = sample.apply(matches_extracted)
                if matched.any():
                    inferred.append(col)
                    continue
            resolved_sample = sample.apply(resolve)
            if resolved_sample.notna().any():
                inferred.append(col)
        image_columns = inferred
        if image_columns:
            LOG.info(f"Inferred image columns: {image_columns}")

    for col in image_columns:
        df[col] = df[col].apply(resolve)

    return image_columns


def verify_outputs(paths):
    ok = True
    for p, desc in paths:
        if os.path.exists(p):
            size = os.path.getsize(p)
            LOG.info(f"✓ Output {desc}: {p} ({size:,} bytes)")
            os.chmod(p, 0o644)
        else:
            LOG.error(f"✗ Output {desc} MISSING: {p}")
            ok = False
    if not ok:
        LOG.error("Some outputs are missing!")
        sys.exit(1)
