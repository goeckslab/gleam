import os
import tempfile
import zipfile
from typing import List
import logging
import random
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
import pandas as pd

logging.basicConfig(level=logging.DEBUG)
LOG = logging.getLogger(__name__)


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
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_local_tmp():
    os.makedirs("/tmp", exist_ok=True)

def enable_tensor_cores_if_available():
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')

def load_file(path: str) -> pd.DataFrame:
    if not path:
        return None
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path, sep=None, engine="python")

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

    def resolve(p):
        if pd.isna(p):
            return None
        orig = Path(str(p).strip())
        candidates = []
        if orig.is_absolute():
            candidates.append(orig)
        if extracted_root is not None:
            candidates.extend([extracted_root / orig, extracted_root / orig.name])
        for cand in candidates:
            if cand.exists():
                return str(cand.resolve())
        return None

    # Infer image columns if none were provided
    if not image_columns:
        obj_cols = [c for c in df.columns if str(df[c].dtype) == "object"]
        inferred = []
        for col in obj_cols:
            sample = df[col].dropna().head(50)
            if sample.empty:
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
