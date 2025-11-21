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
        logger.warning(f"Could not parse --hyperparameters: {e}. Ignoring.")
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
    logger.info(f"Extracting {len(args.images_zip)} image ZIP(s) to {root}")

    for zip_path in args.images_zip:
        path = Path(zip_path)
        if not path.exists():
            raise FileNotFoundError(f"Image ZIP not found: {zip_path}")
        with zipfile.ZipFile(path, 'r') as z:
            z.extractall(root)
        logger.info(f"Extracted {path.name}")

    return root


def absolute_path_expander(df: pd.DataFrame, extracted_root: Optional[Path], image_columns: List[str]):
    if extracted_root is None or not image_columns:
        return

    placeholder = str(extracted_root / "__placeholder__.png")
    from PIL import Image
    if not os.path.exists(placeholder):
        Image.new("RGB", (1, 1)).save(placeholder)

    strategy_remove = os.getenv("MISSING_IMAGE_STRATEGY", "").lower() == "true" or False

    for col in image_columns:
        if col not in df.columns:
            continue

        def resolve(p):
            if pd.isna(p):
                return None
            orig = Path(str(p).strip())
            candidates = [
                extracted_root / orig,
                extracted_root / orig.name,
            ]
            for cand in candidates:
                if cand.exists():
                    return str(cand.resolve())
            return None

        resolved = df[col].apply(resolve)
        missing = resolved.isna()

        if missing.any():
            count = missing.sum()
            if strategy_remove:
                logger.warning(f"{col}: dropping {count} rows with missing images")
            else:
                resolved[missing] = placeholder
                logger.info(f"{col}: filled {count} missing images with placeholder")

        df[col] = resolved

    if strategy_remove:
        before = len(df)
        df.dropna(subset=image_columns, inplace=True)
        logger.info(f"Dropped {before - len(df)} total rows due to missing images")


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
