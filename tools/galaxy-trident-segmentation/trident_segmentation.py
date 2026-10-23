#!/usr/bin/env python3
"""
Galaxy wrapper for TRIDENT tissue segmentation.

This script processes whole-slide images (WSI) to segment tissue regions
from background using TRIDENT's segmentation models.
"""

import argparse
import logging
import os
import platform
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import torch

# Configure logging
logging.basicConfig(
    stream=sys.stdout,
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


def setup_trident_environment():
    """Set up TRIDENT environment variables."""
    # Set HuggingFace cache directory
    hf_cache = os.path.join(os.getcwd(), "hf_cache")
    os.makedirs(hf_cache, exist_ok=True)
    os.environ["HF_HOME"] = hf_cache
    os.environ["TORCH_HOME"] = hf_cache


def process_wsi_collection(
    input_collection,
    job_dir,
    segmenter="hest",
    remove_artifacts=False,
    remove_penmarks=False,
    gpu=0,
    skip_errors=False,
):
    """
    Process a collection of WSI files for tissue segmentation.

    Parameters
    ----------
    input_collection : list
        List of WSI file paths
    job_dir : str
        Output directory for results
    segmenter : str
        Segmentation model ('hest' or 'grandqc')
    remove_artifacts : bool
        Remove artifacts from segmentation
    remove_penmarks : bool
        Remove penmarks from segmentation
    gpu : int
        GPU index to use
    skip_errors : bool
        Skip errored slides and continue
    """
    # Import TRIDENT modules
    from trident import Processor
    from trident.segmentation_models.load import segmentation_model_factory

    # Create temporary directory for WSI files
    with tempfile.TemporaryDirectory(prefix="trident_wsi_", dir=os.getcwd()) as temp_wsi_dir:
        wsi_dir = Path(temp_wsi_dir)

        # Copy WSI files to temporary directory
        logger.info(f"Copying {len(input_collection)} WSI files to temporary directory")
        for wsi_path in input_collection:
            wsi_file = Path(wsi_path)
            dest_path = wsi_dir / wsi_file.name
            shutil.copy2(wsi_file, dest_path)
            logger.info(f"Copied {wsi_file.name}")

        # Determine device (automatically detect GPU)
        if torch.cuda.is_available():
            device = f'cuda:{gpu}'
            logger.info(f"GPU detected: Using CUDA device {gpu}")
            use_multiprocessing = True  # Enable multiprocessing on Linux/GPU
        else:
            device = 'cpu'
            logger.info("No GPU detected: Using CPU")
            # On macOS, disable multiprocessing to avoid OpenSlide pickling errors
            # On Linux, multiprocessing works fine even on CPU
            use_multiprocessing = platform.system() != 'Darwin'
            if not use_multiprocessing:
                logger.info("macOS detected: Disabling multiprocessing for compatibility")

        # Initialize processor
        processor = Processor(
            job_dir=job_dir,
            wsi_source=str(wsi_dir),
            skip_errors=skip_errors,
            max_workers=0 if not use_multiprocessing else None,  # Disable on macOS, auto-detect on Linux
        )

        # Load segmentation model
        logger.info(f"Loading segmentation model: {segmenter}")
        segmentation_model = segmentation_model_factory(segmenter)

        # Load artifact remover if requested
        artifact_remover_model = None
        if remove_artifacts or remove_penmarks:
            artifact_remover_model = segmentation_model_factory(
                'grandqc_artifact',
                remove_penmarks_only=remove_penmarks and not remove_artifacts
            )

        # Run segmentation
        logger.info("Running segmentation job...")
        try:
            # On macOS, disable multiprocessing to avoid OpenSlide pickling errors
            # On Linux/GPU, allow multiprocessing for better performance
            if platform.system() == 'Darwin':
                # Patch WSI.segment_tissue to force num_workers=0 on macOS
                from trident.wsi_objects import WSI

                if not hasattr(WSI.WSI, '_original_segment_tissue'):
                    WSI.WSI._original_segment_tissue = WSI.WSI.segment_tissue

                    def patched_segment_tissue(self, *args, num_workers=None, **kwargs):
                        # Force num_workers=0 on macOS to avoid pickling errors
                        logger.info("macOS detected: Using num_workers=0 (single-threaded)")
                        return self._original_segment_tissue(*args, num_workers=0, **kwargs)

                    WSI.WSI.segment_tissue = patched_segment_tissue

            # Adjust batch size based on device
            if device.startswith('cuda'):
                batch_size = 32  # Larger batch for GPU
                logger.info(f"GPU mode: Using batch_size={batch_size}")
            else:
                batch_size = 16  # Smaller batch for CPU
                logger.info(f"CPU mode: Using batch_size={batch_size}")

            processor.run_segmentation_job(
                segmentation_model=segmentation_model,
                seg_mag=segmentation_model.target_mag,
                holes_are_tissue=True,  # Keep holes by default
                artifact_remover_model=artifact_remover_model,
                batch_size=batch_size,
                device=device,
            )
            logger.info("TRIDENT segmentation completed successfully")
        except Exception as e:
            logger.error(f"TRIDENT segmentation failed: {e}")
            if not skip_errors:
                raise
            logger.warning("Continuing despite error (--skip_errors enabled)")


def collect_outputs(job_dir, output_zip):
    """
    Collect segmentation outputs into a ZIP file.

    Parameters
    ----------
    job_dir : str
        TRIDENT output directory
    output_zip : str
        Path to output ZIP file
    """
    job_path = Path(job_dir)

    # Collect outputs
    outputs_to_collect = {
        "contours_geojson": "*.geojson",
        "contours": "*.jpg",
        "thumbnails": "*.jpg",
    }

    logger.info(f"Collecting outputs from {job_dir}")

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        for subdir, pattern in outputs_to_collect.items():
            subdir_path = job_path / subdir
            if subdir_path.exists():
                for file_path in subdir_path.glob(pattern):
                    arcname = f"{subdir}/{file_path.name}"
                    zipf.write(file_path, arcname)
                    logger.info(f"Added {arcname} to output ZIP")

        # Also collect any other output files (logs, configs)
        for file_path in job_path.glob("_*"):
            if file_path.is_file():
                zipf.write(file_path, file_path.name)
                logger.info(f"Added {file_path.name} to output ZIP")

    logger.info(f"Output ZIP created: {output_zip}")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="TRIDENT Tissue Segmentation for Galaxy"
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Input WSI file paths",
    )
    parser.add_argument(
        "--original_name",
        action="append",
        required=True,
        help="Original file names",
    )
    parser.add_argument(
        "--output_zip",
        required=True,
        help="Output ZIP file path",
    )
    parser.add_argument(
        "--segmenter",
        choices=["hest", "grandqc"],
        default="hest",
        help="Segmentation model to use",
    )
    parser.add_argument(
        "--remove_artifacts",
        action="store_true",
        help="Remove artifacts from segmentation",
    )
    parser.add_argument(
        "--remove_penmarks",
        action="store_true",
        help="Remove penmarks from segmentation",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU index to use",
    )
    parser.add_argument(
        "--skip_errors",
        action="store_true",
        help="Skip errored slides and continue",
    )
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()

    if len(args.input) != len(args.original_name):
        raise ValueError("Mismatch between input paths and original names")

    # Set up environment
    setup_trident_environment()

    # Create output directory
    output_dir = os.path.join(os.getcwd(), "trident_output")
    os.makedirs(output_dir, exist_ok=True)

    # Process WSI collection
    process_wsi_collection(
        input_collection=args.input,
        job_dir=output_dir,
        segmenter=args.segmenter,
        remove_artifacts=args.remove_artifacts,
        remove_penmarks=args.remove_penmarks,
        gpu=args.gpu,
        skip_errors=args.skip_errors,
    )

    # Collect outputs
    collect_outputs(output_dir, args.output_zip)

    logger.info("Segmentation pipeline completed successfully")


if __name__ == "__main__":
    main()
