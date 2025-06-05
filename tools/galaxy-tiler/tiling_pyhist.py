import argparse
import logging
import os
import shutil
import subprocess
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Tuple

import openslide
import psutil
from pyhist import PySlide, TileGenerator

from src import utility_functions

# Configure logging to stdout
logging.basicConfig(
    stream=sys.stdout,
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# Constants
SEGMENT_BINARY_PATH = "/pyhist/src/graph_segmentation/segment"
DEFAULT_PATCH_SIZE = 256
DEFAULT_DOWNSCALE_FACTOR = 8
TILE_FORMAT = "png"
MEMORY_PER_WORKER = 1  # GB, estimated memory per worker process


def log_memory_usage() -> None:
    """Log the current memory usage of the process in megabytes."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logging.info(
        "Memory usage: RSS=%.2f MB, VMS=%.2f MB",
        mem_info.rss / 1024 / 1024,
        mem_info.vms / 1024 / 1024
    )


def validate_slide(image_path: Path) -> None:
    """Validate the input image using OpenSlide."""
    try:
        with openslide.OpenSlide(str(image_path)) as slide:
            logging.info("Validated input file with OpenSlide: %s", image_path)
    except openslide.OpenSlideError as error:
        raise RuntimeError("Invalid input file: %s", error) from error


def check_segmentation_binary() -> bool:
    """Check if the segmentation binary exists and is executable."""
    if os.path.exists(SEGMENT_BINARY_PATH) and os.access(SEGMENT_BINARY_PATH, os.X_OK):
        logging.info("Segmentation executable found: %s", SEGMENT_BINARY_PATH)
        return True
    logging.warning("Segmentation executable missing, using Otsu method")
    return False


def build_pyhist_config(image_path: Path, output_dir: Path) -> dict:
    """Build the configuration dictionary for PyHIST processing."""
    return {
        "svs": str(image_path),
        "patch_size": DEFAULT_PATCH_SIZE,
        "method": "otsu",
        "thres": 0.1,
        "output_downsample": DEFAULT_DOWNSCALE_FACTOR,
        "mask_downsample": DEFAULT_DOWNSCALE_FACTOR,
        "borders": "0000",
        "corners": "1010",
        "pct_bc": 1,
        "k_const": 1000,
        "minimum_segmentsize": 1000,
        "save_patches": True,
        "save_blank": False,
        "save_nonsquare": False,
        "save_tilecrossed_image": False,
        "save_mask": True,
        "save_edges": False,
        "info": "verbose",
        "output": str(output_dir),
        "format": TILE_FORMAT,
    }


def process_image_with_pyhist(
    image_path: Path, output_dir: Path, original_name: str
) -> Path:
    """Process a single image with PyHIST and return the tile directory."""
    logging.info("Processing image: %s", image_path)
    log_memory_usage()

    # Validate input
    validate_slide(image_path)

    # Check segmentation method
    check_segmentation_binary()

    # Prepare PyHIST configuration
    config = build_pyhist_config(image_path, output_dir)

    # Set logging level based on config
    log_levels = {
        "default": logging.INFO,
        "verbose": logging.DEBUG,
        "silent": logging.CRITICAL,
    }
    logging.getLogger().setLevel(log_levels[config["info"]])

    # Process the slide
    utility_functions.check_image(config["svs"])
    slide = PySlide(config)
    logging.info("Slide loaded: %s", slide)

    tile_generator = TileGenerator(slide)
    logging.info("Tile generator initialized: %s", tile_generator)

    try:
        tile_generator.execute()
    except subprocess.CalledProcessError as error:
        raise RuntimeError("Tile extraction failed: %s", error) from error

    tile_dir = Path(slide.tile_folder)
    tiles = list(tile_dir.glob(f"*.{TILE_FORMAT}"))
    logging.info("Found %d tiles in %s", len(tiles), tile_dir)

    utility_functions.clean(slide)
    return tile_dir


def append_tiles_to_zip(
    zip_file: zipfile.ZipFile,
    original_name: str,
    tile_dir: Path
) -> None:
    """Append PNG tiles from the tile directory to the ZIP file."""
    original_base = Path(original_name).stem
    tiles = list(tile_dir.glob(f"*.{TILE_FORMAT}"))

    for tile in tiles:
        tile_number = tile.stem.split("_")[-1]
        arcname = f"{original_base}/{original_base}_{tile_number}.{TILE_FORMAT}"
        zip_file.write(tile, arcname)

    logging.info("Appended %d tiles from %s", len(tiles), tile_dir)


def process_single_image(task: Tuple[Path, str, Path]) -> Path:
    """Process a single image and return the tile directory."""
    image_path, original_name, output_dir = task
    try:
        tile_dir = process_image_with_pyhist(
            image_path,
            output_dir,
            original_name
        )
        return tile_dir
    except Exception as error:
        logging.error("Error processing %s: %s", image_path, error)
        raise


def get_max_workers() -> int:
    """Determine the maximum number of worker processes based on
    available resources."""
    cpu_cores = psutil.cpu_count(logical=False)  # Physical CPU cores
    available_memory = psutil.virtual_memory().available / (1024 ** 3)  # in GB
    max_workers_memory = available_memory // MEMORY_PER_WORKER
    max_workers = min(cpu_cores, max_workers_memory)
    return max(1, int(max_workers))


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Tile extraction for Galaxy")
    parser.add_argument(
        "--input",
        action="append",
        help="Input image paths",
        default=[]
    )
    parser.add_argument(
        "--original_name",
        action="append",
        help="Original file names",
        default=[]
    )
    parser.add_argument(
        "--output_zip",
        required=True,
        help="Output ZIP file path"
    )
    return parser.parse_args()


def main() -> None:
    """Main function to orchestrate tile extraction and
    ZIP creation with dynamic multiprocessing."""
    os.chdir("/pyhist")
    logging.info("Working directory: %s", os.getcwd())

    args = parse_arguments()

    if len(args.input) != len(args.original_name):
        raise ValueError("Mismatch between input paths and original names")

    # Create a temporary directory for tile storage
    temp_dir = Path("temp_tiles")
    temp_dir.mkdir(exist_ok=True)

    # Prepare tasks with unique output directories
    tasks = [
        (Path(image_path), original_name, temp_dir / Path(original_name).stem)
        for image_path, original_name in zip(args.input, args.original_name)
    ]

    # Determine the number of worker processes based on available resources
    max_workers = get_max_workers()
    logging.info("Using %d worker processes", max_workers)

    # Process images in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        tile_dirs = list(executor.map(process_single_image, tasks))

    # Create the ZIP file and append all tiles
    with zipfile.ZipFile(args.output_zip, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for (image_path, original_name, output_dir), tile_dir in zip(tasks, tile_dirs):
            append_tiles_to_zip(zip_file, original_name, tile_dir)

    # Clean up temporary files
    shutil.rmtree(temp_dir, ignore_errors=True)
    logging.info("Temporary files cleaned up")
    logging.info(
        "Final ZIP size: %d bytes",
        Path(args.output_zip).stat().st_size
    )


if __name__ == "__main__":
    main()
