# TRIDENT Tissue Segmentation Tool

Galaxy tool wrapper for TRIDENT's tissue segmentation functionality.

## Overview

This tool segments tissue regions from whole-slide images (WSI) using TRIDENT's segmentation models. It is the first step in the TRIDENT pipeline for whole-slide image processing.

## Features

- Supports multiple WSI formats (.svs, .tiff, .tif, .ndpi)
- Two segmentation models: HEST and GrandQC
- Optional artifact and penmark removal
- GPU acceleration support
- Batch processing of multiple slides

## Inputs

- **WSI Collection**: Collection of whole-slide images
- **Segmentation Model**: HEST (default) or GrandQC
- **Remove Artifacts**: Optional artifact removal
- **Remove Penmarks**: Optional penmark removal
- **GPU Index**: GPU device to use (0-based)

## Outputs

- **ZIP file** containing:
  - GeoJSON files with tissue contours
  - Thumbnail images with contours
  - Segmentation masks

## Usage

The tool processes a collection of WSI files and outputs segmentation results in a ZIP archive. GeoJSON files can be opened in QuPath for visualization and editing.

## Dependencies

- TRIDENT (cloned from mahmoodlab/TRIDENT)
- PyTorch with CUDA support
- OpenSlide for WSI reading
- Various Python packages (see Dockerfile)

## Citation

If you use TRIDENT in your research, please cite:

Zhang, A., Jaume, G., Vaidya, A., Ding, T., & Mahmood, F. (2025). Accelerating Data Processing and Benchmarking of AI Models for Pathology. arXiv preprint arXiv:2502.06750.
