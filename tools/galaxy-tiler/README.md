# Tile Images with PyHIST

Tile Images with PyHIST is a Galaxy tool for converting whole-slide pathology images into tiled patch archives.

## Inputs

- A Galaxy dataset collection of pathology images in `.svs`, `.tiff`, or `.tif` format

## Outputs

- A ZIP archive containing the tiled outputs for each input slide

## Typical use

1. Upload whole-slide images to Galaxy.
2. Group them into a list collection.
3. Run the tiling tool to generate tiles for downstream embedding extraction or quality-control workflows.

## Notes for deployers

- The tool wrapper references `quay.io/goeckslab/galaxy-tiler:1.0.0`.
- The container relies on OpenSlide and PyHIST.
