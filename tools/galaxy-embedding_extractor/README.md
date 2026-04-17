# Image Embedding Extraction

Image Embedding Extraction is a Galaxy tool for converting image archives into feature tables using pretrained vision backbones.

## Inputs

- A ZIP archive of images
- A selected embedding backbone
- Optional embedding normalization
- An optional output format compatible with Ludwig

## Outputs

- A CSV file where each row corresponds to an input image and the remaining columns contain embedding values

## Typical use

1. Run the tool on an image ZIP archive.
2. Choose a backbone such as ResNet, EfficientNet, ViT, or GPFM.
3. Feed the resulting embedding table into downstream tools such as the MIL bag processor or a tabular learner.

## Notes for deployers

- The tool wrapper references `quay.io/goeckslab/galaxy-ludwig-gpu:extract_embeddings`.
- Some model choices may download pretrained weights at first use.
