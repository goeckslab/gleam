# Image Learner

Image Learner is a Galaxy wrapper for training and evaluating image classification or regression models from image archives plus metadata tables.

## Backend

- Ludwig-based training flow
- TorchVision model zoo support
- MetaFormer family support for selected custom backbones

## Inputs

- A metadata CSV file that identifies image paths and target labels
- A ZIP archive containing the referenced images
- A selected model backbone
- Optional task-specific metrics and training overrides
- Optional sample ID column for leakage-aware splitting

## Outputs

- A Galaxy composite model dataset
- An HTML report with training curves and evaluation plots
- Derived prediction and metric assets stored with the model output

## Typical use

1. Upload a ZIP archive of images and a CSV file with image paths and labels.
2. Choose classification or regression behavior through the wrapper parameters.
3. Pick a backbone and, if desired, enable pretrained weights or fine-tuning.
4. Run the tool and inspect the generated report and model artifact.

## Notes for deployers

- The wrapper references `quay.io/goeckslab/galaxy-ludwig-gpu:0.10.1`.
- Some backbones may download pretrained weights during execution.
- GPU-backed execution is recommended for larger training jobs.
