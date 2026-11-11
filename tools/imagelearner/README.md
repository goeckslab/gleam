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
- Optional sample ID column for leakage-aware splitting and grouped cross-validation folds
- Optional k-fold cross-validation and/or several random seeds, trained in one run and averaged

## Outputs

- A Galaxy composite model dataset
- An HTML report with training curves and evaluation plots
- Derived prediction and metric assets stored with the model output
- For ensemble runs: averaged predictions, `ensemble_summary.json`, and one `members/<fold>_<seed>/` folder per trained member (predictions and statistics, no weights) inside the artifacts ZIP

## Cross-validation and ensembles

Under "Customize Default Settings" the wrapper can train several models per run:

- *Number of random seeds* trains that many models on the same split (seeds `random_seed`, `random_seed+1`, ...).
- *k-fold cross-validation* trains one model per fold. "Validation folds" keep the test split fixed and rotate the folds through the validation set; "test folds" score every sample exactly once out-of-fold and refuse a test split supplied in the input CSV.
- Both options combine to folds x seeds members (at most 50). Class probabilities (or regression values) are averaged across the members that did not train on a row, and the report lists every member, their mean ± SD, and the averaged ensemble. Member rows are scored from each member's own predictions at the ensemble's decision threshold, so they compare like for like with the ensemble row.
- The run stops before training when a class has fewer rows than folds, when there are fewer sample IDs than folds, or when two numeric labels do not include exactly one 0 (Ludwig would read both as True).

Members are trained sequentially, so plan GPU time accordingly. The ludwig_model output holds the first trained member's weights, and the report shows that member's own test metrics and threshold; the other members' predictions and statistics live in the artifacts ZIP, and their weights are deleted during the run to bound disk use.

Seeds on the same split reuse the prepared datasets in their original row order. Different folds use separate caches. Validation predictions reuse the trained model and prepared validation data. Automatic batch sizes are reused only for models without mutable persistent buffers: models with buffers such as BatchNorm statistics retain the original probes, which can change those statistics. If reused sizes run out of memory, the member retries with automatic tuning. Explicit batch sizes remain unchanged.

## Typical use

1. Upload a ZIP archive of images and a CSV file with image paths and labels.
2. Choose classification or regression behavior through the wrapper parameters.
3. Pick a backbone and, if desired, enable pretrained weights or fine-tuning.
4. Run the tool and inspect the generated report and model artifact.

## Notes for deployers

- The wrapper references `quay.io/goeckslab/galaxy-ludwig-gpu:0.10.1`.
- Some backbones may download pretrained weights during execution.
- GPU-backed execution is recommended for larger training jobs.

## Binary metric regression tests

Run `python tools/imagelearner/test_binary_metrics_integration.py` in the container
declared by `image_learner.xml`. This uses real Ludwig and TorchMetrics, including a
small CPU image-training run without pretrained weights. It checks decision-threshold
boundaries and verifies saved training, validation, and test metrics against the
model's predictions. The PR workflow runs it when Image Learner changes.

Run this test in its own process: the lightweight backend/workflow tests install
Ludwig module stubs and must not share its interpreter.
