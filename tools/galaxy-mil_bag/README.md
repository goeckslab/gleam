# Bagging Embeddings Processor

Bagging Embeddings Processor is a Galaxy tool for transforming per-image embeddings into bag-level datasets for multiple instance learning workflows.

## Inputs

- An embeddings CSV file with a `sample_name` column
- A metadata CSV file with labels keyed by `sample_name`
- Split proportions, bag size, and pooling configuration
- Optional leakage control, balancing, and Ludwig-format output settings

## Outputs

- A CSV file containing aggregated bag-level examples

## Typical use

1. Generate embeddings from pathology tiles or other image collections.
2. Supply matching metadata with labels.
3. Build bags using the desired pooling strategy.
4. Use the resulting table in downstream Ludwig or other MIL-compatible workflows.

## Notes for deployers

- The tool wrapper references `quay.io/goeckslab/milbag:1.0.0`.
- GPU acceleration is optional and only applies when the execution environment supports it.
