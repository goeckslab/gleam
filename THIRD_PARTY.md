# Third-Party Software, Models, and Containers

This repository is distributed under the MIT License, but the software stack used by individual GLEAM tools includes third-party components with their own licenses, terms, and redistribution conditions. Users and deployers are responsible for reviewing those upstream licenses before redistribution or commercial deployment.

## Core ML frameworks and tool backends

- PyCaret
  - Used by `tools/tabularlearner`
  - Installed in [tools/tabularlearner/Dockerfile](tools/tabularlearner/Dockerfile)
- Ludwig
  - Used by `tools/galaxy-ludwig` and `tools/imagelearner`
  - Referenced by [tools/galaxy-ludwig/ludwig_macros.xml](tools/galaxy-ludwig/ludwig_macros.xml) and [tools/imagelearner/image_learner.xml](tools/imagelearner/image_learner.xml)
- AutoGluon
  - Used by `tools/multimodallearner`
  - Installed in [tools/multimodallearner/Dockerfile](tools/multimodallearner/Dockerfile)
- PyHIST
  - Used by `tools/galaxy-tiler`
  - Pulled from its upstream Git repository in [tools/galaxy-tiler/Docker/Dockerfile](tools/galaxy-tiler/Docker/Dockerfile)
- PyTorch and TorchVision
  - Used by the image, multimodal, embedding, and MIL-related tools

## Pretrained models and model weights

- TorchVision pretrained backbones
  - Exposed by `Image Learner` and `Image Embedding Extraction`
  - Subject to the license terms of TorchVision, the corresponding upstream model definitions, and weight distribution policies
- MetaFormer family checkpoints
  - Supported by `tools/imagelearner`
  - Weight provenance depends on the upstream model provider and checkpoint source
- Hugging Face-hosted models
  - Some text and image backbones may be downloaded at runtime
  - Individual model cards define their own license and usage restrictions
- GPFM pathology foundation model
  - Exposed by `tools/galaxy-embedding_extractor`
  - Review the upstream model card and any weight redistribution conditions before production deployment

## Container images referenced by wrappers

- `quay.io/goeckslab/galaxy-ludwig-gpu:0.10.1`
- `quay.io/goeckslab/galaxy-ludwig:0.10.3`
- `quay.io/goeckslab/multimodal-learner:1.4.0`
- `quay.io/goeckslab/galaxy-tiler:1.0.0`
- `quay.io/goeckslab/milbag:1.0.0`
- `quay.io/goeckslab/galaxy-ludwig-gpu:extract_embeddings`

These images package additional operating-system and Python dependencies that are not individually inventoried in the repository root. The Dockerfiles included in this repository are the primary provenance record for locally built images:

- [tools/tabularlearner/Dockerfile](tools/tabularlearner/Dockerfile)
- [tools/multimodallearner/Dockerfile](tools/multimodallearner/Dockerfile)
- [tools/galaxy-ludwig/Docker/galaxy_ludwig/Dockerfile](tools/galaxy-ludwig/Docker/galaxy_ludwig/Dockerfile)
- [tools/galaxy-ludwig/Docker/galaxy_ludwig_ray_gpu/Dockerfile](tools/galaxy-ludwig/Docker/galaxy_ludwig_ray_gpu/Dockerfile)
- [tools/galaxy-embedding_extractor/Docker/Dockerfile](tools/galaxy-embedding_extractor/Docker/Dockerfile)
- [tools/galaxy-tiler/Docker/Dockerfile](tools/galaxy-tiler/Docker/Dockerfile)
- [tools/galaxy-mil_bag/Docker/Dockerfile](tools/galaxy-mil_bag/Docker/Dockerfile)

## Additional bundled or fetched dependencies

- `smart-report`
  - Installed from a pinned Git commit in the Galaxy-Ludwig Dockerfiles
- `model-unpickler`
  - Installed from the Goecks Lab GitHub repository in the Galaxy-Ludwig Dockerfiles
- OpenSlide
  - Used by the pathology tiling workflow

## Maintainer guidance

Before each tagged release:

1. Review newly added dependencies and model providers.
2. Confirm that referenced container tags still resolve and match the intended software versions.
3. Confirm that any runtime-downloaded model weights can be legally redistributed or documented as external downloads.
4. Update this file when new third-party frameworks, containers, or pretrained model families are introduced.
