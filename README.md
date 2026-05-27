[![Galaxy Tool Linting and Tests for push and PR](https://github.com/goeckslab/gleam/actions/workflows/pr.yaml/badge.svg?branch=main)](https://github.com/goeckslab/gleam/actions/workflows/pr.yaml)
[![Weekly global Tool Linting and Tests](https://github.com/goeckslab/gleam/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/goeckslab/gleam/actions/workflows/ci.yaml)

# GLEAM: Galaxy Learning and Modeling

GLEAM (Galaxy Learning and Modeling) is a maintained suite of Galaxy tools for no-code and low-code machine learning workflows. The repository is the software-maintenance home for the GLEAM workbench: it contains Galaxy wrappers, Python entrypoints, test assets, and container build definitions for tool development and deployment in Galaxy.

Project website: [https://goeckslab.github.io/gleam/](https://goeckslab.github.io/gleam/)

This repository is not intended to be a manuscript-specific analysis archive. Paper-specific benchmark datasets, figure-generation notebooks, and result tables should live in separate companion repositories or public data archives referenced by the corresponding publication.

## Tool Families

### Tabular Learner
- Backend: PyCaret
- Tasks: classification and regression on structured tabular data
- Outputs: trained model artifact, best-model parameters, HTML evaluation report
- Docs: [tools/tabularlearner/README.md](tools/tabularlearner/README.md)

### Image Learner
- Backend: Ludwig with TorchVision and MetaFormer model support
- Tasks: image classification and regression from image ZIP archives plus metadata CSV files
- Outputs: trained model artifact, HTML report, metrics/prediction assets
- Docs: [tools/imagelearner/README.md](tools/imagelearner/README.md)

### Multimodal Learner
- Backend: AutoGluon Multimodal
- Tasks: classification and regression using tabular, text, and image inputs
- Outputs: HTML report, metrics JSON, training config YAML
- Docs: [tools/multimodallearner/README.md](tools/multimodallearner/README.md)

### Galaxy-Ludwig
- Backend: Ludwig
- Tasks: general-purpose model configuration, training, evaluation, prediction, hyperparameter search, and visualization
- Outputs: Ludwig model artifacts, metrics, reports, plots, and configuration files
- Docs: [tools/galaxy-ludwig/README.md](tools/galaxy-ludwig/README.md)

### Digital Pathology Utilities
- Image tiling with PyHIST
- Embedding extraction with TorchVision and pathology-oriented backbones
- MIL bag construction from embedding tables
- Docs:
  - [tools/galaxy-tiler/README.md](tools/galaxy-tiler/README.md)
  - [tools/galaxy-embedding_extractor/README.md](tools/galaxy-embedding_extractor/README.md)
  - [tools/galaxy-mil_bag/README.md](tools/galaxy-mil_bag/README.md)

## Installation

### Option 1: Install released tools from the Galaxy ToolShed

GLEAM tools are published for Galaxy administrators through the [Galaxy ToolShed](https://toolshed.g2.bx.psu.edu/).

1. Sign in to your Galaxy instance as an administrator.
2. Open `Admin` and then `Install and Uninstall` or `Manage Tools`.
3. Search for tool suites published by the `goeckslab` owner.
4. Install the suites you need from their exact ToolShed repositories:
   - [`suite_tabular_learner`](https://toolshed.g2.bx.psu.edu/repository/view_repository?id=babd0d26f1edc4a6)
   - [`suite_imagelearner`](https://toolshed.g2.bx.psu.edu/repository/view_repository?id=d141b70930a8ae9e)
   - [`suite_multimodallearner`](https://toolshed.g2.bx.psu.edu/repository/view_repository?id=02626b9dcabebff1)
   - [`suite_ludwig`](https://toolshed.g2.bx.psu.edu/repository/view_repository?id=7766a9382c5a05e6)
   - [`suite_tiler`](https://toolshed.g2.bx.psu.edu/repository/view_repository?id=07dd6bd237b21b26)
   - [`suite_embedding_extractor`](https://toolshed.g2.bx.psu.edu/repository/view_repository?id=c75060934e8e5c2c)
   - [`suite_mil_bag`](https://toolshed.g2.bx.psu.edu/repository/view_repository?id=1c742723b8e1449e)
5. Let Galaxy resolve the declared dependencies and restart the server if your deployment requires it.

This is the recommended path for production Galaxy instances because it tracks released tool definitions rather than an arbitrary development snapshot.

### Option 2: Install directly from this repository for development

Use this path if you are developing GLEAM itself, testing local modifications, or validating wrapper behavior before a ToolShed release.

1. Clone the repository:

   ```bash
   git clone https://github.com/goeckslab/gleam.git
   cd gleam
   ```

2. Copy or symlink the tool directories you want into your Galaxy `tools/` tree.

3. Register the desired wrappers in your Galaxy tool panel configuration. For example:

   ```xml
   <section id="gleam" name="GLEAM">
     <tool file="gleam/tools/tabularlearner/tabular_learner.xml" />
     <tool file="gleam/tools/tabularlearner/pycaret_predict.xml" />
     <tool file="gleam/tools/imagelearner/image_learner.xml" />
     <tool file="gleam/tools/multimodallearner/multimodal_learner.xml" />
     <tool file="gleam/tools/galaxy-ludwig/ludwig_train.xml" />
     <tool file="gleam/tools/galaxy-ludwig/ludwig_evaluate.xml" />
     <tool file="gleam/tools/galaxy-ludwig/ludwig_predict.xml" />
     <tool file="gleam/tools/galaxy-tiler/tiling_pyhist.xml" />
     <tool file="gleam/tools/galaxy-embedding_extractor/pytorch_embedding.xml" />
     <tool file="gleam/tools/galaxy-mil_bag/mil_bag.xml" />
   </section>
   ```

4. Ensure your Galaxy deployment can execute the containers referenced by the wrappers. Most GLEAM tools expect Docker or another Galaxy-supported container backend.

5. Restart Galaxy and verify that the tools load without wrapper errors.

### Container and runtime notes

- Several tools use prebuilt images from `quay.io/goeckslab/...`.
- GPU-backed tools require compatible CUDA drivers and a Galaxy job configuration that permits GPU/container execution.
- Some models download pretrained weights at runtime on first use. For reproducible production deployments, pre-populate caches or pin the corresponding container image and model source.

## Testing and CI

The repository includes Galaxy wrapper tests and CI workflows under [.github/workflows](.github/workflows). Local development typically relies on `planemo` plus wrapper-specific test data already versioned in `tools/*/test-data`.

## Citation and Releases

- Citation metadata is provided in [CITATION.cff](CITATION.cff) and [codemeta.json](codemeta.json).
- Release history is tracked in [CHANGELOG.md](CHANGELOG.md).
- Maintainers and author credit are listed in [AUTHORS.md](AUTHORS.md) and [MAINTAINERS.md](MAINTAINERS.md).
- Third-party software, model, and container provenance is summarized in [THIRD_PARTY.md](THIRD_PARTY.md).
- The archival release workflow is documented in [RELEASE.md](RELEASE.md).

## Contributing

Contributions that improve Galaxy wrapper quality, testing, documentation, and container reproducibility are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Run the relevant wrapper and CI tests.
4. Open a pull request with a clear description of the user-facing impact.
