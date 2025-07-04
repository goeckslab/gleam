[![Galaxy Tool Linting and Tests for push and PR](https://github.com/goeckslab/gleam/actions/workflows/pr.yaml/badge.svg?branch=main)](https://github.com/goeckslab/gleam/actions/workflows/pr.yaml/badge.svg)
[![Weekly global Tool Linting and Tests](https://github.com/goeckslab/gleam/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/goeckslab/gleam/actions/workflows/ci.yaml/badge.svg)

# GLEAM: Galaxy Learning and Modeling

GLEAM (Galaxy Learning and Modeling) is a suite of machine learning tools for the [Galaxy](https://usegalaxy.org/) platform. Developed by the [Goecks Lab](https://goeckslab.org/), GLEAM empowers researchers to train models, generate predictions, and produce reproducible reports—all from a user-friendly interface without writing code.

## Features
- Modern best practices for machine learning
- Reproducible and scalable workflows
- Machine learning support for diverse data types: tabular, image, text, categorical, and more
- Deep learning via Ludwig and automated ML via PyCaret
- Easy installation in Galaxy via XML wrappers
- Auto-generated visual reports

## Available Tools

### 1. TabularLearner

Machine learning for structured tabular datasets using [PyCaret](https://pycaret.org/).

- Train classification and regression models
- Evaluate performance and extract feature importance
- Generate predictions on new datasets
- Create interactive HTML reports

### 2. ImageLearner

Deep learning-based image classification using [Ludwig](https://ludwig.ai/).

- input files: Zip file with images  and csv with metadata
- Tasks: classification
- Models available: ResNet, EfficientNet, VGG, Shufflenet, Vit, AlexNet and More...
- Output: Ludwig_model file, a report in the form of an HTML file (with learning curves, confusion matrices, and etc...), and a collection of CSV/json/png files containing the predictions, experiment stats and visualizations.

### 3. Galaxy-Ludwig

General-purpose interface to Ludwig's full machine learning capabilities.

- Train and evaluate models on structured input (tabular, image, text, etc.)
- Expose Ludwig’s flexible configuration system
- Ideal for users needing advanced model customization

### 4. Galaxy-Digital Pathology Processing

Set of three specialized tools designed to transforms raw, large pathology images into a structured format, enabling the application of best practices for model development and ensuring data readiness for robust and efficient training.

- Image Tiler: Accepts .svs image format, which is the most common proprietary format for digital pathology whole slide images.
- Embedding Extractor: Leverages pre-trained models from the TorchVision foundation models for feature extraction (for example, ResNet50, EfficientNet_B0, DenseNet121).
- Multiple Instance Learning (MIL) Bag Processor: Facilitates the aggregation of embeddings from individual image tiles into "bags" using various pooling techniques (such as Max Pooling or Attention Pooling).

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/goeckslab/gleam.git
   ```

2. Add entries for each tool in your tool_conf.xml of your galaxy instance:
    ```xml
    <tool file="<path-to-your-local-tabularlearner/tabular_learner.xml>" />
    <tool file="<path-to-your-local-imagelearner/image_learner_train.xml>" />
    <tool file="<path-to-your-local-galaxy-ludwig/ludwig_train.xml>" />
    ```


## Contributing
We welcome contributions. To propose new tools, report bugs, or suggest improvements:

1. Fork the repository

2. Create a feature branch

3. Commit and test your changes

4. Submit a pull request


