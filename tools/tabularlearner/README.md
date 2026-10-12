# Tabular Learner Tools

This repository contains two machine learning tools for working with tabular data in the Gleam framework:

## 1. Tabular Learner

A comprehensive tool for training and evaluating multiple machine learning models on tabular datasets.

### Features:
- Supports both classification and regression tasks
- Automatically compares multiple algorithms to find the best model
- Extensive customization options:
  - Data normalization
  - Feature selection
  - Cross-validation
  - Outlier removal
  - Multicollinearity handling
  - Polynomial feature generation
  - Class imbalance correction
- Outputs detailed HTML reports with performance metrics and visualizations
- Saves the best model for later use

## 2. PyCaret Predictor/Evaluator

A companion tool for making predictions and evaluating trained models on new data.

### Features:
- Works with models trained by Tabular Learner
- Supports both classification and regression tasks
- Generates predictions on new data
- Creates evaluation reports when target values are provided
- Outputs predictions in CSV format

## Workflow

These tools are designed to work together:
1. Use **Tabular Learner** to train and find the best model for your dataset
2. Use **PyCaret Predictor/Evaluator** to apply your trained model to new data

Both tools are powered by [PyCaret](https://pycaret.org/), an open-source machine learning library that automates the ML workflow.

## SHAP failures after training

Earlier versions passed up to 100 sampled test rows as the background for
`tree_path_dependent` explanations. If those rows did not reach every tree leaf,
SHAP raised a coverage error and the tool retried with interventional SHAP.
Tree explanations now use the fitted model's training leaf counts without a
background sample and do not retry coverage errors with interventional SHAP.
For LightGBM this enables its native `pred_contrib=True` calculation.

The old retry can trigger a known native crash with very wide gene matrices:
SHAP 0.44.1 stores interventional feature indexes in a signed 16-bit integer, so
a fitted split on a zero-based column index of 32,768 or higher can overflow.
[SHAP PR #3990](https://github.com/shap/shap/pull/3990) fixes this in SHAP 0.47.0.
PyCaret 3.3.2's analysis extra requests SHAP `~=0.44.0`, so upgrading SHAP to that
release requires resolving the dependency constraint and validating the rebuilt
environment. Check the actual failing Galaxy job's container with:

```sh
python -m pip show pycaret shap lightgbm numpy explainerdashboard
```

This index bug is a candidate cause of the segmentation fault, not proof of the
cause for a particular dataset. Confirm the transformed feature count and model
split indexes as well as the deployed package versions. A segmentation fault
cannot be caught by Python's `except Exception`; the fix avoids the affected
retry rather than recovering from a native crash. Ordinary SHAP construction,
calculation, or plotting errors produce a report note while retaining native
model importance.

SHAP rows are capped at 200 in the training report. The top-feature plotting
limit controls display only: SHAP receives every transformed feature in the
fitted model's order. Selecting the 1,000 most variable genes can reduce cost,
but requires retraining with the selected features. Fit selection on training
data (inside cross-validation when evaluating it) and apply the same selection
to held-out and prediction data; do not truncate an already fitted model's SHAP
input.
