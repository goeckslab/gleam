import base64
import json
import logging
import os
import pickle
import re
import sys
from io import BytesIO

import pandas as pd
from ludwig.api import LudwigModel
from ludwig.experiment import cli
from ludwig.globals import (
    DESCRIPTION_FILE_NAME,
    PREDICTIONS_PARQUET_FILE_NAME,
    TEST_STATISTICS_FILE_NAME,
    TRAIN_SET_METADATA_FILE_NAME
)
from ludwig.utils.data_utils import get_split_path
from ludwig.visualize import get_visualizations_registry
from model_unpickler import SafeUnpickler
from utils import (
    encode_image_to_base64,
    get_html_closing,
    get_html_template
)

try:  # pragma: no cover - optional dependency in runtime containers
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


logging.basicConfig(level=logging.DEBUG)

LOG = logging.getLogger(__name__)

setattr(pickle, 'Unpickler', SafeUnpickler)

# visualization
output_directory = None
for ix, arg in enumerate(sys.argv):
    if arg == "--output_directory":
        output_directory = sys.argv[ix + 1]
        break

viz_output_directory = os.path.join(output_directory, "visualizations")


def get_output_feature_name(experiment_dir, output_feature=0):
    """Helper function to extract specified output feature name.

    :param experiment_dir: Path to the experiment directory
    :param output_feature: position of the output feature the description.json
    :return output_feature_name: name of the first output feature name
                        from the experiment
    """
    if os.path.exists(os.path.join(experiment_dir, DESCRIPTION_FILE_NAME)):
        description_file = os.path.join(experiment_dir, DESCRIPTION_FILE_NAME)
        with open(description_file, "rb") as f:
            content = json.load(f)
        output_feature_name = \
            content["config"]["output_features"][output_feature]["name"]
        dataset_path = content["dataset"]
        return output_feature_name, dataset_path
    return None, None


def check_file(file_path):
    """Check if the file exists; return None if it doesn't."""
    return file_path if os.path.exists(file_path) else None


def make_visualizations(ludwig_output_directory_name):
    ludwig_output_directory = os.path.join(
        output_directory,
        ludwig_output_directory_name,
    )
    visualizations = [
        "confidence_thresholding",
        "confidence_thresholding_data_vs_acc",
        "confidence_thresholding_data_vs_acc_subset",
        "confidence_thresholding_data_vs_acc_subset_per_class",
        "confidence_thresholding_2thresholds_2d",
        "confidence_thresholding_2thresholds_3d",
        "binary_threshold_vs_metric",
        "roc_curves",
        "roc_curves_from_test_statistics",
        "calibration_1_vs_all",
        "calibration_multiclass",
        "confusion_matrix",
        "frequency_vs_f1",
        "learning_curves",
    ]

    # Check existence of required files
    training_statistics = check_file(os.path.join(
        ludwig_output_directory,
        "training_statistics.json",
    ))
    test_statistics = check_file(os.path.join(
        ludwig_output_directory,
        TEST_STATISTICS_FILE_NAME,
    ))
    ground_truth_metadata = check_file(os.path.join(
        ludwig_output_directory,
        "model",
        TRAIN_SET_METADATA_FILE_NAME,
    ))
    probabilities = check_file(os.path.join(
        ludwig_output_directory,
        PREDICTIONS_PARQUET_FILE_NAME,
    ))

    output_feature, dataset_path = get_output_feature_name(
        ludwig_output_directory)
    ground_truth = None
    split_file = None
    if dataset_path:
        ground_truth = check_file(dataset_path)
        split_file = check_file(get_split_path(dataset_path))

    if (not output_feature) and (test_statistics):
        test_stat = os.path.join(test_statistics)
        with open(test_stat, "rb") as f:
            content = json.load(f)
        output_feature = next(iter(content.keys()))

    for viz in visualizations:
        viz_func = get_visualizations_registry()[viz]
        try:
            viz_func(
                training_statistics=[training_statistics]
                if training_statistics else [],
                test_statistics=[test_statistics] if test_statistics else [],
                probabilities=[probabilities] if probabilities else [],
                top_n_classes=[0],
                output_feature_name=output_feature if output_feature else "",
                ground_truth_split=2,
                top_k=3,
                ground_truth_metadata=ground_truth_metadata,
                ground_truth=ground_truth,
                split_file=split_file,
                output_directory=viz_output_directory,
                normalize=False,
                file_format="png",
            )
        except Exception as e:
            LOG.info(f"Visualization: {viz}")
            LOG.info(f"Error: {e}")


def convert_parquet_to_csv(ludwig_output_directory_name):
    """Convert the predictions Parquet file to CSV."""
    ludwig_output_directory = os.path.join(
        output_directory, ludwig_output_directory_name)
    parquet_path = os.path.join(
        ludwig_output_directory, "predictions.parquet")
    csv_path = os.path.join(
        ludwig_output_directory, "predictions_parquet.csv")

    try:
        df = pd.read_parquet(parquet_path)
        df.to_csv(csv_path, index=False)
        LOG.info(f"Converted Parquet to CSV: {csv_path}")
    except Exception as e:
        LOG.error(f"Error converting Parquet to CSV: {e}")


def _resolve_dataset_path(dataset_path):
    if not dataset_path:
        return None

    candidates = [dataset_path]

    if not os.path.isabs(dataset_path):
        candidates.extend([
            os.path.join(output_directory, dataset_path),
            os.path.join(os.getcwd(), dataset_path),
        ])

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return os.path.abspath(candidate)

    return None


def _load_dataset_dataframe(dataset_path):
    if not dataset_path:
        return None

    _, ext = os.path.splitext(dataset_path.lower())

    try:
        if ext in {".csv", ".tsv"}:
            sep = "\t" if ext == ".tsv" else ","
            return pd.read_csv(dataset_path, sep=sep)
        if ext == ".parquet":
            return pd.read_parquet(dataset_path)
        if ext == ".json":
            return pd.read_json(dataset_path)
        if ext == ".h5":
            return pd.read_hdf(dataset_path)
    except Exception as exc:
        LOG.warning(f"Unable to load dataset '{dataset_path}': {exc}")

    LOG.warning("Unsupported dataset format for feature importance computation")
    return None


def sanitize_feature_name(name):
    """Mirror Ludwig's get_sanitized_feature_name implementation."""
    return re.sub(r"[(){}.:\"\"\'\'\[\]]", "_", str(name))


def _sanitize_dataframe_columns(dataframe):
    """Rename dataframe columns to Ludwig-sanitized names for explainability."""
    column_map = {col: sanitize_feature_name(col) for col in dataframe.columns}

    sanitized_df = dataframe.rename(columns=column_map)
    if len(set(column_map.values())) != len(column_map.values()):
        LOG.warning(
            "Column name collision after sanitization; feature importance may be unreliable"
        )

    return sanitized_df


def _feature_importance_plot(label_df, label_name, top_n=10):
    """Return base64-encoded bar plot for a label's top-N feature importances."""
    if plt is None or label_df.empty:
        return ""

    top_features = label_df.nlargest(top_n, "abs_importance")
    if top_features.empty:
        return ""

    fig, ax = plt.subplots(figsize=(6, 3 + 0.2 * len(top_features)))
    ax.barh(top_features["feature"], top_features["abs_importance"], color="#3f8fd2")
    ax.set_xlabel("|importance|")
    ax.set_title(f"{label_name} (absolute)")
    ax.invert_yaxis()
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return encoded


def compute_feature_importance(ludwig_output_directory_name,
                               sample_size=200,
                               random_seed=42):
    ludwig_output_directory = os.path.join(
        output_directory, ludwig_output_directory_name)
    model_dir = os.path.join(ludwig_output_directory, "model")

    output_csv_path = os.path.join(
        ludwig_output_directory, "feature_importance.csv")

    if not os.path.exists(model_dir):
        LOG.info("Model directory not found; skipping feature importance computation")
        return

    try:
        ludwig_model = LudwigModel.load(model_dir)
    except Exception as exc:
        LOG.warning(f"Unable to load Ludwig model for explanations: {exc}")
        return

    training_metadata = getattr(ludwig_model, "training_set_metadata", {})

    output_feature_name, dataset_path = get_output_feature_name(
        ludwig_output_directory)

    if not output_feature_name or not dataset_path:
        LOG.warning("Output feature or dataset path missing; skipping feature importance")
        if hasattr(ludwig_model, "close"):
            ludwig_model.close()
        return

    dataset_full_path = _resolve_dataset_path(dataset_path)
    if not dataset_full_path:
        LOG.warning(f"Unable to resolve dataset path '{dataset_path}' for explanations")
        if hasattr(ludwig_model, "close"):
            ludwig_model.close()
        return

    dataframe = _load_dataset_dataframe(dataset_full_path)
    if dataframe is None or dataframe.empty:
        LOG.warning("Dataset unavailable or empty; skipping feature importance")
        if hasattr(ludwig_model, "close"):
            ludwig_model.close()
        return

    dataframe = _sanitize_dataframe_columns(dataframe)

    data_subset = dataframe if len(dataframe) <= sample_size else dataframe.head(sample_size)
    sample_df = dataframe.sample(
        n=min(sample_size, len(dataframe)),
        random_state=random_seed,
        replace=False,
    ) if len(dataframe) > sample_size else dataframe

    try:
        from ludwig.explain.captum import IntegratedGradientsExplainer
    except ImportError as exc:
        LOG.warning(f"Integrated Gradients explainer unavailable: {exc}")
        if hasattr(ludwig_model, "close"):
            ludwig_model.close()
        return

    sanitized_output_feature = sanitize_feature_name(output_feature_name)

    try:
        explainer = IntegratedGradientsExplainer(
            ludwig_model,
            data_subset,
            sample_df,
            sanitized_output_feature,
        )
        explanations = explainer.explain()
    except Exception as exc:
        LOG.warning(f"Unable to compute feature importance: {exc}")
        if hasattr(ludwig_model, "close"):
            ludwig_model.close()
        return

    if hasattr(ludwig_model, "close"):
        try:
            ludwig_model.close()
        except Exception:
            pass

    label_names = []
    target_metadata = {}
    if isinstance(training_metadata, dict):
        target_metadata = training_metadata.get(sanitized_output_feature, {})

    if isinstance(target_metadata, dict):
        if "idx2str" in target_metadata:
            idx2str = target_metadata["idx2str"]
            if isinstance(idx2str, dict):
                def _idx_key(item):
                    idx_key = item[0]
                    try:
                        return (0, int(idx_key))
                    except (TypeError, ValueError):
                        return (1, str(idx_key))

                label_names = [value for key, value in sorted(
                    idx2str.items(), key=_idx_key)]
            else:
                label_names = idx2str
        elif "str2idx" in target_metadata and isinstance(
                target_metadata["str2idx"], dict):
            # invert mapping
            label_names = [label for label, _ in sorted(
                target_metadata["str2idx"].items(),
                key=lambda item: item[1])]

    rows = []
    global_explanation = explanations.global_explanation
    for label_index, label_explanation in enumerate(
            global_explanation.label_explanations):
        if label_names and label_index < len(label_names):
            label_value = str(label_names[label_index])
        elif len(global_explanation.label_explanations) == 1:
            label_value = output_feature_name
        else:
            label_value = str(label_index)

        for feature in label_explanation.feature_attributions:
            rows.append({
                "label": label_value,
                "feature": feature.feature_name,
                "importance": feature.attribution,
                "abs_importance": abs(feature.attribution),
            })

    if not rows:
        LOG.warning("No feature importance rows produced")
        return

    importance_df = pd.DataFrame(rows)
    importance_df.sort_values([
        "label",
        "abs_importance"
    ], ascending=[True, False], inplace=True)

    importance_df.to_csv(output_csv_path, index=False)

    LOG.info(f"Feature importance saved to {output_csv_path}")


def generate_html_report(title, ludwig_output_directory_name):
    plots_html = ""
    plot_files = []
    if os.path.isdir(viz_output_directory):
        plot_files = sorted(os.listdir(viz_output_directory))
    if plot_files:
        plots_html = "<h2>Visualizations</h2>"
    for plot_file in plot_files:
        plot_path = os.path.join(viz_output_directory, plot_file)
        if os.path.isfile(plot_path) and plot_file.endswith((".png", ".jpg")):
            encoded_image = encode_image_to_base64(plot_path)
            plot_title = os.path.splitext(plot_file)[0].replace("_", " ")
            plots_html += (
                f'<div class="plot">'
                f'<h3>{plot_title}</h3>'
                '<img src="data:image/png;base64,'
                f'{encoded_image}" alt="{plot_file}">'
                f'</div>'
            )

    feature_importance_html = ""
    importance_path = os.path.join(
        output_directory,
        ludwig_output_directory_name,
        "feature_importance.csv",
    )
    if os.path.exists(importance_path):
        try:
            importance_df = pd.read_csv(importance_path)
            if not importance_df.empty:
                top_rows = (
                    importance_df
                    .sort_values(["label", "abs_importance"], ascending=[True, False])
                    .groupby("label", as_index=False)
                    .head(5)
                )
                plot_sections = []
                for label in sorted(importance_df["label"].unique()):
                    encoded_plot = _feature_importance_plot(
                        importance_df[importance_df["label"] == label], label
                    )
                    if encoded_plot:
                        plot_sections.append(
                            f'<div class="plot feature-importance-plot">'
                            f'<h3>Top features for {label}</h3>'
                            f'<img src="data:image/png;base64,{encoded_plot}" '
                            f'alt="Feature importance plot for {label}">'
                            f'</div>'
                        )
                explanation_text = (
                    "<p>Feature importance scores come from Ludwig's Integrated Gradients explainer. "
                    "It interpolates between each example and a neutral baseline sample, summing "
                    "the change in the model output along that path. Higher absolute values "
                    "indicate stronger influence for the given label.</p>"
                )
                feature_importance_html = (
                    "<h2>Feature Importance</h2>"
                    + explanation_text
                    + top_rows.to_html(index=False, border=0, classes="feature-importance-table")
                    + "".join(plot_sections)
                )
        except Exception as exc:
            LOG.info(f"Unable to embed feature importance table: {exc}")

    # Generate the full HTML content
    feature_section = feature_importance_html or "<p>No feature importance artifacts were generated.</p>"
    viz_section = plots_html or "<p>No visualizations were generated.</p>"
    tabs_style = """
    <style>
        .tabs {
            display: flex;
            border-bottom: 2px solid #ccc;
            margin-top: 20px;
            margin-bottom: 1rem;
        }
        .tablink {
            padding: 9px 18px;
            cursor: pointer;
            border: 1px solid #ccc;
            border-bottom: none;
            background: #f9f9f9;
            margin-right: 5px;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            font-size: 0.95rem;
            font-weight: 500;
            font-family: Arial, sans-serif;
            color: #4A4A4A;
        }
        .tablink.active {
            background: #ffffff;
            font-weight: bold;
        }
        .tabcontent {
            border: 1px solid #ccc;
            border-top: none;
            padding: 20px;
            display: none;
        }
        .tabcontent.active {
            display: block;
        }
    </style>
    """
    tabs_script = """
    <script>
        function openTab(evt, tabId) {
            var i, tabcontent, tablinks;
            tabcontent = document.getElementsByClassName("tabcontent");
            for (i = 0; i < tabcontent.length; i++) {
                tabcontent[i].style.display = "none";
                tabcontent[i].classList.remove("active");
            }
            tablinks = document.getElementsByClassName("tablink");
            for (i = 0; i < tablinks.length; i++) {
                tablinks[i].classList.remove("active");
            }
            var current = document.getElementById(tabId);
            if (current) {
                current.style.display = "block";
                current.classList.add("active");
            }
            if (evt && evt.currentTarget) {
                evt.currentTarget.classList.add("active");
            }
        }
        document.addEventListener("DOMContentLoaded", function() {
            openTab({currentTarget: document.querySelector(".tablink")}, "viz-tab");
        });
    </script>
    """
    tabs_html = f"""
    <div class="tabs">
        <button class="tablink active" onclick="openTab(event, 'viz-tab')">Visualizations</button>
        <button class="tablink" onclick="openTab(event, 'feature-tab')">Feature Importance</button>
    </div>
    <div id="viz-tab" class="tabcontent active">
        {viz_section}
    </div>
    <div id="feature-tab" class="tabcontent">
        {feature_section}
    </div>
    """
    html_content = f"""
    {get_html_template()}
        <h1>{title}</h1>
        {tabs_style}
        {tabs_html}
        {tabs_script}
    {get_html_closing()}
    """

    # Save the HTML report
    title: str
    report_name = title.lower().replace(" ", "_")
    report_path = os.path.join(output_directory, f"{report_name}_report.html")
    with open(report_path, "w") as report_file:
        report_file.write(html_content)

    LOG.info(f"HTML report generated at: {report_path}")


if __name__ == "__main__":

    cli(sys.argv[1:])

    ludwig_output_directory_name = "experiment_run"

    make_visualizations(ludwig_output_directory_name)
    convert_parquet_to_csv(ludwig_output_directory_name)
    compute_feature_importance(ludwig_output_directory_name)
    generate_html_report("Ludwig Experiment", ludwig_output_directory_name)
