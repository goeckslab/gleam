import base64
import logging
from typing import Optional

import numpy as np

logging.basicConfig(level=logging.DEBUG)
LOG = logging.getLogger(__name__)


def get_html_template() -> str:
    return """
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Model Training Report</title>
        <style>
          body {
              font-family: Arial, sans-serif;
              margin: 0;
              padding: 20px;
              background-color: #f4f4f4;
          }
          /* allow horizontal scrolling if content overflows */
          .container {
              max-width: 800px;
              margin: auto;
              background: white;
              padding: 20px;
              box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
              overflow-x: auto;
          }

          h1 {
              text-align: center;
              color: #333;
          }
          h2 {
              border-bottom: 2px solid #4CAF50;
              color: #4CAF50;
              padding-bottom: 5px;
          }

          /* wrapper for tables to allow individual horizontal scroll */
          .table-wrapper {
              overflow-x: auto;
              margin: 1rem 0;
          }

          /* revert table styling to full borders */
          table {
              width: 100%;
              border-collapse: collapse;
              margin: 20px 0;
          }
          table, th, td {
              border: 1px solid #ddd;
          }
          th, td {
              padding: 8px;
              text-align: left;
          }
          th {
              background-color: #4CAF50;
              color: white;
          }

          /* Center specific numeric columns */
          .table-dataset-overview td:nth-child(n+2),
          .table-dataset-overview th:nth-child(n+2) {
              text-align: center;
          }
          .table-perf-summary td:nth-child(n+2),
          .table-perf-summary th:nth-child(n+2) {
              text-align: center;
          }
          .table-model-comparison td:nth-child(n+2),
          .table-model-comparison th:nth-child(n+2) {
              text-align: center;
          }
          .table-model-comparison td:first-child,
          .table-model-comparison th:first-child {
              text-align: left;
          }
          .table-cv-fold-allocation td,
          .table-cv-fold-allocation th {
              text-align: center;
          }
          .table-setup-params td:nth-child(2),
          .table-setup-params th:nth-child(2) {
              text-align: center;
          }
          .table-hyperparams td:nth-child(2),
          .table-hyperparams th:nth-child(2) {
              text-align: center;
          }
          .table-fi-scope td:nth-child(2),
          .table-fi-scope th:nth-child(2) {
              text-align: center;
          }
          .report-footnote {
              margin: -0.5rem 0 1rem;
              color: #666;
              font-size: 0.88rem;
              line-height: 1.35;
          }
          .report-notice {
              margin: 1rem 0;
              padding: 0.75rem 1rem;
              border-left: 4px solid #4CAF50;
              background: #eef8ef;
              color: #2e5f31;
              line-height: 1.4;
          }

          .plot {
              text-align: center;
              margin: 20px 0;
          }
          .validation-plot-title {
              text-align: left;
          }
          .plot img {
              max-width: 100%;
              height: auto;
          }

          .tabs {
              display: flex;
              align-items: center;
              border-bottom: 2px solid #ccc;
              margin-bottom: 1rem;
          }
          .tab {
              padding: 10px 20px;
              cursor: pointer;
              border: 1px solid #ccc;
              border-bottom: none;
              background: #f9f9f9;
              margin-right: 5px;
              border-top-left-radius: 8px;
              border-top-right-radius: 8px;
          }
          .tab.active {
              background: white;
              font-weight: bold;
          }

          .tab-content {
              display: none;
              padding: 20px;
              border: 1px solid #ccc;
              border-top: none;
              background: white;
          }
          .tab-content.active {
              display: block;
          }

          .help-btn {
              margin-left: auto;
              padding: 6px 12px;
              font-size: 0.9rem;
              border: 1px solid #4CAF50;
              border-radius: 4px;
              background: #4CAF50;
              color: white;
              cursor: pointer;
          }

          /* sortable table header arrows */
          table.sortable th {
              position: relative;
              padding-right: 20px; /* room for the arrow */
              cursor: pointer;
          }
          table.sortable th::after {
              content: '↕';
              position: absolute;
              right: 8px;
              opacity: 0.4;
              transition: opacity 0.2s;
          }
          table.sortable th:hover::after {
              opacity: 0.7;
          }
          table.sortable th.sorted-asc::after {
              content: '↑';
              opacity: 1;
          }
          table.sortable th.sorted-desc::after {
              content: '↓';
              opacity: 1;
          }
        </style>
    </head>
    <body>
    <div class="container">
    """


def get_html_closing() -> str:
    return """
    </div>
    <script>
    document.addEventListener('DOMContentLoaded', () => {
      document.querySelectorAll('table.sortable').forEach(table => {
        const getCellValue = (row, idx) =>
          row.children[idx].innerText.trim() || '';

        const comparer = (idx, asc) => (a, b) => {
          const v1 = getCellValue(asc ? a : b, idx);
          const v2 = getCellValue(asc ? b : a, idx);
          const n1 = parseFloat(v1), n2 = parseFloat(v2);
          if (!isNaN(n1) && !isNaN(n2)) return n1 - n2;
          return v1.localeCompare(v2);
        };

        table.querySelectorAll('th').forEach((th, idx) => {
          let asc = true;
          th.addEventListener('click', () => {
            // sort rows
            const tbody = table.tBodies[0];
            Array.from(tbody.rows)
              .sort(comparer(idx, asc))
              .forEach(row => tbody.appendChild(row));
            // update arrow classes
            table.querySelectorAll('th').forEach(h => {
              h.classList.remove('sorted-asc','sorted-desc');
            });
            th.classList.add(asc ? 'sorted-asc' : 'sorted-desc');
            asc = !asc;
          });
        });
      });
    });
    </script>
    </body>
    </html>
    """


def build_tabbed_html(
    summary_html: str,
    test_html: str,
    feature_html: Optional[str],
    explainer_html: Optional[str] = None,
    config_html: Optional[str] = None,
    summary_tab_label: str = "Validation Summary",
    test_tab_label: str = "Test Summary",
    feature_tab_label: str = "Feature Importance",
) -> str:
    """
    Render the tabbed sections and an always-visible Help button.
    """
    # Tabs header
    tabs = ['<div class="tabs">']
    default_active = "summary"
    if config_html:
        default_active = "config"
        tabs.append(
            '<div class="tab active" onclick="showTab(\'config\')">Experiment Summary</div>'
        )
        tabs.append(
            f'<div class="tab" onclick="showTab(\'summary\')">{summary_tab_label}</div>'
        )
    else:
        tabs.append(
            f'<div class="tab active" onclick="showTab(\'summary\')">{summary_tab_label}</div>'
        )
    tabs.extend([
        f'<div class="tab" onclick="showTab(\'test\')">{test_tab_label}</div>',
    ])
    if feature_html is not None:
        tabs.append(
            f'<div class="tab" onclick="showTab(\'feature\')">{feature_tab_label}</div>'
        )
    if explainer_html:
        tabs.append(
            '<div class="tab" onclick="showTab(\'explainer\')">Explainer Plots</div>'
        )
    tabs.append('<button id="openMetricsHelp" class="help-btn">Help</button>')
    tabs.append("</div>")
    tabs_section = "\n".join(tabs)

    # Content
    contents = []
    if config_html:
        contents.append(
            f'<div id="config" class="tab-content {"active" if default_active == "config" else ""}">{config_html}</div>'
        )
    contents.append(
        f'<div id="summary" class="tab-content {"active" if default_active == "summary" else ""}">{summary_html}</div>'
    )
    contents.append(f'<div id="test" class="tab-content">{test_html}</div>')
    if feature_html is not None:
        contents.append(f'<div id="feature" class="tab-content">{feature_html}</div>')
    if explainer_html:
        contents.append(
            f'<div id="explainer" class="tab-content">{explainer_html}</div>'
        )
    content_section = "\n".join(contents)

    # JS
    js = """
<script>
function showTab(id) {
  document.querySelectorAll('.tab-content').forEach(el=>el.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(el=>el.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  document.querySelector(`.tab[onclick*="${id}"]`).classList.add('active');
}
</script>
"""

    return tabs_section + "\n" + content_section + "\n" + js


def customize_figure_layout(fig, margin_dict=None):
    if margin_dict is None:
        margin_dict = {"l": 40, "r": 40, "t": 40, "b": 40}
    fig.update_layout(
        margin=margin_dict,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
    )
    fig.update_xaxes(gridcolor="#e8e8e8")
    fig.update_yaxes(gridcolor="#e8e8e8")
    return fig


def add_plot_to_html(fig, include_plotlyjs=True) -> str:
    custom_margin = {"l": 40, "r": 40, "t": 60, "b": 60}
    fig = customize_figure_layout(fig, margin_dict=custom_margin)
    return fig.to_html(
        full_html=False,
        default_height=350,
        include_plotlyjs="cdn" if include_plotlyjs else False,
    )


def add_hr_to_html() -> str:
    return "<hr>"


def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def predict_proba(self, X):
    pred = self.predict(X)
    return np.vstack((1 - pred, pred)).T
