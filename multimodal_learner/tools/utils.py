import base64
import logging
from typing import Optional

logging.basicConfig(level=logging.DEBUG)
LOG = logging.getLogger(__name__)


def get_html_template() -> str:
    """
    Returns the HTML header, opening <body> and container div,
    including all CSS needed for the report (tables, tabs, help button, etc.).
    """
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
      .container {
          max-width: 800px;
          margin: auto;
          background: white;
          padding: 20px;
          box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
          overflow-x: auto;
      }
      h1 { text-align: center; color: #333; }
      h2 {
          border-bottom: 2px solid #4CAF50;
          color: #4CAF50;
          padding-bottom: 5px;
      }
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
      .plot {
          text-align: center;
          margin: 20px 0;
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
          padding-right: 20px;
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
    """
    Closes container and body, and injects the JS for sortable tables.
    """
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
        const tbody = table.tBodies[0];
        Array.from(tbody.rows)
          .sort(comparer(idx, asc))
          .forEach(row => tbody.appendChild(row));
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
    feature_html: str,
    explainer_html: Optional[str] = None,
) -> str:
    """
    Renders the tab headers, tab contents, and the JS to switch tabs.
    """
    # Tabs header
    tabs = [
        '<div class="tabs">',
        '<div class="tab active" onclick="showTab(\'summary\')">Validation Summary & Config</div>',
        '<div class="tab" onclick="showTab(\'test\')">Test Summary</div>',
        '<div class="tab" onclick="showTab(\'feature\')">Feature Importance</div>',
    ]
    if explainer_html:
        tabs.append('<div class="tab" onclick="showTab(\'explainer\')">Explainer Plots</div>')
    tabs.append('<button id="openMetricsHelp" class="help-btn">Help</button>')
    tabs.append('</div>')
    tabs_section = "\n".join(tabs)

    # Tab contents
    contents = [
        f'<div id="summary" class="tab-content active">{summary_html}</div>',
        f'<div id="test" class="tab-content">{test_html}</div>',
        f'<div id="feature" class="tab-content">{feature_html}</div>',
    ]
    if explainer_html:
        contents.append(f'<div id="explainer" class="tab-content">{explainer_html}</div>')
    content_section = "\n".join(contents)

    # Tab-switching JS
    js = """
<script>
function showTab(id) {
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  document.querySelector(`.tab[onclick*="${id}"]`).classList.add('active');
}
</script>
"""
    return tabs_section + "\n" + content_section + "\n" + js


def encode_image_to_base64(image_path: str) -> str:
    """
    Reads an image file from disk and returns a base64-encoded string
    for embedding directly in HTML <img> tags.
    """
    try:
        with open(image_path, "rb") as img_f:
            return base64.b64encode(img_f.read()).decode("utf-8")
    except Exception as e:
        LOG.error(f"Failed to encode image '{image_path}': {e}")
        return ""
