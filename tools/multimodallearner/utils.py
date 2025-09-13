import base64
import logging
from typing import Optional

logging.basicConfig(level=logging.DEBUG)
LOG = logging.getLogger(__name__)


def get_html_template() -> str:
    """
    Returns the opening HTML, <head> (with CSS/JS), and opens <body> + .container.
    Includes:
      - Base styling for layout and tables
      - Sortable table headers with 3-state arrows (none ⇅, asc ↑, desc ↓)
      - A scroll helper class (.scroll-rows-30) that approximates ~30 visible rows
      - A guarded script so initializing runs only once even if injected twice
    """
    return """
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Galaxy-Ludwig Report</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      margin: 0;
      padding: 20px;
      background-color: #f4f4f4;
    }
    .container {
      max-width: 1200px;
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
      margin-top: 28px;
    }

    /* baseline table setup */
    table {
      border-collapse: collapse;
      margin: 20px 0;
      width: 100%;
      table-layout: fixed;
      background: #fff;
    }
    table, th, td {
      border: 1px solid #ddd;
    }
    th, td {
      padding: 10px;
      text-align: center;
      vertical-align: middle;
      word-break: break-word;
      white-space: normal;
      overflow-wrap: anywhere;
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
      border: 1px solid #ddd;
    }

    /* -------------------
       sortable columns (3-state: none ⇅, asc ↑, desc ↓)
       ------------------- */
    table.performance-summary th.sortable {
      cursor: pointer;
      position: relative;
      user-select: none;
    }
    /* default icon space */
    table.performance-summary th.sortable::after {
      content: '⇅';
      position: absolute;
      right: 12px;
      top: 50%;
      transform: translateY(-50%);
      font-size: 0.8em;
      color: #eaf5ea; /* light on green */
      text-shadow: 0 0 1px rgba(0,0,0,0.15);
    }
    /* three states override the default */
    table.performance-summary th.sortable.sorted-none::after { content: '⇅'; color: #eaf5ea; }
    table.performance-summary th.sortable.sorted-asc::after  { content: '↑';  color: #ffffff; }
    table.performance-summary th.sortable.sorted-desc::after { content: '↓';  color: #ffffff; }

    /* show ~30 rows with a scrollbar (tweak if you want) */
    .scroll-rows-30 {
      max-height: 900px;       /* ~30 rows depending on row height */
      overflow-y: auto;        /* vertical scrollbar (“sidebar”) */
      overflow-x: auto;
    }

    /* Tabs + Help button (used by build_tabbed_html) */
    .tabs {
      display: flex;
      align-items: center;
      border-bottom: 2px solid #ccc;
      margin-bottom: 1rem;
      gap: 6px;
      flex-wrap: wrap;
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
    .tab-content {
      display: none;
      padding: 20px;
      border: 1px solid #ccc;
      border-top: none;
      background: #fff;
    }
    .tab-content.active {
      display: block;
    }

    /* Modal (used by get_metrics_help_modal) */
    .modal {
      display: none;
      position: fixed;
      z-index: 9999;
      left: 0; top: 0;
      width: 100%; height: 100%;
      overflow: auto;
      background-color: rgba(0,0,0,0.4);
    }
    .modal-content {
      background-color: #fefefe;
      margin: 8% auto;
      padding: 20px;
      border: 1px solid #888;
      width: 90%;
      max-width: 900px;
      border-radius: 8px;
    }
    .modal .close {
      color: #777;
      float: right;
      font-size: 28px;
      font-weight: bold;
      line-height: 1;
      margin-left: 8px;
    }
    .modal .close:hover,
    .modal .close:focus {
      color: black;
      text-decoration: none;
      cursor: pointer;
    }
    .metrics-guide h3 { margin-top: 20px; }
    .metrics-guide p { margin: 6px 0; }
    .metrics-guide ul { margin: 10px 0; padding-left: 20px; }
  </style>

  <script>
    // Guard to avoid double-initialization if this block is included twice
    (function(){
      if (window.__perfSummarySortInit) return;
      window.__perfSummarySortInit = true;

      function initPerfSummarySorting() {
        // Record original order for "back to original"
        document.querySelectorAll('table.performance-summary tbody').forEach(tbody => {
          Array.from(tbody.rows).forEach((row, i) => { row.dataset.originalOrder = i; });
        });

        const getText = td => (td?.innerText || '').trim();
        const cmp = (idx, asc) => (a, b) => {
          const v1 = getText(a.children[idx]);
          const v2 = getText(b.children[idx]);
          const n1 = parseFloat(v1), n2 = parseFloat(v2);
          if (!isNaN(n1) && !isNaN(n2)) return asc ? n1 - n2 : n2 - n1; // numeric
          return asc ? v1.localeCompare(v2) : v2.localeCompare(v1);       // lexical
        };

        document.querySelectorAll('table.performance-summary th.sortable').forEach(th => {
          // initialize to “none”
          th.classList.remove('sorted-asc','sorted-desc');
          th.classList.add('sorted-none');

          th.addEventListener('click', () => {
            const table = th.closest('table');
            const headerRow = th.parentNode;
            const allTh = headerRow.querySelectorAll('th.sortable');
            const tbody = table.querySelector('tbody');

            // Determine current state BEFORE clearing
            const isAsc  = th.classList.contains('sorted-asc');
            const isDesc = th.classList.contains('sorted-desc');

            // Reset all headers in this row
            allTh.forEach(x => x.classList.remove('sorted-asc','sorted-desc','sorted-none'));

            // Compute next state
            let next;
            if (!isAsc && !isDesc) {
              next = 'asc';
            } else if (isAsc) {
              next = 'desc';
            } else {
              next = 'none';
            }
            th.classList.add('sorted-' + next);

            // Sort rows according to the chosen state
            const rows = Array.from(tbody.rows);
            if (next === 'none') {
              rows.sort((a, b) => (a.dataset.originalOrder - b.dataset.originalOrder));
            } else {
              const idx = Array.from(headerRow.children).indexOf(th);
              rows.sort(cmp(idx, next === 'asc'));
            }
            rows.forEach(r => tbody.appendChild(r));
          });
        });
      }

      // Run after DOM is ready
      if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initPerfSummarySorting);
      } else {
        initPerfSummarySorting();
      }
    })();
  </script>
</head>
<body>
  <div class="container">
"""


def get_html_closing():
    """Closes .container, body, and html."""
    return """
  </div>
</body>
</html>
"""

def build_tabbed_html(
    summary_html: str,
    train_html: str,
    test_html: str,
    feature_html: str,
    explainer_html: Optional[str] = None,
) -> str:
    """
    Renders the tab headers, contents, and JS to switch tabs.
    """
    tabs = [
        '<div class="tabs">',
        '<div class="tab active" onclick="showTab(\'summary\')">Model Summary & Config</div>',
        '<div class="tab" onclick="showTab(\'train\')">Train/Validation Summary</div>',
        '<div class="tab" onclick="showTab(\'test\')">Test Summary</div>',
        '<div class="tab" onclick="showTab(\'feature\')">Feature Importance</div>',
    ]
    if explainer_html:
        tabs.append('<div class="tab" onclick="showTab(\'explainer\')">Explainer Plots</div>')
    tabs.append('<button id="openMetricsHelp" class="help-btn">Help</button>')
    tabs.append('</div>')
    tabs_section = "\n".join(tabs)

    contents = [
        f'<div id="summary" class="tab-content active">{summary_html}</div>',
        f'<div id="train" class="tab-content">{train_html}</div>',
        f'<div id="test" class="tab-content">{test_html}</div>',
        f'<div id="feature" class="tab-content">{feature_html}</div>',
    ]
    if explainer_html:
        contents.append(f'<div id="explainer" class="tab-content">{explainer_html}</div>')
    content_section = "\n".join(contents)

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
