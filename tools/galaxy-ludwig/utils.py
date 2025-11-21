import base64
import json


def get_html_template():
    return """
    <html>
    <head>
        <title>Galaxy-Ludwig Report</title>
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
          h1 {
              text-align: center;
              color: #333;
          }
          h2 {
              border-bottom: 2px solid #4CAF50;
              color: #4CAF50;
              padding-bottom: 5px;
          }
          table {
              border-collapse: collapse;
              margin: 20px 0;
              width: 100%;
              table-layout: fixed; /* Enforces consistent column widths */
          }
          table, th, td {
              border: 1px solid #ddd;
          }
          th, td {
              padding: 8px;
              text-align: center; /* Center-align text */
              vertical-align: middle; /* Center-align content vertically */
              word-wrap: break-word; /* Break long words to avoid overflow */
          }
          th:first-child, td:first-child {
              width: 5%; /* Smaller width for the first column */
          }
          th:nth-child(2), td:nth-child(2) {
              width: 50%; /* Wider for the metric/description column */
          }
          th:last-child, td:last-child {
              width: 25%; /* Value column gets remaining space */
          }
          th {
              background-color: #4CAF50;
              color: white;
          }
          /* feature importance layout tweaks */
          table.feature-importance-table {
              table-layout: auto;
          }
          table.feature-importance-table th,
          table.feature-importance-table td {
              white-space: nowrap;
              word-break: normal;
          }
          /* sortable tables */
          .sortable-table th.sortable {
              cursor: pointer;
              position: relative;
              user-select: none;
          }
          .sortable-table th.sortable::after {
              content: '⇅';
              position: absolute;
              right: 12px;
              top: 50%;
              transform: translateY(-50%);
              font-size: 0.8em;
              color: #eaf5ea;
              text-shadow: 0 0 1px rgba(0,0,0,0.15);
          }
          .sortable-table th.sortable.sorted-none::after { content: '⇅'; color: #eaf5ea; }
          .sortable-table th.sortable.sorted-asc::after  { content: '↑';  color: #ffffff; }
          .sortable-table th.sortable.sorted-desc::after { content: '↓';  color: #ffffff; }
          .scroll-rows-30 {
              max-height: 900px;
              overflow-y: auto;
              overflow-x: auto;
          }
          .plot {
              text-align: center;
              margin: 20px 0;
          }
          .plot img {
              max-width: 100%;
              height: auto;
          }
        </style>
        <script>
          (function() {
            if (window.__sortableInit) return;
            window.__sortableInit = true;

            function initSortableTables() {
              document.querySelectorAll('table.sortable-table tbody').forEach(tbody => {
                Array.from(tbody.rows).forEach((row, i) => { row.dataset.originalOrder = i; });
              });

              const getText = td => (td?.innerText || '').trim();
              const cmp = (idx, asc) => (a, b) => {
                const v1 = getText(a.children[idx]);
                const v2 = getText(b.children[idx]);
                const n1 = parseFloat(v1), n2 = parseFloat(v2);
                if (!isNaN(n1) && !isNaN(n2)) return asc ? n1 - n2 : n2 - n1;
                return asc ? v1.localeCompare(v2) : v2.localeCompare(v1);
              };

              document.querySelectorAll('table.sortable-table th.sortable').forEach(th => {
                th.classList.remove('sorted-asc','sorted-desc');
                th.classList.add('sorted-none');

                th.addEventListener('click', () => {
                  const table = th.closest('table');
                  const headerRow = th.parentNode;
                  const allTh = headerRow.querySelectorAll('th.sortable');
                  const tbody = table.querySelector('tbody');

                  const isAsc  = th.classList.contains('sorted-asc');
                  const isDesc = th.classList.contains('sorted-desc');

                  allTh.forEach(x => x.classList.remove('sorted-asc','sorted-desc','sorted-none'));

                  let next;
                  if (!isAsc && !isDesc) {
                    next = 'asc';
                  } else if (isAsc) {
                    next = 'desc';
                  } else {
                    next = 'none';
                  }
                  th.classList.add('sorted-' + next);

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

            if (document.readyState === 'loading') {
              document.addEventListener('DOMContentLoaded', initSortableTables);
            } else {
              initSortableTables();
            }
          })();
        </script>
    </head>
    <body>
    <div class="container">
    """


def get_html_closing():
    return """
    </div>
    </body>
    </html>
    """


def encode_image_to_base64(image_path):
    """Convert an image file to a base64 encoded string."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def json_to_nested_html_table(json_data, depth=0):
    """
    Convert JSON object to an HTML nested table.

    Parameters:
        json_data (dict or list): The JSON data to convert.
        depth (int): Current depth level for indentation.

    Returns:
        str: HTML string for the nested table.
    """
    # Base case: if JSON is a simple key-value pair dictionary
    if isinstance(json_data, dict) and all(
        not isinstance(v, (dict, list)) for v in json_data.values()
    ):
        # Render a flat table
        rows = [
            f"<tr><th>{key}</th><td>{value}</td></tr>"
            for key, value in json_data.items()
        ]
        return f"<table>{''.join(rows)}</table>"

    # Base case: if JSON is a list of simple values
    if isinstance(json_data, list) and all(
        not isinstance(v, (dict, list)) for v in json_data
    ):
        rows = [
            f"<tr><th>Index {i}</th><td>{value}</td></tr>"
            for i, value in enumerate(json_data)
        ]
        return f"<table>{''.join(rows)}</table>"

    # Recursive case: if JSON contains nested structures
    if isinstance(json_data, dict):
        rows = [
            f"<tr><th style='padding-left:{depth * 20}px;'>{key}</th>"
            f"<td>{json_to_nested_html_table(value, depth + 1)}</td></tr>"
            for key, value in json_data.items()
        ]
        return f"<table>{''.join(rows)}</table>"

    if isinstance(json_data, list):
        rows = [
            f"<tr><th style='padding-left:{depth * 20}px;'>[{i}]</th>"
            f"<td>{json_to_nested_html_table(value, depth + 1)}</td></tr>"
            for i, value in enumerate(json_data)
        ]
        return f"<table>{''.join(rows)}</table>"

    # Base case: simple value
    return f"{json_data}"


def json_to_html_table(json_data):
    """
    Convert JSON to a vertically oriented HTML table.

    Parameters:
        json_data (str or dict): JSON string or dictionary.

    Returns:
        str: HTML table representation.
    """
    if isinstance(json_data, str):
        json_data = json.loads(json_data)
    return json_to_nested_html_table(json_data)
