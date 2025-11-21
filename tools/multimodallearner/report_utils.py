import logging
import os
import sys
import html
import json
import platform
import shutil
import subprocess
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import tempfile

from utils import verify_outputs

logger = logging.getLogger(__name__)
def _escape(s: Any) -> str:
    return html.escape(str(s))


def _write_predictor_path(predictor):
    try:
        pred_path = getattr(predictor, "path", None)
        if pred_path:
            with open("predictor_path.txt", "w") as pf:
                pf.write(str(pred_path))
            logger.info("Wrote predictor path → predictor_path.txt")
        return pred_path
    except Exception:
        logger.warning("Could not write predictor_path.txt")
        return None


def _copy_config_if_available(pred_path: Optional[str], output_config: Optional[str]):
    if not output_config:
        return
    try:
        config_yaml_path = os.path.join(pred_path, "config.yaml") if pred_path else None
        if config_yaml_path and os.path.isfile(config_yaml_path):
            shutil.copy2(config_yaml_path, output_config)
            logger.info(f"Wrote AutoGluon config → {output_config}")
        else:
            with open(output_config, "w") as cfg_out:
                cfg_out.write("# config.yaml not found for this run\n")
            logger.warning(f"AutoGluon config.yaml not found; created placeholder at {output_config}")
    except Exception as e:
        logger.error(f"Failed to write config output '{output_config}': {e}")
        try:
            with open(output_config, "w") as cfg_out:
                cfg_out.write(f"# Failed to copy config.yaml: {e}\n")
        except Exception:
            pass


def write_outputs(
    args,
    predictor,
    problem_type: str,
    eval_results: dict,
    data_ctx: dict,
    raw_folds=None,
    ag_folds=None,
    raw_metrics_std=None,
    ag_by_split_std=None,
):
    from plot_logic import (
        build_summary_html,
        build_test_html_and_plots,
        build_feature_html,
        assemble_full_html_report,
        build_train_html_and_plots,
    )
    from feature_importance import build_feature_importance_html
    from autogluon.multimodal import MultiModalPredictor
    from metrics_logic import aggregate_metrics

    raw_metrics = eval_results.get("raw_metrics", {})
    ag_by_split = eval_results.get("ag_eval", {})
    fit_summary_obj = eval_results.get("fit_summary")

    df_train = data_ctx.get("train")
    df_val = data_ctx.get("val")
    df_test_internal = data_ctx.get("test_internal")
    df_test_external = data_ctx.get("test_external")
    df_test = df_test_external if df_test_external is not None else df_test_internal
    df_train_full = df_train if df_val is None else pd.concat([df_train, df_val], ignore_index=True)

    # Aggregate folds if provided without stds
    if raw_folds and raw_metrics_std is None:
        raw_metrics, raw_metrics_std = aggregate_metrics(raw_folds)
    if ag_folds and ag_by_split_std is None:
        ag_by_split, ag_by_split_std = aggregate_metrics(ag_folds)

    # Inject AG eval into raw metrics for visibility
    def _inject_ag(src: dict, dst: dict):
        for k, v in (src or {}).items():
            try:
                dst[f"AG_{k}"] = float(v)
            except Exception:
                dst[f"AG_{k}"] = v
    if "Train" in raw_metrics and "Train" in ag_by_split:
        _inject_ag(ag_by_split["Train"], raw_metrics["Train"])
    if "Validation" in raw_metrics and "Validation" in ag_by_split:
        _inject_ag(ag_by_split["Validation"], raw_metrics["Validation"])
    if "Test" in raw_metrics and "Test" in ag_by_split:
        _inject_ag(ag_by_split["Test"], raw_metrics["Test"])

    # CSV
    all_keys: List[str] = []
    for split in ("Train", "Validation", "Test", "Test (external)"):
        if split in raw_metrics:
            for k in raw_metrics[split].keys():
                if k not in all_keys:
                    all_keys.append(k)
    rows = []
    if "Train" in raw_metrics:
        rows.append({"phase": "train", **{k: raw_metrics["Train"].get(k, np.nan) for k in all_keys}})
    if "Validation" in raw_metrics:
        rows.append({"phase": "validation", **{k: raw_metrics["Validation"].get(k, np.nan) for k in all_keys}})
    if "Test" in raw_metrics:
        rows.append({"phase": "test", **{k: raw_metrics["Test"].get(k, np.nan) for k in all_keys}})
    if "Test (external)" in raw_metrics:
        rows.append({"phase": "test_external", **{k: raw_metrics["Test (external)"].get(k, np.nan) for k in all_keys}})
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)
    logger.info(f"Wrote metrics CSV → {args.output_csv}")

    # JSON
    with open(args.output_json, "w") as f:
        json.dump(
            {
                "train": raw_metrics.get("Train", {}),
                "val": raw_metrics.get("Validation", {}),
                "test": raw_metrics.get("Test", {}),
                "test_external": raw_metrics.get("Test (external)", {}),
                "ag_eval": ag_by_split,
                "ag_eval_std": ag_by_split_std,
                "fit_summary": fit_summary_obj,
                "problem_type": problem_type,
                "predictor_path": getattr(predictor, "path", None),
                "threshold": args.threshold,
                "threshold_test": args.threshold,
                "preset": args.preset,
                "eval_metric": args.eval_metric,
                "folds": {
                    "raw_folds": raw_folds,
                    "ag_folds": ag_folds,
                    "summary_mean": raw_metrics if raw_folds else None,
                    "summary_std": raw_metrics_std,
                    "ag_summary_mean": ag_by_split,
                    "ag_summary_std": ag_by_split_std,
                },
            },
            f,
            indent=2,
            default=str,
        )
    logger.info(f"Wrote full JSON → {args.output_json}")

    # HTML report assembly
    label_col = args.target_column
    image_cols = []
    include_text = False

    class_balance_block_html = build_class_balance_html(df_train_full, label_col)
    summary_perf_table_html = build_model_performance_summary_table(
        train_scores=raw_metrics.get("Train", {}),
        val_scores=raw_metrics.get("Validation", {}),
        test_scores=raw_metrics.get("Test", {}),
        include_test=True,
        title=None,
        show_title=False,
    )

    try:
        feature_importance_html = build_feature_importance_html(predictor, df_train_full, label_col)
    except Exception as e:
        feature_importance_html = f"<p>Feature importance unavailable: {e}</p>"

    extra_run_rows = [
        ("Target column", label_col),
        ("Model evaluation metric", args.eval_metric or "AutoGluon default"),
        ("Experiment quality", args.preset or "AutoGluon default"),
        ("Tabular backbone", getattr(args, "backbone_tabular", None) or "—"),
        ("Image backbone", args.backbone_image or "—"),
        ("Text backbone", args.backbone_text or "—"),
        ("Fusion backbone", getattr(args, "backbone_fusion", None) or "—"),
    ]

    summary_html = build_summary_html(
        predictor=predictor,
        df_train=df_train_full,
        df_val=df_val,
        df_test=df_test,
        label_column=label_col,
        extra_run_rows=extra_run_rows,
        class_balance_html=class_balance_block_html,
        perf_table_html=summary_perf_table_html,
        feature_html=feature_importance_html,
    )

    train_tab_perf_html = build_model_performance_summary_table(
        train_scores=raw_metrics.get("Train", {}),
        val_scores=raw_metrics.get("Validation", {}),
        test_scores=raw_metrics.get("Test", {}),
        include_test=False,
        title=None,
        show_title=False,
    )

    train_html = build_train_html_and_plots(
        predictor=predictor,
        problem_type=problem_type,
        df_train=df_train,
        label_column=label_col,
        tmpdir=tempfile.mkdtemp(),
        seed=int(args.random_seed),
        perf_table_html=train_tab_perf_html,
        threshold=None,
    )

    test_html_template, plots = build_test_html_and_plots(
        predictor,
        problem_type,
        df_test,
        label_col,
        tempfile.mkdtemp(),
        threshold=args.threshold,
    )

    def _fmt_val(v):
        if isinstance(v, (int, np.integer)):
            return f"{int(v)}"
        if isinstance(v, (float, np.floating)):
            return f"{v:.6f}"
        return str(v)

    test_scores = raw_metrics.get("Test", {})
    metric_rows = "".join(
        f"<tr><td>{k.replace('_',' ').replace('(TNR)','(TNR)').replace('(Sensitivity/TPR)', '(Sensitivity/TPR)')}</td>"
        f"<td>{_fmt_val(v)}</td></tr>"
        for k, v in test_scores.items()
    )
    test_html_filled = test_html_template.format(metric_rows)

    is_multimodal = isinstance(predictor, MultiModalPredictor)
    leaderboard_html = "" if is_multimodal else build_leaderboard_html(predictor)
    inputs_html = ""
    ignored_features_html = "" if is_multimodal else build_ignored_features_html(predictor, df_train_full)
    presets_hparams_html = build_presets_hparams_html(predictor)
    notices: List[str] = []
    if args.threshold is not None and problem_type == "binary":
        notices.append(f"Using decision threshold = {float(args.threshold):.3f} on Test.")
    warnings_html = build_warnings_html([], notices)
    repro_html = build_reproducibility_html(args, {}, getattr(predictor, "path", None))

    transparency_blocks = "\n".join(
        [
            leaderboard_html,
            inputs_html,
            ignored_features_html,
            presets_hparams_html,
            warnings_html,
            repro_html,
        ]
    )

    try:
        feature_text = build_feature_html(predictor, df_test, label_col, tempfile.mkdtemp(), args.random_seed) if df_test is not None else ""
    except Exception:
        feature_text = "<p>Feature analysis unavailable for this model.</p>"

    full_html = assemble_full_html_report(
        summary_html,
        train_html,
        test_html_filled,
        plots,
        feature_text + transparency_blocks,
    )
    with open(args.output_html, "w") as f:
        f.write(full_html)
    logger.info(f"Wrote HTML report → {args.output_html}")

    pred_path = _write_predictor_path(predictor)
    _copy_config_if_available(pred_path, args.output_config)

    outputs_to_check = [
        (args.output_csv, "CSV metrics"),
        (args.output_json, "JSON results"),
        (args.output_html, "HTML report"),
    ]
    if args.output_config:
        outputs_to_check.append((args.output_config, "AutoGluon config"))
    verify_outputs(outputs_to_check)


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


def get_model_architecture(predictor: Any) -> str:
    """
    Returns a human-friendly description of the final model architecture based on the
    MultiModalPredictor configuration (e.g., timm_image=resnet50, hf_text=bert-base-uncased).
    """
    # MultiModalPredictor path: read backbones from config if available
    archs = []
    for attr in ("_config", "config"):
        cfg = getattr(predictor, attr, None)
        try:
            model_cfg = getattr(cfg, "model", None)
            if model_cfg:
                # OmegaConf-like mapping
                for name, sub in dict(model_cfg).items():
                    ck = None
                    # sub may be an object or a dict-like node
                    for k in ("checkpoint_name", "name", "model_name"):
                        try:
                            ck = getattr(sub, k)
                        except Exception:
                            ck = sub.get(k) if isinstance(sub, dict) else ck
                        if ck:
                            break
                    if ck:
                        archs.append(f"{name}={ck}")
        except Exception:
            continue

    if archs:
        return ", ".join(archs)

    # Fallback
    return type(predictor).__name__

def collect_run_context(args, predictor, problem_type: str,
                        df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame,
                        warnings_list: List[str],
                        notes_list: List[str]) -> Dict[str, Any]:
    """Build a dictionary with run/system context for transparency."""
    # System info (best-effort; not depending on AutoGluon stdout)
    try:
        import psutil  # optional
        mem = psutil.virtual_memory()
        mem_total_gb = mem.total / (1024 ** 3)
        mem_avail_gb = mem.available / (1024 ** 3)
    except Exception:
        mem_total_gb = mem_avail_gb = None

    ctx = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "cpu_count": os.cpu_count(),
        "memory_total_gb": mem_total_gb,
        "memory_available_gb": mem_avail_gb,
        "packages": {},
        "problem_type": problem_type,
        "label_column": args.label_column,
        "time_limit_sec": args.time_limit,
        "random_seed": args.random_seed,
        "splits": {
            "train_rows": int(len(df_train)),
            "val_rows": int(len(df_val)),
            "test_rows": int(len(df_test)),
            "n_features_raw": int(len(df_train.columns) - 1),  # minus label
        },
        "warnings": warnings_list,
        "notes": notes_list,
    }
    # Package versions (safe best-effort)
    try:
        import autogluon
        ctx["packages"]["autogluon"] = getattr(autogluon, "__version__", "unknown")
    except Exception:
        pass
    try:
        import torch as _torch
        ctx["packages"]["torch"] = getattr(_torch, "__version__", "unknown")
    except Exception:
        pass
    try:
        import sklearn
        ctx["packages"]["scikit_learn"] = getattr(sklearn, "__version__", "unknown")
    except Exception:
        pass
    try:
        import numpy as _np
        ctx["packages"]["numpy"] = getattr(_np, "__version__", "unknown")
    except Exception:
        pass
    try:
        import pandas as _pd
        ctx["packages"]["pandas"] = getattr(_pd, "__version__", "unknown")
    except Exception:
        pass
    return ctx

def build_class_balance_html(df: pd.DataFrame, label_col: str) -> str:
    if df[label_col].dtype.kind in "ifu":  # numeric
        uniq = pd.Series(df[label_col]).value_counts(dropna=False).sort_index()
    else:
        uniq = pd.Series(df[label_col].astype(str)).value_counts(dropna=False)
    total = int(uniq.sum())
    rows = []
    for k, v in uniq.items():
        p = (100.0 * v / max(total, 1))
        rows.append(f"<tr><td>{_escape(k)}</td><td>{v}</td><td>{p:.2f}%</td></tr>")
    return f"""
    <h3>Class Balance (Train Full)</h3>
    <table class="table">
      <thead><tr><th>Class</th><th>Count</th><th>Percent</th></tr></thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
    """

def build_leaderboard_html(predictor) -> str:
    try:
        lb = predictor.leaderboard(silent=True)
        # keep common helpful columns if present
        cols_pref = ["model", "score_val", "eval_metric", "pred_time_val", "fit_time",
                     "pred_time_val_marginal", "fit_time_marginal", "stack_level", "can_infer", "fit_order"]
        cols = [c for c in cols_pref if c in lb.columns] or list(lb.columns)
        return "<h3>Model Leaderboard (Validation)</h3>" + lb[cols].to_html(index=False)
    except Exception as e:
        return f"<h3>Model Leaderboard</h3><p>Unavailable: {_escape(e)}</p>"

def build_ignored_features_html(predictor, df_any: pd.DataFrame) -> str:
    # MultiModalPredictor does not always expose .features(); guard accordingly.
    used = set()
    try:
        used = set(predictor.features())
    except Exception:
        # If we can't determine, don't emit a misleading section
        return ""
    raw_cols = [c for c in df_any.columns if c != getattr(predictor, "label", None)]
    ignored = [c for c in raw_cols if c not in used]
    if not ignored:
        return ""
    items = "".join(f"<li>{html.escape(c)}</li>" for c in ignored)
    return f"""
    <h3>Ignored / Unused Features</h3>
    <p>The following columns were not used by the trained predictor at inference time:</p>
    <ul>{items}</ul>
    """

def build_presets_hparams_html(predictor) -> str:
    # MultiModalPredictor path
    mm_hp = {}
    for attr in ("_config", "config", "_fit_args"):
        if hasattr(predictor, attr):
            try:
                val = getattr(predictor, attr)
                # make it JSON-ish
                mm_hp[attr] = str(val)
            except Exception:
                continue
    hp_html = f"<pre>{html.escape(json.dumps(mm_hp, indent=2))}</pre>" if mm_hp else "<i>Unavailable</i>"
    return f"<h3>Training Presets & Hyperparameters</h3><details open><summary>Show hyperparameters</summary>{hp_html}</details>"

def build_warnings_html(warnings_list: List[str], notes_list: List[str]) -> str:
    if not warnings_list and not notes_list:
        return ""
    w_html = "".join(f"<li>{_escape(w)}</li>" for w in warnings_list)
    n_html = "".join(f"<li>{_escape(n)}</li>" for n in notes_list)
    return f"""
    <h3>Warnings & Notices</h3>
    {'<h4>Warnings</h4><ul>'+w_html+'</ul>' if warnings_list else ''}
    {'<h4>Notices</h4><ul>'+n_html+'</ul>' if notes_list else ''}
    """

def build_reproducibility_html(args, ctx: Dict[str, Any], model_path: Optional[str]) -> str:
    cmd = " ".join(_escape(x) for x in sys.argv)
    load_snippet = ""
    if model_path:
        load_snippet = f"""<pre>
from autogluon.multimodal import MultiModalPredictor
predictor = MultiModalPredictor.load("{_escape(model_path)}")
</pre>"""
    pkg_rows = "".join(f"<tr><td>{_escape(k)}</td><td>{_escape(v)}</td></tr>" for k, v in (ctx.get("packages") or {}).items())
    sys_table = f"""
    <table class="table">
      <tbody>
        <tr><th>Timestamp</th><td>{_escape(ctx.get('timestamp'))}</td></tr>
        <tr><th>Python</th><td>{_escape(ctx.get('python_version'))}</td></tr>
        <tr><th>Platform</th><td>{_escape(ctx.get('platform'))}</td></tr>
        <tr><th>CPU Count</th><td>{_escape(ctx.get('cpu_count'))}</td></tr>
        <tr><th>Memory (GB)</th><td>Total: {_escape(ctx.get('memory_total_gb'))} | Avail: {_escape(ctx.get('memory_available_gb'))}</td></tr>
        <tr><th>Seed</th><td>{_escape(ctx.get('random_seed'))}</td></tr>
        <tr><th>Time Limit (s)</th><td>{_escape(ctx.get('time_limit_sec'))}</td></tr>
      </tbody>
    </table>
    """
    pkgs_table = f"""
    <h4>Package Versions</h4>
    <table class="table">
      <thead><tr><th>Package</th><th>Version</th></tr></thead>
      <tbody>{pkg_rows}</tbody>
    </table>
    """
    return f"""
    <h3>Reproducibility</h3>
    <h4>Command</h4>
    <pre>{cmd}</pre>
    {sys_table}
    {pkgs_table}
    <h4>Load Trained Model</h4>
    {load_snippet or '<i>Model path not available</i>'}
    """

def build_modalities_html(predictor, df_any: pd.DataFrame, label_col: str, image_col: Optional[str]) -> str:
    """Summarize which inputs/modalities are used for MultiModalPredictor."""
    cols = [c for c in df_any.columns]
    # exclude label from feature list
    feat_cols = [c for c in cols if c != label_col]
    # identify image vs tabular columns from args / presence
    img_present = (image_col in df_any.columns) if image_col else False
    tab_cols = [c for c in feat_cols if c != image_col]

    # brief lists (avoid dumping all, unless small)
    def list_or_count(arr, max_show=20):
        if len(arr) <= max_show:
            items = "".join(f"<li>{html.escape(str(x))}</li>" for x in arr)
            return f"<ul>{items}</ul>"
        return f"<p>{len(arr)} columns</p>"

    img_block = f"<p><b>Image column:</b> {html.escape(image_col)}</p>" if img_present else "<p><b>Image column:</b> None</p>"
    tab_block = f"<div><b>Structured columns:</b> {len(tab_cols)}{list_or_count(tab_cols, max_show=15)}</div>"

    return f"""
    <h3>Modalities & Inputs</h3>
    <p>This run used <b>MultiModalPredictor</b> (images + structured features).</p>
    <p><b>Label column:</b> {html.escape(label_col)}</p>
    {img_block}
    {tab_block}
    """
def build_model_performance_summary_table(
    train_scores: dict,
    val_scores: dict,
    test_scores: dict | None = None,
    include_test: bool = True,
    title: str | None = 'Model Performance Summary',
    show_title: bool = True,
) -> str:
    """
    Returns an HTML table for metrics, optionally hiding the Test column.
    Keys across score dicts are unioned; missing values render as '—'.
    """
    def fmt(v):
        if v is None:
            return '—'
        if isinstance(v, (int, float)):
            return f'{v:.4f}'
        return str(v)

    # Collect union of metric keys across splits
    metrics = set(train_scores.keys()) | set(val_scores.keys()) | (set(test_scores.keys()) if (include_test and test_scores) else set())

    # Remove AG_roc_auc entirely as requested
    metrics.discard('AG_roc_auc')

    # Helper: normalize metric keys for matching preferred names
    def _norm(k: str) -> str:
        return ''.join(ch for ch in str(k).lower() if ch.isalnum())

    # Preferred metrics to appear at the end in this specific order (display names):
    preferred_display = ['Accuracy', 'ROC-AUC', 'Precision', 'Recall', 'F1-Score', 'PR-AUC', 'Specificity', 'MCC', 'LogLoss']
    # Mapping of normalized key -> display label
    norm_to_display = {
        'accuracy': 'Accuracy',
        'acc': 'Accuracy',
        'rocauc': 'ROC-AUC',
        'roc_auc': 'ROC-AUC',
        'rocaucscore': 'ROC-AUC',
        'precision': 'Precision',
        'prec': 'Precision',
        'recall': 'Recall',
        'recallsensitivitytpr': 'Recall',
        'f1': 'F1-Score',
        'f1score': 'F1-Score',
        'pr_auc': 'PR-AUC',
        'prauc': 'PR-AUC',
        'averageprecision': 'PR-AUC',
        'specificity': 'Specificity',
        'tnr': 'Specificity',
        'mcc': 'MCC',
        'logloss': 'LogLoss',
        'crossentropy': 'LogLoss',
    }

    # Build ordered list: all non-preferred metrics sorted alphabetically, then preferred metrics in the requested order if present
    preferred_norms = [ _norm(x) for x in preferred_display ]
    all_metrics = list(metrics)
    # Partition
    preferred_present = []
    others = []
    for m in sorted(all_metrics):
        nm = _norm(m)
        if nm in preferred_norms or any(p in nm for p in ['rocauc','prauc','f1','mcc','logloss','accuracy','precision','recall','specificity']):
            # Defer preferred-like metrics to the end (we will place them in canonical order)
            preferred_present.append(m)
        else:
            others.append(m)

    # Now assemble final metric order: others (alpha), then preferred in exact requested order if they exist in metrics
    final_metrics = []
    final_metrics.extend(others)
    for disp in preferred_display:
        # find any original key matching this display (by normalized mapping)
        target_norm = _norm(disp)
        found = None
        for m in preferred_present:
            if _norm(m) == target_norm or norm_to_display.get(_norm(m)) == disp or _norm(m).replace(' ', '') == target_norm:
                found = m
                break
            # also allow substring matches (e.g., 'roc_auc' vs 'rocauc')
            if target_norm in _norm(m):
                found = m
                break
        if found:
            final_metrics.append(found)

    metrics = final_metrics

    # Make all headers sortable by adding the 'sortable' class; the JS in utils.py hooks table.performance-summary
    header_cells = [
        '<th class="sortable">Metric</th>',
        '<th class="sortable">Train</th>',
        '<th class="sortable">Validation</th>'
    ]
    if include_test and test_scores:
        header_cells.append('<th class="sortable">Test</th>')

    rows_html = []
    for m in metrics:
        # Display label mapping: clean up common verbose names
        disp = m
        nm = _norm(m)
        if nm in norm_to_display:
            disp = norm_to_display[nm]
        else:
            # generic cleanup: replace underscores with space and remove parenthetical qualifiers
            disp = str(m).replace('_', ' ')
            disp = disp.replace('(Sensitivity/TPR)', '')
            disp = disp.replace('(TNR)', '')
            disp = disp.strip()

        cells = [
            f'<td>{_escape(disp)}</td>',
            f'<td>{fmt(train_scores.get(m))}</td>',
            f'<td>{fmt(val_scores.get(m))}</td>',
        ]
        if include_test and test_scores:
            cells.append(f'<td>{fmt(test_scores.get(m))}</td>')

        rows_html.append('<tr>' + ''.join(cells) + '</tr>')

    title_html = f'<h3 style="margin-top:0">{title}</h3>' if (show_title and title) else ''

    table_html = f"""
      {title_html}
      <table class="performance-summary">
        <thead><tr>{''.join(header_cells)}</tr></thead>
        <tbody>{''.join(rows_html)}</tbody>
      </table>
    """
    return table_html
