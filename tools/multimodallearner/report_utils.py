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

def _escape(s: Any) -> str:
    return html.escape(str(s))

def get_model_architecture(predictor: Any) -> str:
    """
    Returns a human-friendly description of the final model architecture.
    - TabularPredictor: best model name
    - MultiModalPredictor: backbone(s) from the config (e.g., timm_image=resnet50, hf_text=bert-base-uncased)
    """
    # TabularPredictor path
    try:
        from autogluon.tabular import TabularPredictor as _TP
        if isinstance(predictor, _TP):
            try:
                return str(predictor.get_model_best())
            except Exception:
                try:
                    lb = predictor.leaderboard(silent=True)
                    if "model" in lb.columns and len(lb):
                        return str(lb.iloc[0]["model"])
                except Exception:
                    pass
            return "AutoGluon Tabular (ensemble)"
    except Exception:
        pass

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
    # TabularPredictor exposes .features(); MultiModalPredictor may not
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
    # TabularPredictor: prefer fit_summary(); MultiModal: try ._config or ._fit_args
    try:
        from autogluon.tabular import TabularPredictor as _TP
        if isinstance(predictor, _TP):
            summ = predictor.fit_summary(verbosity=0)
            hp = summ.get("hyperparameters") or summ.get("model_hyperparams") or {}
            hp_html = f"<pre>{html.escape(json.dumps(hp, indent=2))}</pre>" if hp else "<i>Unavailable</i>"
            return f"<h3>Training Presets & Hyperparameters</h3><details open><summary>Show hyperparameters</summary>{hp_html}</details>"
    except Exception:
        pass
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
from autogluon.tabular import TabularPredictor
predictor = TabularPredictor.load("{_escape(model_path)}")
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
    tab_block = f"<div><b>Tabular columns:</b> {len(tab_cols)}{list_or_count(tab_cols, max_show=15)}</div>"

    return f"""
    <h3>Modalities & Inputs</h3>
    <p>This run used <b>MultiModalPredictor</b> (images + tabular).</p>
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
