const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  WidthType, BorderStyle, AlignmentType, VerticalAlign,
} = require("docx");

const FONT = "Arial";
const BODY = 16;      // 8 pt (half-points)
const CAPTION = 18;   // 9 pt
const NOTE = 14;      // 7 pt

const NO = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const RULE_HEAVY = { style: BorderStyle.SINGLE, size: 12, color: "000000" }; // 1.5 pt
const RULE_LIGHT = { style: BorderStyle.SINGLE, size: 6, color: "000000" };  // 0.75 pt

function runs(text, opts = {}) {
  // supports simple inline markers: **bold**, ^sup^, and italics via _x_
  const parts = String(text).split(/(\*\*[^*]+\*\*|\^[^^]+\^|_[^_]+_)/g).filter(Boolean);
  return parts.map((p) => {
    if (p.startsWith("**") && p.endsWith("**"))
      return new TextRun({ text: p.slice(2, -2), bold: true, font: FONT, size: opts.size || BODY });
    if (p.startsWith("^") && p.endsWith("^"))
      return new TextRun({ text: p.slice(1, -1), superScript: true, font: FONT, size: opts.size || BODY });
    if (p.startsWith("_") && p.endsWith("_"))
      return new TextRun({ text: p.slice(1, -1), italics: true, font: FONT, size: opts.size || BODY });
    return new TextRun({ text: p, font: FONT, size: opts.size || BODY, bold: !!opts.bold });
  });
}

function para(text, opts = {}) {
  return new Paragraph({
    children: runs(text, opts),
    alignment: opts.alignment || AlignmentType.LEFT,
    spacing: { before: opts.before ?? 0, after: opts.after ?? 0, line: opts.line ?? 220, lineRule: "auto" },
  });
}

function cell(text, width, opts = {}) {
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    borders: {
      top: opts.top || NO,
      bottom: opts.bottom || NO,
      left: NO,
      right: NO,
    },
    verticalAlign: VerticalAlign.TOP,
    children: [para(text, { bold: opts.bold, size: opts.size })],
  });
}

/**
 * Build a Bioinformatics-style table: three horizontal rules, no vertical rules,
 * caption above, lower-case letter footnotes below.
 */
function buildTable({ number, title, headers, rows, widths, notes }) {
  const total = widths.reduce((a, b) => a + b, 0);

  const headerRow = new TableRow({
    tableHeader: true,
    children: headers.map((h, i) =>
      cell(h, widths[i], { bold: true, top: RULE_HEAVY, bottom: RULE_LIGHT })
    ),
  });

  const bodyRows = rows.map((r, ri) =>
    new TableRow({
      children: r.map((c, i) =>
        cell(c, widths[i], { bottom: ri === rows.length - 1 ? RULE_HEAVY : NO })
      ),
    })
  );

  const table = new Table({
    columnWidths: widths,
    width: { size: total, type: WidthType.DXA },
    margins: { top: 60, bottom: 60, left: 0, right: 110 },
    rows: [headerRow, ...bodyRows],
  });

  const children = [
    new Paragraph({
      children: [
        new TextRun({ text: `Table ${number}. `, bold: true, font: FONT, size: CAPTION }),
        ...runs(title, { size: CAPTION }),
      ],
      spacing: { after: 120, line: 240, lineRule: "auto" },
    }),
    table,
  ];

  (notes || []).forEach((n, i) =>
    children.push(
      new Paragraph({
        children: runs(n, { size: NOTE }),
        spacing: { before: i === 0 ? 100 : 20, after: 20, line: 200, lineRule: "auto" },
      })
    )
  );

  return children;
}

function makeDoc(children) {
  return new Document({
    styles: { default: { document: { run: { font: FONT, size: BODY } } } },
    sections: [
      {
        properties: {
          page: {
            // US Letter, 18.9 mm side margins -> 178 mm (double-column) text width
            size: { width: 12240, height: 15840 },
            margin: { top: 1134, right: 1075, bottom: 1134, left: 1075 },
          },
        },
        children,
      },
    ],
  });
}

async function write(name, children) {
  const doc = makeDoc(children);
  const buf = await Packer.toBuffer(doc);
  fs.writeFileSync(name, buf);
  console.log("wrote", name);
}

/* ------------------------------------------------------------------ */
/* Table 1 — module overview                                           */
/* ------------------------------------------------------------------ */
const W4 = [1700, 2797, 2797, 2797];

const table1 = buildTable({
  number: 1,
  title: "Overview of the three GLEAM learner modules.",
  headers: ["Aspect", "Tabular Learner", "Image Learner", "Multimodal Learner"],
  widths: W4,
  rows: [
    ["Module version", "0.1.5", "0.1.6", "0.1.9"],
    ["Backend framework", "PyCaret 3.3.2", "Ludwig 0.10.1", "AutoGluon MultiModalPredictor 1.4.0"],
    ["Container image", "quay.io/goeckslab/galaxy-pycaret:3.3.2", "quay.io/goeckslab/galaxy-ludwig-gpu:0.10.1", "quay.io/goeckslab/multimodal-learner:1.4.0"],
    ["Training input", "CSV or TSV table containing the target column", "Image ZIP archive with a metadata CSV giving image paths and labels", "CSV or TSV table with optional image ZIP archives; tabular, text and image columns"],
    ["Supported tasks", "Classification; regression", "Binary and multi-class classification; regression", "Classification; regression (task inferred from the target)"],
    ["Model-selection strategy", "Trains a set of candidate estimators, ranks them by a user-selected metric and optionally tunes the best model", "Trains one user-selected backbone per run, pretrained or from scratch, with optional encoder fine-tuning", "Preset-driven AutoGluon ensemble built over user-selected text and image backbones"],
    ["Selectable models or backbones^a^", "18 classification and 25 regression estimators", "129 image backbones (74 TorchVision, 55 MetaFormer)", "8 text and 76 image backbones; 3 quality presets"],
    ["Representative model families", "Regularized linear models, discriminant analysis, naive Bayes, k-nearest neighbours, support vector machines, multilayer perceptrons, tree ensembles and gradient boosting (XGBoost, LightGBM, CatBoost)", "ResNet/ResNeXt, EfficientNet, RegNet, VGG, MobileNet, ShuffleNet, SqueezeNet, ViT, Swin, ConvNeXt, MaxViT and MetaFormer variants (IdentityFormer, RandFormer, PoolFormerV2, ConvFormer, CAFormer)", "Text: DeBERTa-v3, ELECTRA, RoBERTa, BERT, ALBERT. Image: Swin, ViT, ConvNeXt, CAFormer, EVA-02, ResNet"],
    ["Primary outputs", "Self-contained HTML report; serialized best model (HDF5); best-model parameter table (CSV)", "Self-contained HTML report; Ludwig model directory; predictions and run artefacts (ZIP)", "Self-contained HTML report; metrics (JSON); training configuration (YAML); model archive (ZIP)"],
  ],
  notes: [
    "^a^Counts refer to the options exposed in the Galaxy tool form of GLEAM suite v1.0.0; complete option lists are given in Supplementary Table S1.",
  ],
});

/* ------------------------------------------------------------------ */
/* Table 2 — software stack                                            */
/* ------------------------------------------------------------------ */
const table2 = buildTable({
  number: 2,
  title: "Python software stack of the GLEAM learner modules.^a^",
  headers: ["Package", "Tabular Learner", "Image Learner", "Multimodal Learner"],
  widths: [2200, 2630, 2630, 2631],
  rows: [
    ["pycaret", "3.3.2", "–", "–"],
    ["ludwig", "–", "0.10.1", "–"],
    ["autogluon", "–", "–", "1.4.0"],
    ["torch", "–", "2.2.1^b^", "2.7.1"],
    ["torchvision", "–", "0.17.1^b^", "0.22.1"],
    ["transformers", "–", "4.38.2^b^", "4.49.0"],
    ["scikit-learn", "1.4.2", "1.4.1.post1", "1.7.2"],
    ["pandas", "2.1.4", "2.1.4", "2.3.3"],
    ["numpy", "1.26.4", "1.26.4", "2.1.3"],
    ["scipy", "1.11.4", "1.12.0", "1.15.3"],
    ["matplotlib", "3.7.5", "3.8.3", "3.10.7"],
    ["plotly", "5.24.1", "5.19.0", "6.5.0"],
    ["shap", "0.44.1", "–", "0.49.1"],
    ["joblib", "1.3.2", "–", "–"],
    ["h5py", "3.13.0", "–", "–"],
    ["PyYAML", "–", "6.0", "6.0.3"],
    ["Pillow", "–", "10.2.0", "11.3.0"],
    ["MetaFormer", "–", "vendored^c^", "–"],
  ],
  notes: [
    "^a^Versions are those resolved by the pinned container images galaxy-pycaret:3.3.2, galaxy-ludwig-gpu:0.10.1 and multimodal-learner:1.4.0 (Table 1). A dash indicates that the package is not part of that module's runtime stack.",
    "^b^Installed transitively as a Ludwig dependency.",
    "^c^MetaFormer model definitions are distributed with the Image Learner tool rather than installed from PyPI.",
  ],
});

/* ------------------------------------------------------------------ */
/* Table 3 — report content                                            */
/* ------------------------------------------------------------------ */
const table3 = buildTable({
  number: 3,
  title: "Number of elements in the HTML report generated by each GLEAM learner module.^a^",
  headers: ["Report element", "Tabular Learner", "Image Learner", "Multimodal Learner"],
  widths: [2500, 2530, 2530, 2531],
  rows: [
    ["Interactive plots", "23", "21", "23"],
    ["Summary tables", "7", "5", "9"],
  ],
  notes: [
    "^a^Maximum number of elements produced for a binary-classification run. The elements generated for a particular run depend on the task type and on the options selected.",
  ],
});

/* ------------------------------------------------------------------ */
/* Table 4 — methodological safeguards                                 */
/* ------------------------------------------------------------------ */
const table4 = buildTable({
  number: 4,
  title: "Best-practice safeguards and default settings of the GLEAM learner modules.",
  headers: ["Aspect", "Tabular Learner", "Image Learner", "Multimodal Learner"],
  widths: W4,
  rows: [
    ["Separate external test set", "Optional upload", "Not supported; a single metadata CSV is used", "Optional upload"],
    ["User-supplied split column", "Not supported", "Honoured when present in the metadata CSV", "Not supported"],
    ["Automatic internal splitting", "Yes; train fraction, default 0.7", "Yes; train/validation/test proportions, default 0.7/0.1/0.2", "Yes; train/validation/test proportions, default 0.7/0.1/0.2"],
    ["Cross-validation", "Yes; k-fold, default 10 folds, enabled by default", "Not supported; a single train/validation/test split is used", "Yes; k-fold, default 5 folds, disabled by default"],
    ["Leakage-aware grouping", "Optional sample ID column enforces group-aware splits and group k-fold", "Optional sample ID column enforces group-aware splits when no split column is supplied", "Optional sample ID column enforces group-aware splits and stratified group k-fold"],
    ["Binary decision threshold", "Metric-optimized or user-specified; the value and its source are reported", "Metric-optimized on the validation split or user-specified", "Metric-optimized on the validation split or user-specified"],
    ["Transparency and explainability", "Candidate-model comparison, feature importance, SHAP values, ROC and precision–recall curves, calibration curve, threshold diagnostics", "Learning curves, confusion matrix, ROC and precision–recall curves, calibration curve, prediction-confidence diagnostics, Grad-CAM overlays for convolutional encoders", "Feature importance, SHAP summary, force and waterfall plots, confusion matrix, ROC and precision–recall curves, calibration curve, threshold diagnostics"],
    ["Reproducibility artefacts", "Random seed; serialized best model; best-model parameter table; HTML report", "Random seed; Ludwig model directory; per-sample predictions; training statistics; HTML report", "Random seed and optional deterministic mode; model archive; metrics JSON; training configuration YAML; HTML report"],
  ],
});

/* ------------------------------------------------------------------ */
/* Table 5 — Galaxy interface option groups                            */
/* ------------------------------------------------------------------ */
const table5 = buildTable({
  number: 5,
  title: "User-configurable option groups in the Galaxy tool forms of the GLEAM learner modules.",
  headers: ["Option group", "Tabular Learner", "Image Learner", "Multimodal Learner"],
  widths: [1600, 2830, 2830, 2831],
  rows: [
    ["Task selection", "Classification or regression", "Automatic inference, binary classification, multi-class classification or regression", "Inferred automatically from the target column"],
    ["Model or backbone", "18 classification or 25 regression estimators; an empty selection compares all of them", "129 backbones (74 TorchVision, 55 MetaFormer); pretrained or from scratch, with optional encoder fine-tuning", "Three quality presets; 8 text and 76 image backbones"],
    ["Objective metric", "Classification: accuracy, ROC-AUC, precision, recall, F1, Cohen's kappa, log loss, PR-AUC. Regression: R^2^, MAE, MSE, RMSE, RMSLE, MAPE", "Task-specific Ludwig validation metrics (five binary, two multi-class, five regression) or an automatic default", "Automatic selection or one of 26 AutoGluon evaluation metrics"],
    ["Data splitting", "Optional external test set; train fraction (default 0.7); k-fold cross-validation (default 10 folds)", "Split column honoured when present, otherwise train/validation/test proportions (default 0.7/0.1/0.2)", "Optional external test set; validation fraction (default 0.2) or train/validation/test proportions (default 0.7/0.1/0.2); optional k-fold cross-validation (default 5 folds)"],
    ["Leakage control", "Optional sample ID column for group-aware splitting and group k-fold", "Optional sample ID column for group-aware splitting when no split column is supplied", "Optional sample ID column for group-aware splitting and stratified group k-fold"],
    ["Binary threshold policy", "Automatic optimization for F1, accuracy, precision, recall, Cohen's kappa or MCC, or a manual value", "Automatic optimization for F1, accuracy, balanced accuracy, precision, recall or MCC, or a manual value", "Automatic, or manual with a choice of seven optimization metrics or an explicit threshold value"],
    ["Training regime", "Optional hyperparameter tuning of the selected best model", "Epochs (default 10), early-stopping patience (default 5), optional learning rate and batch size", "Training time limit, optional epochs, learning rate and batch size, and free-form AutoGluon hyperparameter overrides"],
    ["Preprocessing", "Normalization, feature selection, outlier removal, multicollinearity removal, polynomial features, class-imbalance correction", "Image resizing (13 presets) and six optional augmentations", "Handling of rows with missing images (drop or placeholder)"],
    ["Reproducibility", "Random seed (default 42)", "Random seed (default 42)", "Random seed (default 42) and optional deterministic mode"],
  ],
  notes: [
    "MAE, mean absolute error; MAPE, mean absolute percentage error; MCC, Matthews correlation coefficient; MSE, mean squared error; PR-AUC, area under the precision–recall curve; RMSE, root mean squared error; RMSLE, root mean squared logarithmic error; ROC-AUC, area under the receiver operating characteristic curve.",
  ],
});

(async () => {
  await write("Table1_module_overview.docx", table1);
  await write("Table2_software_versions.docx", table2);
  await write("Table3_report_content.docx", table3);
  await write("Table4_best_practice_defaults.docx", table4);
  await write("Table5_galaxy_option_groups.docx", table5);
})();
