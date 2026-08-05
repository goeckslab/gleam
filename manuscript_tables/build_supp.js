const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell,
  WidthType, BorderStyle, AlignmentType, VerticalAlign, PageBreak,
  Footer, PageNumber, ExternalHyperlink,
} = require("docx");

const FONT = "Arial";
const BODY = 16;      // 8 pt  (table body)
const LEGEND = 18;    // 9 pt  (figure legends, table captions)
const NOTE = 14;      // 7 pt  (table footnotes)
const TEXT = 20;      // 10 pt (title page body)

const NO = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const RULE_HEAVY = { style: BorderStyle.SINGLE, size: 12, color: "000000" };
const RULE_LIGHT = { style: BorderStyle.SINGLE, size: 6, color: "000000" };

// ---- page geometry: US Letter, 178 mm (double-column) text width ----
const MM = 56.6929;                 // DXA per mm (1440 / 25.4)
const PAGE_W = 12240, PAGE_H = 15840;
const MARGIN_X = Math.round((PAGE_W - 178 * MM) / 2);  // -> 178 mm text width
const MARGIN_Y = Math.round(18 * MM);
const TEXT_W = PAGE_W - 2 * MARGIN_X;                  // 10091 DXA
const FIG_MAX_H_MM = 205;
const FIG_MAX_W_MM = 178;
const PX_PER_MM = 3.779528;

function runs(text, opts = {}) {
  const parts = String(text).split(/(\*\*[^*]+\*\*|\^[^^]+\^)/g).filter(Boolean);
  return parts.map((p) => {
    if (p.startsWith("**") && p.endsWith("**"))
      return new TextRun({ text: p.slice(2, -2), bold: true, font: FONT, size: opts.size || BODY });
    if (p.startsWith("^") && p.endsWith("^"))
      return new TextRun({ text: p.slice(1, -1), superScript: true, font: FONT, size: opts.size || BODY });
    return new TextRun({ text: p, font: FONT, size: opts.size || BODY, bold: !!opts.bold, italics: !!opts.italics });
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
    borders: { top: opts.top || NO, bottom: opts.bottom || NO, left: NO, right: NO },
    verticalAlign: VerticalAlign.TOP,
    children: [para(text, { bold: opts.bold, size: opts.size })],
  });
}

function buildTable({ label, title, headers, rows, widths, notes }) {
  const total = widths.reduce((a, b) => a + b, 0);
  const headerRow = new TableRow({
    tableHeader: true,
    children: headers.map((h, i) => cell(h, widths[i], { bold: true, top: RULE_HEAVY, bottom: RULE_LIGHT })),
  });
  const bodyRows = rows.map((r, ri) =>
    new TableRow({
      children: r.map((c, i) => cell(c, widths[i], { bottom: ri === rows.length - 1 ? RULE_HEAVY : NO })),
    })
  );
  const table = new Table({
    columnWidths: widths,
    width: { size: total, type: WidthType.DXA },
    margins: { top: 60, bottom: 60, left: 0, right: 110 },
    rows: [headerRow, ...bodyRows],
  });
  const out = [
    new Paragraph({
      children: [
        new TextRun({ text: `${label}. `, bold: true, font: FONT, size: LEGEND }),
        ...runs(title, { size: LEGEND }),
      ],
      spacing: { after: 120, line: 240, lineRule: "auto" },
    }),
    table,
  ];
  (notes || []).forEach((n, i) =>
    out.push(new Paragraph({
      children: runs(n, { size: NOTE }),
      spacing: { before: i === 0 ? 100 : 20, after: 20, line: 200, lineRule: "auto" },
    }))
  );
  return out;
}

/* ================================================================== */
/* Figure legends (verbatim from the source document, except where     */
/* noted in CORRECTIONS below)                                         */
/* ================================================================== */
const legends = [
  "Feature importance for Tabular Learner prediction of immunotherapy response. Feature-importance output for the logistic regression model selected by Tabular Learner in the immunotherapy response task. Variables are ranked using model-based importance, mean absolute SHAP value and permutation importance, providing complementary views of the features that contributed most to prediction.",
  "Tabular Learner configuration for automatic decision-threshold selection. Configuration report for the immunotherapy response analysis using automatic threshold selection. The report records the input dataset, target label, selected logistic regression model, evaluation setup, threshold-optimization metric and selected decision threshold. Tabular Learner optimized F1 score using cross-validated training predictions and selected a probability threshold of 0.269, which was then applied to the held-out test set.",
  "Test performance of Tabular Learner using the automatically selected decision threshold. Test Summary report for the selected logistic regression model evaluated on the held-out test set after automatic threshold selection. Tabular Learner optimized F1 score using cross-validated training predictions and selected a probability threshold of 0.269. At this operating point, the model achieved a ROC AUC of 0.760, AUPRC of 0.553, precision of 0.427, recall of 0.669 and F1 score of 0.521 on the held-out test set.",
  "Image Learner configuration for the class-balanced HAM10000 subset without lesion-level grouping. Configuration report for the class-balanced HAM10000 benchmark subset, which included 1,400 dermatoscopic images after horizontal flipping. The report records class balance, training settings, the pretrained CAFormer S18 384 backbone and the stratified train, validation and test split used without grouping related lesion images.",
  "Test performance on the class-balanced HAM10000 subset without lesion-level grouping. Test Summary report for the Image Learner model trained on the class-balanced HAM10000 subset. The report shows held-out test performance, confusion patterns across the seven diagnostic classes, per-class metrics, confidence distributions and Grad-CAM examples. The model reached a held-out test accuracy of 0.8464 and F1 score of 0.8464.",
  "Image Learner configuration for the full HAM10000 dataset without lesion-level grouping. Configuration report for the full HAM10000 analysis of 10,015 dermatoscopic images across seven diagnostic classes. The report records dataset composition, training settings, image preprocessing, augmentation settings and fine-tuning of the pretrained CAFormer S18 384 backbone. The train, validation and test split was generated without grouping related lesion images.",
  "Test performance on the full HAM10000 dataset without lesion-level grouping. Test Summary report for the Image Learner model trained on the full HAM10000 dataset. The report shows held-out test metrics, confusion patterns, per-class performance, confidence distributions and Grad-CAM examples across the seven diagnostic classes. The model reached a held-out test accuracy of 0.8807 and micro F1 score of 0.8807 under a split that did not group related lesion images.",
  // S8  -- CORRECTED: source text described an ungrouped split (contradicting the title)
  "Image Learner configuration for the class-balanced HAM10000 subset with lesion-level grouping. Configuration report for the class-balanced HAM10000 benchmark subset, which included 1,400 dermatoscopic images after horizontal flipping. The report records class balance, training settings, the pretrained CAFormer S18 384 backbone and the stratified train, validation and test split, in which all images from the same lesion were kept within a single split.",
  // S9  -- CORRECTED: source text reported the ungrouped metrics (0.8464/0.8464)
  "Test performance on the class-balanced HAM10000 subset with lesion-level grouping. Test Summary report for the Image Learner model trained on the class-balanced HAM10000 subset. The report shows held-out test performance, confusion patterns across the seven diagnostic classes, per-class metrics, confidence distributions and Grad-CAM examples. The model reached a held-out test accuracy of 0.6809 and micro F1 score of 0.6809.",
  // S10 -- CORRECTED: source text described an ungrouped split (contradicting the title)
  "Image Learner configuration for the full HAM10000 dataset with lesion-level grouping. Configuration report for the full HAM10000 analysis of 10,015 dermatoscopic images across seven diagnostic classes. The report records dataset composition, training settings, image preprocessing, augmentation settings and fine-tuning of the pretrained CAFormer S18 384 backbone. The train, validation and test split was generated so that all images from the same lesion were kept within a single split.",
  // S11 -- CORRECTED: source text reported the ungrouped metrics (0.8807/0.8807)
  "Test performance on the full HAM10000 dataset with lesion-level grouping. Test Summary report for the Image Learner model trained on the full HAM10000 dataset. The report shows held-out test metrics, confusion patterns, per-class performance, confidence distributions and Grad-CAM examples across the seven diagnostic classes. The model reached a held-out test accuracy of 0.8036 and micro F1 score of 0.8036 under a lesion-grouped split.",
  "Multimodal Learner configuration for recurrence prediction in the HANCOCK cohort. Configuration report for recurrence prediction from structured clinical variables, ICD text and paired CD3 and CD8 histology images. The report records the tabular, text, image and fusion modules, selected backbones, threshold settings, training parameters and evaluation metrics used for the HANCOCK analysis.",
  "Test performance of Multimodal Learner for HANCOCK recurrence prediction. Test Summary report for the HANCOCK recurrence analysis. The report shows held-out test metrics, confusion patterns between recurrence and no recurrence, class-level precision, recall and F1 score, ROC and precision-recall curves and prediction confidence distributions. On the held-out test set, no recurrence achieved precision, recall and F1 score of 0.85, 0.95 and 0.90, while recurrence achieved 0.76, 0.48 and 0.59. The model achieved a ROC AUC of 0.78 on the predefined test partition.",
];

// PNG dimensions, read from the recovered files
function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}

function figureBlock(n, isLast) {
  const file = path.join(__dirname, "figs", `figS${n}.png`);
  const { w, h } = pngSize(file);
  const ratio = h / w;
  // largest size that fits the text block and the double-column width limit
  let wMM = Math.min(FIG_MAX_W_MM, FIG_MAX_H_MM / ratio);
  let hMM = wMM * ratio;
  const wPx = Math.round(wMM * PX_PER_MM);
  const hPx = Math.round(hMM * PX_PER_MM);
  const dpi = Math.round(w / (wMM / 25.4));

  const text = legends[n - 1];
  // bold the label and the title sentence; the rest is regular
  const dot = text.indexOf(". ");
  const title = text.slice(0, dot + 1);
  const rest = text.slice(dot + 1);

  const out = [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 160, line: 240, lineRule: "auto" },
      children: [new ImageRun({ data: fs.readFileSync(file), type: "png", transformation: { width: wPx, height: hPx } })],
    }),
    new Paragraph({
      alignment: AlignmentType.BOTH,
      spacing: { after: 0, line: 240, lineRule: "auto" },
      children: [
        new TextRun({ text: `Supplementary Fig. S${n} | `, bold: true, font: FONT, size: LEGEND }),
        new TextRun({ text: title, bold: true, font: FONT, size: LEGEND }),
        new TextRun({ text: rest, font: FONT, size: LEGEND }),
      ],
    }),
  ];
  if (!isLast) out.push(new Paragraph({ children: [new PageBreak()] }));
  console.log(`  Fig S${n}: ${w}x${h} px -> ${wMM.toFixed(1)} x ${hMM.toFixed(1)} mm  (${dpi} dpi)`);
  return out;
}

/* ================================================================== */
/* Title page                                                          */
/* ================================================================== */
const titlePage = [
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 240, line: 300, lineRule: "auto" },
    children: [new TextRun({
      text: "GLEAM: accessible, reproducible, and best-practice machine learning for biomedical research",
      bold: true, font: FONT, size: 32,
    })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 240, line: 260, lineRule: "auto" },
    children: [new TextRun({ text: "Supplementary Material", bold: true, font: FONT, size: 26 })],
  }),
  para("Paulo Lyra Jr^1,α^, Junhao Qiu^1,α^, Khai Dang^1^, Alyssa Pybus^1^, Isis Narvaez-Bandera^1^, Maansi Singh^1^, Qiang Gu^2^, Luke Sargent^2^, Allison Creason^2^, and Jeremy Goecks^1,*^",
    { size: TEXT, alignment: AlignmentType.CENTER, after: 240, line: 260 }),
  para("^1^Department of Machine Learning, Moffitt Cancer Center, Tampa, FL 33612, USA.", { size: 18, after: 80, line: 240 }),
  para("^2^Computational Biology Program, Knight Cancer Institute, Oregon Health & Science University, Portland, OR 97239, USA.", { size: 18, after: 80, line: 240 }),
  para("^α^Equal contributions.", { size: 18, after: 80, line: 240 }),
  para("^*^Corresponding author: jeremy.goecks@moffitt.org", { size: 18, after: 320, line: 240 }),
  new Paragraph({
    spacing: { before: 240, after: 120, line: 240, lineRule: "auto" },
    children: [new TextRun({ text: "Contents", bold: true, font: FONT, size: 22 })],
  }),
  para("Supplementary Figures S1–S13", { size: 18, after: 60, line: 240 }),
  para("Supplementary Tables S1–S5", { size: 18, after: 60, line: 240 }),
  new Paragraph({ children: [new PageBreak()] }),
];

/* ================================================================== */
/* Supplementary tables                                                */
/* ================================================================== */
const W4 = [1700, 2796, 2797, 2797];

const tableS1 = buildTable({
  label: "Supplementary Table S1",
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
  notes: ["^a^Counts refer to the options exposed in the Galaxy tool form of GLEAM suite v1.0.0; complete option lists are given in Supplementary Table S5."],
});

const tableS2 = buildTable({
  label: "Supplementary Table S2",
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
    "^a^Versions are those resolved by the pinned container images galaxy-pycaret:3.3.2, galaxy-ludwig-gpu:0.10.1 and multimodal-learner:1.4.0 (Supplementary Table S1). A dash indicates that the package is not part of that module's runtime stack.",
    "^b^Installed transitively as a Ludwig dependency.",
    "^c^MetaFormer model definitions are distributed with the Image Learner tool rather than installed from PyPI.",
  ],
});

const tableS3 = buildTable({
  label: "Supplementary Table S3",
  title: "Number of elements in the HTML report generated by each GLEAM learner module.^a^",
  headers: ["Report element", "Tabular Learner", "Image Learner", "Multimodal Learner"],
  widths: [2500, 2530, 2530, 2531],
  rows: [
    ["Interactive plots", "23", "21", "23"],
    ["Summary tables", "7", "5", "9"],
  ],
  notes: ["^a^Maximum number of elements produced for a binary-classification run. The elements generated for a particular run depend on the task type and on the options selected."],
});

const tableS4 = buildTable({
  label: "Supplementary Table S4",
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

const tableS5 = buildTable({
  label: "Supplementary Table S5",
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
  notes: ["MAE, mean absolute error; MAPE, mean absolute percentage error; MCC, Matthews correlation coefficient; MSE, mean squared error; PR-AUC, area under the precision–recall curve; RMSE, root mean squared error; RMSLE, root mean squared logarithmic error; ROC-AUC, area under the receiver operating characteristic curve."],
});

/* ================================================================== */
const children = [...titlePage];
for (let n = 1; n <= 13; n++) children.push(...figureBlock(n, false));
const tables = [tableS1, tableS2, tableS3, tableS4, tableS5];
tables.forEach((t, i) => {
  children.push(...t);
  if (i < tables.length - 1) children.push(new Paragraph({ children: [new PageBreak()] }));
});

const doc = new Document({
  styles: { default: { document: { run: { font: FONT, size: BODY } } } },
  sections: [{
    properties: {
      page: {
        size: { width: PAGE_W, height: PAGE_H },
        margin: { top: MARGIN_Y, right: MARGIN_X, bottom: MARGIN_Y, left: MARGIN_X },
      },
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 16 })],
        })],
      }),
    },
    children,
  }],
});

Packer.toBuffer(doc).then((b) => {
  fs.writeFileSync("GLEAM_Supplementary_Material.docx", b);
  console.log("text width:", (TEXT_W / MM).toFixed(1), "mm | margins:", (MARGIN_X / MM).toFixed(1), "mm");
  console.log("wrote GLEAM_Supplementary_Material.docx", (b.length / 1048576).toFixed(1), "MB");
});
