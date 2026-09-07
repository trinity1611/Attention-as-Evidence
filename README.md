# Attention as Evidence

*Coupling Grad-CAM Localisation to Vision-Language Report Generation for Chest Radiographs*

Multi-label classification of 14 thoracic findings on NIH ChestX-ray14, with
explanation heatmaps that are **measured against radiologist boxes** rather than
just shown, and (final phase) a medical vision-language model that drafts a
report grounded in what the classifier looked at.

Every number and figure below is regenerated from `outputs/` by the scripts in
this repository. The full write-up is `Project_Briefing.docx`, built by
`scripts/make_report_doc.py`.

> Research and educational use only. Not a medical device and not for clinical
> diagnosis.

---

## Key results

**Classification** (macro AUROC over 14 findings; every model scored only on
images it never trained on)

| Model | Training data | 5-fold CV (5,606 films) | Same 1,293 held-out films | Δ vs sample CNN |
|---|---|--:|--:|--:|
| DenseNet-121, fine-tuned 320 px | 5,606 (sample) | 0.757 ± 0.022 | 0.720 | — |
| RAD-DINO frozen + linear layer | 5,606 (sample) | 0.755 ± 0.020 | 0.730 | +0.010 [−0.010, +0.030] |
| Ensemble of the two | 5,606 (sample) | **0.786 ± 0.022** (p = 0.001) | 0.756 | +0.036 [+0.024, +0.048] |
| **DenseNet-121, fine-tuned 224 px** | **77,988 (full NIH)** | 0.812 on official test (n = 25,596) | **0.807** | **+0.087 [+0.071, +0.104]** |

Data volume dominated every other lever tried — pretraining source,
architecture, resolution, schedule, foundation-model features, ensembling —
by more than 2×. The full-data model wins on all 13 scorable findings.

**Deployed model, final metrics** (`densenet121_full224.pt`; NIH official test
split; thresholds fitted on a random half of the test films, threshold metrics
reported on the other 12,798):

| Macro AUROC | Macro AUPRC | Accuracy (macro) | Precision | Recall | F1 |
|--:|--:|--:|--:|--:|--:|
| **0.812** | 0.276 | 87.7 % | 0.292 (micro 0.314) | 0.436 (micro 0.523) | 0.341 (micro 0.392) |

Accuracy is dominated by negatives at 0.3–18 % prevalence and is reported only
because it is conventional; AUROC / AUPRC / F1 carry the information. At the
per-finding thresholds, 68 % of abnormal films get at least one correct finding
and 48 % of normal films are left clean. Per finding: `outputs/metrics/final_model_per_class.csv`.

**Explainability** (53 radiologist boxes; pointing game = heatmap peak inside
the box, chance 7.2 %)

| Configuration | Hits / 53 | IoU | Notes |
|---|--:|--:|---|
| Original default: block 4, Grad-CAM | 19 (35.8 %) | 0.230 | ranked 81st of 111 configurations |
| **Adopted: blocks 3+4 fused, LayerCAM** | **31 (58.5 %)** | **0.260** | best IoU of all 111; one forward pass |
| Sweep winner: block 3, LayerCAM, smoothing, 320 px | 33 (62.3 %) | 0.166 | sharper peaks, useless regions; not adopted |

The network **layer**, not the attribution algorithm, was the bottleneck.
Cardiomegaly localises 9/10; Nodule 0/3. The same method on the sample-trained
model scores 16/53 — more data improved *where* the model looks, not only what
it predicts.

**Triage** (any finding vs none, 46/54 split — the one place accuracy means
something): ensemble AUROC 0.761, accuracy 0.71, sensitivity 0.71 / specificity 0.72.

**Report generation** (MedGemma 4B via Ollama; 200 films × 3 briefings = 600
reports): giving the model the classifier's probabilities alone made it assert
2.9× more findings than from the image and leave no normal film clean; adding
the Grad-CAM zone cut unsupported assertions by **1.07 per report (95 % CI
0.80–1.37)** and reduced false findings against the ground truth from 5.0 to
3.9. Grounding helps; no condition produces a draft fit to go out unread.

---

## Repository layout

```
config.yaml                  all paths and hyperparameters
requirements.txt
Project_Briefing.docx        the full report (regenerate with scripts/make_report_doc.py)

src/
  config.py                  class list, config loader, whitespace-tolerant CSV reader
  prepare_data.py            resize cache, patient-grouped stratified folds, distribution figures
  dataset.py                 Dataset, augmentation, capped pos_weight
  model.py                   DenseNet-121 / EfficientNet-B0 / CheXpert-DenseNet builders
  train.py                   two-stage fine-tuning, 5-fold CV (checkpoints write-protected)
  features.py                frozen RAD-DINO embeddings, cached once
  probe.py                   linear probe on RAD-DINO features, same folds
  infer.py                   flip TTA, deployment ensemble, prediction grids
  evaluate.py                per-class AUROC, ROC, thresholds, paired stats (auto-discovers models)
  confusion.py               per-finding 2x2 matrices, precision / recall / F1
  plots.py                   loss curves, PR curves, calibration, metric bars
  triage.py                  binary any-finding-vs-none evaluation
  gradcam.py                 CAM overlays + bbox localisation metrics; configurable method / layer
  gradcam_sweep.py           111-configuration explanation sweep (resumable)
  medgemma.py                Ollama client, prompt builder (conditions A/B/C), CAM-to-zone text, parser
  report_study.py            the 600-report grounding study: inputs, run (resumable), scoring
  palette.py                 one fixed colour per finding, app chrome colours, film tints
  viz.py                     per-finding alpha heatmaps, contours, composites, static bar chart
  service.py                 app back-end: analyse a film -> probabilities, CAMs, zones, metrics, report
  export.py                  JSON bundle and PDF for one analysed film
  final_metrics.py           deployed model on the official test split: AUROC, AUPRC, accuracy, P/R/F1

app/
  Home.py                    landing page with the animated hero (real films, real heatmaps)
  pages/1_Analyse.py         upload -> findings -> heatmaps -> MedGemma draft -> JSON / PDF / PNG
  pages/2_Dashboard.py       every model, phase and result, read live from outputs/
  _common.py                 shared CSS, cached service, widgets

scripts/
  make_report_doc.py         builds Project_Briefing.docx
  briefing_sections*.py, briefing_appendix.py, make_diagrams*.py, make_formulas.py,
  make_gallery_figures.py    report content, diagrams, formula images, contact sheets
  make_visuals.py            per-class exemplar images
  test_predictions.py        inspect held-out predictions on sampled films
  import_kaggle_run.py       register the full-data run; shared-test comparison

kaggle/
  train_full_nih.py          self-contained full-dataset trainer for a Kaggle GPU
  train_full_nih.ipynb       the same as an uploadable notebook
  README.md                  step-by-step: attach dataset, run, bring results home

checkpoints/                 fold weights + densenet121_full224.pt (gitignored)
outputs/
  folds.csv                  image -> patient, fold, 14 labels
  sample_images/             3 exemplars per finding + overview grid
  distribution/              class counts, label cardinality, co-occurrence, fold balance
  predictions/               per-fold dumps, full-test dump, prediction grids
  plots/                     every figure; gradcam/{bbox,cases,sheets}/<config>/<model>/
  metrics/                   every table (CSV / JSON); res224_study/ archives the 224 px runs
```

Image data lives at `C:/ml-data/nih-cxr`, **outside** this OneDrive-synced
folder, on purpose.

---

## Reproduce

```bash
python -m venv venv && venv\Scripts\activate          # Python 3.11
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# data: the official 5 % sample (Kaggle: nih-chest-xrays/sample, ~2 GB unique)
#   -> C:/ml-data/nih-cxr/sample/images + sample_labels.csv
#   plus test_list.txt / train_val_list.txt / BBox_List_2017.csv from the full
#   dataset page into C:/ml-data/nih-cxr/meta/  (small text files)

python -m src.prepare_data                     # 352 px cache, folds, distribution figures
python -m src.train                            # DenseNet-121, 5 folds, ~25 min/fold on an RTX 4060
python -m src.infer                            # TTA predictions + prediction grids
python -m src.features && python -m src.probe  # RAD-DINO embeddings (once) + linear probe
python -m src.evaluate --suffix _tta           # AUROC tables, ROC, thresholds, paired stats
python -m src.confusion --suffix _tta          # confusion matrices, P / R / F1
python -m src.plots --suffix _tta              # loss / PR / calibration figures
python -m src.triage                           # binary triage
python -m src.gradcam                          # heatmaps + bbox metrics, adopted configuration
python -m src.gradcam_sweep                    # optional: the 111-configuration sweep (hours)

# full-data model: run kaggle/train_full_nih.ipynb on Kaggle (see kaggle/README.md), then
python scripts/import_kaggle_run.py --dir <downloaded outputs>
python -m src.final_metrics                    # deployed model: AUROC / AUPRC / accuracy / P / R / F1

python -m src.report_study --run --n 200       # MedGemma grounding study (needs `ollama serve`; ~3 h)
python -m src.report_study --score
python scripts/make_report_doc.py              # rebuild Project_Briefing.docx

# the app
ollama serve                                   # in another terminal, once; medgemma:latest must be pulled
python scripts/make_hero_assets.py             # once: pre-render the landing-page hero
streamlit run app/Home.py                      # http://localhost:8501
```

`src.train` refuses to overwrite an existing fold checkpoint without
`--overwrite`; the 320 px DenseNet is the baseline every later experiment is
paired against.

---

## Data

The official 5 % sample of NIH ChestX-ray14 (Wang et al., CVPR 2017; CC0):
**5,606 films from 4,230 patients**, 1024 × 1024, labels NLP-mined from reports
(~90 % accurate). 984 radiologist-drawn boxes ship with the full set; 53 fall
on our films, all in the official test split.

| Finding | n | Finding | n |
|---|--:|---|--:|
| Infiltration | 967 | Pleural_Thickening | 176 |
| Effusion | 644 | Cardiomegaly | 141 |
| Atelectasis | 508 | Emphysema | 127 |
| Nodule | 313 | Edema ⚠ | 118 |
| Mass | 284 | Fibrosis ⚠ | 84 |
| Pneumothorax | 271 | Pneumonia ⚠ | 62 |
| Consolidation | 226 | Hernia ⚠ | 13 |

⚠ under 150 positives: reported but underpowered. *No Finding* (3,044, 54 %) is
the all-zero vector, not a 15th class.

**Splits.** Patient-grouped, stratified 5-fold CV; `prepare_data.py` asserts
zero patients span folds. Every image is scored once, out-of-fold. The full-data
model uses NIH's official `train_val` / `test` lists (val split by patient).
The **1,293 films in both our sample and the official test list** are the
common ground for comparing all models fairly.

---

## Method

- **Multi-label, not multi-class.** 14 independent sigmoids, weighted BCE with
  `pos_weight` capped at 20 (uncapped, Hernia's would be ~430), label smoothing
  0.05. Never softmax; never plain accuracy.
- **Two-stage fine-tuning.** Head only (5 epochs, lr 1e-3), then everything with
  discriminative LRs (1e-5 backbone / 1e-4 head, cosine, 30 epochs, patience 8).
- **Augmentation.** Random resized crop, hflip, ±10° rotation, small translate,
  brightness / contrast. Flip TTA at inference (+0.004). hflip is anatomically
  questionable for a chest and is noted as a limitation.
- **RAD-DINO probe.** Frozen `microsoft/rad-dino` CLS embeddings (768-d,
  cached), standardised per fold, one linear layer, same loss. *Caveat:*
  RAD-DINO's self-supervised pretraining included all NIH images (no labels).
- **Evaluation.** AUROC (primary), AUPRC with lift over prevalence, P/R/F1 at
  per-class thresholds fitted on train and applied to held-out, per-finding
  confusion matrices, calibration. Paired t-test across folds; paired bootstrap
  on the shared 1,293 films.

---

## Results in detail

### Sample-trained DenseNet-121

| Fold | 224 px, 15 ep | 320 px, 30 ep | 320 px + TTA |
|---|--:|--:|--:|
| 0 | 0.7359 | 0.7587 | 0.7619 |
| 1 | 0.7204 | 0.7471 | 0.7532 |
| 2 | 0.7399 | 0.7640 | 0.7698 |
| 3 | 0.7043 | 0.7217 | 0.7225 |
| 4 | 0.7398 | 0.7724 | 0.7783 |
| **Mean** | **0.7281** | **0.7528** | **0.7571** |

Resolution and a converged schedule gave +0.025 on every fold; TTA +0.004.
Largest per-finding gains on small structures (Emphysema +0.074, Pneumothorax
+0.049); Nodule unchanged. Held-out P/R/F1 at F1-optimal thresholds: 0.170 /
0.214 / 0.188 macro; on abnormal films at least one finding is detected 41 % of
the time. Macro AUPRC 0.144 (3.7× lift over prevalence).

**224 px study (archived in `outputs/*/res224_study/`).** DenseNet-ImageNet
0.728, DenseNet-CheXpert 0.723 (Δ −0.005, p = 0.36), EfficientNet-B0 0.720
(Δ −0.008, p = 0.25). Neither pretraining source nor architecture mattered; all
three were also under-trained (peaked on the final epoch), which motivated the
longer 320 px schedule.

### Four models on the same 1,293 held-out films

| Model | Macro AUROC (13 classes) | Δ vs sample CNN | 95 % CI |
|---|--:|--:|---|
| DenseNet-121, sample, 320 px | 0.720 | — | — |
| RAD-DINO linear probe | 0.730 | +0.010 | [−0.010, +0.030] |
| Ensemble | 0.756 | +0.036 | [+0.024, +0.048] |
| **DenseNet-121, full data, 224 px** | **0.807** | **+0.087** | **[+0.071, +0.104]** |

Full-data gains are largest on small-structure findings (Mass +0.18, Emphysema
+0.12, Cardiomegaly +0.12, Edema +0.11, Nodule +0.10) — achieved at a *lower*
resolution than the local model, in 8 epochs and one T4 GPU-hour. The official
test split is harder than the 5-fold average (the same sample model scores 0.757
there and 0.720 here). The full model was still improving at epoch 8.

### Grad-CAM, measured

`src.gradcam_sweep` scored {block 4, block 3, block 3+4} × {Grad-CAM, Grad-CAM++,
HiResCAM, LayerCAM, XGrad-CAM} × {none, aug, eigen, both} × {224, 320 px} on the
53 boxes, selection rule fixed beforehand (pointing hits, tie-break IoU@0.5).
111 of 120 ran (the slowest fused-layer XGrad-CAM variants were skipped; the
family never led).

- Layer mattered most: best block-4 configuration 23 hits, best block-3 33.
- Method second: LayerCAM and HiResCAM led every layer group.
- Smoothing and 320 px inference: marginal, 2–6× the compute.
- The strict winner produces 5 %-of-image blobs (Cardiomegaly IoU 0.59 → 0.12);
  the adopted runner-up is 2 hits behind with the best IoU of all 111.

Per finding, adopted configuration, full-data model: Cardiomegaly 9/10,
Pneumonia 7/9, Atelectasis 5/11 (was 0), Effusion 3/8, Mass 3/4, Pneumothorax
2/6, Infiltration 2/2, Nodule 0/3. On the confidently-wrong films the heatmaps
sit on lung fields and the costophrenic angle, not on burned-in text or
hardware — the errors are radiological confusions, not annotation shortcuts.

*Caveat:* 111 configurations ranked on 53 boxes is optimistic. Confirmation on
all 984 boxes (a 30-minute Kaggle job) is the next task.

---

## Phase 4 — MedGemma report generation: the grounding study

MedGemma 4B-it runs locally through Ollama (`medgemma:latest`, Q4_K_M, 100 %
GPU-resident). It is a *writer*, not a second classifier: given the film and
(depending on condition) the classifier's evidence, it returns a JSON block
(each of the 14 findings present / absent / uncertain, plus locations) followed
by a Findings / Impression / Recommendation draft. `src.medgemma` builds the
prompts and parses replies; `src.report_study` runs and scores the experiment.

**Design.** 200 films from the 1,293-image shared test set (100 abnormal, 100
normal), full-data model probabilities, thresholds fitted on the ~24,000
official-test films not in the study, LayerCAM zones for flagged findings.
Same film, same model, temperature 0, three briefings:

| Condition | Image | Probabilities | Grad-CAM zone |
|---|:-:|:-:|:-:|
| A | ✓ | | |
| B | ✓ | ✓ | |
| C | ✓ | ✓ | ✓ |

**Results (600 reports, 100 % parsed):**

| per report | A: image | B: + probs | C: + zones |
|---|--:|--:|--:|
| Findings asserted | 1.9 | 5.6 | 4.6 |
| Asserted but not flagged by classifier | 1.24 | 3.86 | 2.79 |
| …of which actually in the labels | 6.5 % | 4.5 % | 5.0 % |
| Flagged findings omitted | 62 % | 1 % | 1 % |
| False findings vs ground truth | 1.66 | 4.96 | 3.94 |
| Recall vs ground truth | 0.32 | 0.81 | 0.78 |
| Normal films left clean | 44 % | 0 % | 1 % |
| Stated location agrees with given zone | — | — | 93 % |

**C − B, unsupported findings per report: −1.07, 95 % CI [−1.37, −0.80]**
(paired bootstrap, n = 200). Grounding reduces over-reporting relative to raw
probabilities, and it does so against the ground truth too (3.9 vs 5.0 false
findings), not merely by copying the classifier.

**Reading it honestly.**
- Handing the model a probability list without location (B) makes it assert
  2.9× more findings than the image alone, and it never leaves a normal film
  clean. Raw scores are a bad prompt.
- Only ~5 % of the findings MedGemma adds beyond the classifier are in the
  labels: it is not, in the main, catching what the CNN missed.
- The trade-off is real: A is quiet and misses 68 % of true findings; C catches
  78 % at ~4 false findings per report. Every condition needs a human reader.
- Localisation agreement of 93 % in C is largely compliance -- the model
  repeats the zone it was given -- and is reported as such.

Files: `outputs/reports/reports.jsonl` (all 600), `outputs/reports/examples/`,
`outputs/metrics/report_study_*.csv`, `outputs/plots/report_study.png`.

---

## The application

Streamlit, three pages, dark clinical-workstation theme (navy ground, cyan
accent, glowing film frames, monospace numerics), one fixed colour per finding
used everywhere (bars, heatmaps, contours, report table, legend).

- **Home** — an animated hero built from *real* pipeline output on four held-out
  films: the film breathes, a scan line sweeps, each flagged finding's LayerCAM
  blooms in its colour with its contour and a probability counting up, and the
  MedGemma impression types itself out. Three stat cards read live from
  `outputs/metrics`. Respects `prefers-reduced-motion`.
- **Analyse** — upload a film or pick a held-out NIH film with a radiologist
  box. Probability bars with per-finding thresholds and the uncalibrated-score
  warning; per-finding heatmaps (alpha fill + 50 %-of-peak contour + peak mark),
  overlay any subset, film tint and opacity controls; attention zone per
  finding; **hit / miss / IoU / coverage when the film has a radiologist box**
  (recognised by content hash), otherwise the per-finding peak-in-box rate from
  our 53-box evaluation — never an invented metric. "Generate draft report"
  calls MedGemma under the grounded condition and shows the prose, the
  per-finding call table and the report scores (agreement with classifier,
  unsupported / omitted findings, localisation agreement, and precision /
  recall against NIH labels when known). Downloads: **JSON** (everything),
  **PDF** (film, heatmaps, findings, report, scores, disclaimer), **PNG**.
- **Dashboard** — KPIs for every phase; tabs for data, models (table +
  architecture diagrams), classification (four-model comparison and all plots),
  explainability (sweep, per-finding hits, contact sheets), report generation
  (study table, figure, example drafts), limitations.

All three pages pass Streamlit's `AppTest` harness end to end, including the
MedGemma call.

## Status

- [x] Data pipeline, patient-grouped folds, distribution analysis
- [x] Classifier: 224 px study (null result), 320 px model, RAD-DINO probe, ensemble
- [x] Full-data model on Kaggle; fair shared-test comparison
- [x] Grad-CAM: configuration sweep, bbox-validated localisation
- [x] Report (`Project_Briefing.docx`) with all figures
- [ ] Confirm Grad-CAM configuration on all 984 boxes (Kaggle)
- [x] MedGemma report generation + grounding study (600 reports)
- [x] Streamlit interface (Home with animated hero, Analyse, Dashboard; JSON / PDF / PNG export)
- [ ] Optional: full-data model at 320 px, hflip ablation

---

## Citation

Wang X. et al., *ChestX-ray8: Hospital-scale Chest X-ray Database and
Benchmarks on Weakly-Supervised Classification and Localization of Common
Thorax Diseases*, CVPR 2017.
Pérez-García F. et al., *RAD-DINO: Exploring Scalable Medical Image Encoders
Beyond Text Supervision*, 2024. Selvaraju R. R. et al., *Grad-CAM*, ICCV 2017.
Jiang P.-T. et al., *LayerCAM*, IEEE TIP 2021.
