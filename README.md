# Attention as Evidence

*Coupling Grad-CAM Localisation to Vision-Language Report Generation for Chest Radiographs*

A chest X-ray goes in. A DenseNet-121 scores it for 14 thoracic findings, a
LayerCAM heatmap shows **where** the model looked for each flagged finding, the
heatmap is measured against radiologists' boxes, and MedGemma 4B drafts a
report that is **grounded in that evidence**. A three-page Streamlit app wraps
it all, with JSON / PDF export.

Trained model weights are in this repository. **You can clone it and run the
app in about fifteen minutes without downloading any dataset.**

> Research and educational use only. Not a medical device and not for clinical
> diagnosis.

---

## Contents

1. [Getting started (teammates, read this first)](#1-getting-started)
2. [What is in the repository](#2-what-is-in-the-repository)
3. [Data: getting the NIH sample for retraining](#3-data)
4. [Running the full pipeline](#4-running-the-full-pipeline)
5. [Troubleshooting](#5-troubleshooting)
6. [Key results](#6-key-results)
7. [Method](#7-method)
8. [Results in detail](#8-results-in-detail)
9. [Phase 4: the MedGemma grounding study](#9-phase-4-medgemma-report-generation)
10. [The application](#10-the-application)
11. [Status and citation](#11-status)

---

## 1. Getting started

### What you need

| | Required | Notes |
|---|---|---|
| OS | Windows 10/11, Linux or macOS | Developed on Windows 11; commands below are given for PowerShell and bash |
| Python | **3.11** (3.10–3.12 should work) | `python --version`. Get it from python.org; tick "Add to PATH" |
| Git | any recent version | git-scm.com |
| Disk | ~1.5 GB | clone ≈ 340 MB, Python packages ≈ 1 GB (PyTorch is large) |
| GPU | optional | NVIDIA GPU with ≥ 4 GB makes analysis ~1 s per film. CPU works, ~5 s per film |
| Ollama | optional | only for the "Generate draft report" button (MedGemma). See step 5 |
| NIH images | optional | only for retraining or re-scoring. See section 3 |

### Step 1 · Clone

```bash
git clone https://github.com/trinity1611/Attention-as-Evidence.git
cd Attention-as-Evidence
```

The clone is ~340 MB because it contains the trained weights
(`checkpoints/`, 27 MB per model) and every result table and figure. If the
clone stalls on a slow connection, run it again; git resumes.

### Step 2 · Create the environment and install

**Windows (PowerShell)**

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
# If PowerShell refuses to run the activate script:
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned    (once), then retry
python -m pip install --upgrade pip
```

**macOS / Linux (bash)**

```bash
python3.11 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
```

Then install PyTorch for **your** machine, and everything else:

```bash
# Option A: NVIDIA GPU (driver supporting CUDA 12.x; check with `nvidia-smi`)
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Option B: no NVIDIA GPU (Intel/AMD laptop, Mac)
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu

# then, either way
pip install -r requirements.txt
```

Every command in this README assumes the venv is activated (your prompt shows
`(venv)`) and you are in the repository root.

### Step 3 · Check the setup

```bash
python scripts/check_setup.py
```

It prints one line per check. `OK` and `WARN` are fine (WARN = optional parts
such as the dataset or Ollama). Fix anything marked `FAIL` before continuing;
the message says what to do. A healthy fresh clone ends with:

```
[OK  ] deployed classifier -- checkpoints\densenet121_full224.pt (28.5 MB)
[OK  ] result tables the app reads -- 9 files present
[WARN] NIH sample images -- not found at C:\ml-data\nih-cxr\sample\images; app works without them ...
[WARN] Ollama not reachable on :11434 -- optional; needed only for draft reports ...

Ready: streamlit run app/Home.py
```

### Step 4 · Run the app

```bash
streamlit run app/Home.py
```

A browser tab opens at http://localhost:8501 (if not, open it yourself). The
first page load takes ~20 s while the classifier and explanation engine load;
after that it is cached.

- **Home**: the animated hero plus an overview of the whole project, with live numbers.
- **Analyse**: pick one of the 11 demo films shipped in `data/demo_films`
  (all from NIH's official test split, never trained on; several have a
  radiologist box so you see hit / miss / IoU), or upload any frontal chest
  X-ray PNG/JPG. You get the 14 scores with thresholds, per-finding coloured
  heatmaps, attention zones, and JSON / PDF / PNG downloads.
- **Dashboard**: the deployed model's final metrics, the adopted heatmap
  method's results, and how the heatmaps feed MedGemma.

Stop the server with `Ctrl+C` in the terminal.

### Step 5 · (Optional) MedGemma draft reports through Ollama

The "Generate draft report" button on the Analyse page needs a local Ollama
server with the MedGemma model. Everything else works without it.

1. Install Ollama from https://ollama.com/download (Windows, macOS, Linux).
2. Pull the model (~3.3 GB download, once):
   ```bash
   ollama pull medgemma
   ```
   The code expects the model name `medgemma:latest`, which is what that pull
   creates. `ollama list` should show it.
3. Make sure the server is running (`ollama serve` in a separate terminal; on
   Windows the desktop app runs it in the background automatically).
4. Re-run `python scripts/check_setup.py`; the Ollama line should say `OK`.

A report takes ~15 s on an RTX 4060 (8 GB) and several minutes on CPU. The
app calls `http://localhost:11434/api/chat`; change `OLLAMA_URL` / `MODEL` at
the top of `src/medgemma.py` if your setup differs.

### Step 6 · (Optional) Point the project at your data

Only needed if you want to retrain, re-score, or browse all 5,606 sample
films in the Analyse page. See section 3 for the download. Paths live in
`config.yaml` under `data:`. Rather than editing the tracked file, make a
personal copy and point the code at it:

```powershell
Copy-Item config.yaml config.local.yaml          # edit the four paths under data:
$env:CXR_CONFIG = "config.local.yaml"            # PowerShell, current session
```
```bash
cp config.yaml config.local.yaml
export CXR_CONFIG=config.local.yaml              # bash
```

`config.local.yaml` is gitignored, so your paths never end up in a commit.

### Where to read next

1. `Project_Briefing.docx`: the full report (abstract, data, models, every
   result and figure, metric definitions with formulas, worked examples).
2. `src/service.py`: the one function the app calls, `Service.analyse()`. Read
   it top to bottom and you know what the system does.
3. `app/pages/1_Analyse.py`: how the results are shown.
4. `src/train.py`, `src/gradcam.py`, `src/medgemma.py`, `src/report_study.py`:
   the three phases, in order.

---

## 2. What is in the repository

```
config.yaml                  all paths and hyperparameters (copy to config.local.yaml for your machine)
requirements.txt             pinned Python packages (PyTorch installed separately, see step 2)
Project_Briefing.docx        the full report (regenerate with scripts/make_report_doc.py)

checkpoints/                 TRAINED WEIGHTS, tracked in git
  densenet121_full224.pt         the deployed classifier (77,988 NIH films, Kaggle T4, AUROC 0.812)
  densenet121_imagenet_fold*.pt  5 folds of the local 320 px sample-trained model (comparative analysis)
  raddino_linear_fold*.pt        5 folds of the RAD-DINO linear probe (tiny)

data/                        small NIH metadata + demo films (CC0); the images themselves are NOT here
  sample_labels.csv              labels for the 5,606-film sample
  meta/BBox_List_2017.csv        984 radiologist boxes
  meta/train_val_list.txt        NIH official split lists
  meta/test_list.txt
  demo_films/                    11 held-out films so the Analyse page has something to show

src/
  config.py                  class list, config loader (CXR_CONFIG), whitespace-tolerant CSV reader
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
  pages/2_Dashboard.py       deployed-model metrics, adopted XAI results, heatmaps -> MedGemma
  _common.py                 shared CSS, cached service, widgets

scripts/
  check_setup.py             verifies a fresh clone: packages, weights, tables, data, Ollama
  make_report_doc.py         builds Project_Briefing.docx
  briefing_sections*.py, briefing_appendix.py, make_diagrams*.py, make_formulas.py,
  make_gallery_figures.py    report content, diagrams, formula images, contact sheets
  make_visuals.py            per-class exemplar images
  make_hero_assets.py        pre-renders the Home-page hero from real pipeline output
  test_predictions.py        inspect held-out predictions on sampled films
  import_kaggle_run.py       register the full-data run; shared-test comparison

kaggle/
  train_full_nih.py          self-contained full-dataset trainer for a Kaggle GPU
  train_full_nih.ipynb       the same as an uploadable notebook
  README.md                  step-by-step: attach dataset, run, bring results home

outputs/                     every result, tracked in git (the app reads from here)
  folds.csv                  image -> patient, fold, 14 labels
  sample_images/             3 exemplars per finding + overview grid
  distribution/              class counts, label cardinality, co-occurrence, fold balance
  predictions/               per-fold dumps, full-test dump, prediction grids
  plots/                     every figure; gradcam/{bbox,cases,sheets}/<config>/<model>/
  metrics/                   every table (CSV / JSON); res224_study/ archives the 224 px runs
  reports/                   the 600 MedGemma reports (reports.jsonl) + examples
  hero/                      pre-rendered assets for the Home-page animation
```

---

## 3. Data

The project uses the **official 5 % sample of NIH ChestX-ray14** (Wang et al.,
CVPR 2017; CC0): 5,606 films from 4,230 patients, 1024 × 1024 PNG, labels
NLP-mined from reports (~90 % accurate). The full set (112,120 films, 45 GB)
was used only for the Kaggle training run and is never downloaded locally.

You need the images only to retrain, re-run the evaluation, or browse all
sample films in the app. Nothing in the repo is bigger than 27 MB, and the
images are 2.2 GB, so they stay outside git.

**Download.** Kaggle dataset `nih-chest-xrays/sample`
(https://www.kaggle.com/datasets/nih-chest-xrays/sample), ~2 GB zipped. You
need a free Kaggle account. Either use the "Download" button, or the Kaggle CLI:

```bash
pip install kaggle          # then place your kaggle.json token per Kaggle's instructions
kaggle datasets download -d nih-chest-xrays/sample -p C:/ml-data/nih-cxr --unzip
```

**Layout the code expects** (the default `config.yaml` root is
`C:/ml-data/nih-cxr`; change it in your `config.local.yaml`). After unzipping,
check where the PNGs landed and move them so the paths match:

```
<root>/
  sample/images/*.png        5,606 films
  sample_labels.csv          copy from this repo: data/sample_labels.csv
  meta/BBox_List_2017.csv    copy from this repo: data/meta/
  meta/train_val_list.txt
  meta/test_list.txt
  images_352/                created by src.prepare_data (resized training cache)
```

Keep the data **outside** any cloud-synced folder (OneDrive, Dropbox);
thousands of PNGs will saturate the sync client.

Then build the training cache and folds:

```bash
python -m src.prepare_data        # ~5 min; writes images_352/, outputs/folds.csv, distribution figures
python scripts/check_setup.py     # the "NIH sample images" line should now say OK
```

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
zero patients span folds. Every image is scored once, out-of-fold. The
full-data model uses NIH's official `train_val` / `test` lists (val split by
patient). The **1,293 films in both our sample and the official test list** are
the common ground for comparing all models fairly. The 53 radiologist boxes
that fall on our films are all in the official test split.

---

## 4. Running the full pipeline

Everything below is already done; its outputs are in `outputs/` and the
weights in `checkpoints/`. Run a step only if you want to change or verify it.
Times are for an RTX 4060 laptop GPU (8 GB).

```bash
# --- Phase 1: data (needs the NIH sample, section 3)
python -m src.prepare_data                     # 352 px cache, folds, distribution figures      ~5 min

# --- Phase 2: classifiers on the sample
python -m src.train                            # DenseNet-121 320 px, 5 folds                  ~25 min/fold
python -m src.infer                            # TTA predictions + prediction grids            ~3 min
python -m src.features && python -m src.probe  # RAD-DINO embeddings (once, ~15 min) + probe   ~1 min
python -m src.evaluate --suffix _tta           # AUROC tables, ROC, thresholds, paired stats
python -m src.confusion --suffix _tta          # confusion matrices, P / R / F1
python -m src.plots --suffix _tta              # loss / PR / calibration figures
python -m src.triage                           # binary triage

# --- Phase 2b: full-data model (Kaggle, free T4 x2; see kaggle/README.md)    ~1 GPU-hour
#     run kaggle/train_full_nih.ipynb on Kaggle, download its output folder, then
python scripts/import_kaggle_run.py --dir <downloaded outputs>
python -m src.final_metrics                    # deployed model: AUROC / AUPRC / accuracy / P / R / F1

# --- Phase 3: explanations
python -m src.gradcam                          # heatmaps + bbox metrics, adopted configuration  ~3 min
python -m src.gradcam_sweep                    # optional: the 111-configuration sweep          hours

# --- Phase 4: MedGemma grounding study (needs Ollama + medgemma)
python -m src.medgemma --smoke 3               # 3 films x 3 conditions, printed                ~2 min
python -m src.report_study --run --n 200       # 600 reports, resumable                         ~3 h
python -m src.report_study --score

# --- Report and app assets
python scripts/make_report_doc.py              # rebuild Project_Briefing.docx (close Word first)
python scripts/make_hero_assets.py             # re-render the landing-page hero
```

`src.train` refuses to overwrite an existing fold checkpoint without
`--overwrite`; the 320 px DenseNet is the baseline every later experiment is
paired against, so do not retrain it casually.

Always run modules as `python -m src.<name>` from the repository root so the
`src` package resolves.

---

## 5. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `streamlit: command not found` / `python` opens the Microsoft Store | The venv is not activated. Run `venv\Scripts\Activate.ps1` (PowerShell) or `source venv/bin/activate`. |
| `Activate.ps1 cannot be loaded because running scripts is disabled` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, once, then retry. |
| `ModuleNotFoundError: No module named 'src'` | You are not in the repository root, or ran `python src/x.py` instead of `python -m src.x`. |
| `torch.cuda.is_available()` is False but you have an NVIDIA GPU | You installed the CPU wheel. `pip uninstall torch torchvision`, then the cu124 line from step 2. Update the NVIDIA driver if `nvidia-smi` fails. |
| `check_setup` says the deployed classifier is missing or truncated | The clone was interrupted. `git status` then `git checkout -- checkpoints/`, or re-clone. |
| App: "Could not reach MedGemma through Ollama" | Ollama is not running or `medgemma` is not pulled. `ollama list`, then `ollama pull medgemma`; make sure `ollama serve` (or the desktop app) is up. |
| App: Analyse page shows only 11 demo films | Expected without the dataset. Section 3 explains how to get all 5,606. |
| First analysis is slow / page reloads twice | Normal: model load is cached after the first call (~20 s). CPU-only machines take ~5 s per film thereafter. |
| `PermissionError` when building the docx | Word has the file open. Close it; the builder otherwise saves `Project_Briefing_updated.docx`. |
| Windows: DataLoader workers crash or hang during training | Lower `train.num_workers` to 0 or 2 in your `config.local.yaml`. |
| `Port 8501 is already in use` | Another Streamlit is running. Close it, or `streamlit run app/Home.py --server.port 8502`. |
| Training out of memory | Lower `train.batch_size` (24 fits 320 px on 8 GB; use 12 on 4–6 GB). |
| `git clone` slow or fails on the ~340 MB repo | Retry; or `git clone --depth 1 ...` to skip history. |

---

## 6. Key results

**Classification** (macro AUROC over 14 findings; every model scored only on
images it never trained on)

| Model | Training data | 5-fold CV (5,606 films) | Same 1,293 held-out films | Δ vs sample CNN |
|---|---|--:|--:|--:|
| DenseNet-121, fine-tuned 320 px | 5,606 (sample) | 0.757 ± 0.022 | 0.720 | — |
| RAD-DINO frozen + linear layer | 5,606 (sample) | 0.755 ± 0.020 | 0.730 | +0.010 [−0.010, +0.030] |
| Ensemble of the two | 5,606 (sample) | **0.786 ± 0.022** (p = 0.001) | 0.756 | +0.036 [+0.024, +0.048] |
| **DenseNet-121, fine-tuned 224 px** | **77,988 (full NIH)** | 0.812 on official test (n = 25,596) | **0.807** | **+0.087 [+0.071, +0.104]** |

Data volume dominated every other lever tried (pretraining source,
architecture, resolution, schedule, foundation-model features, ensembling)
by more than 2×. The full-data model wins on all 13 scorable findings.

**Deployed model, final metrics** (`checkpoints/densenet121_full224.pt`; NIH
official test split; thresholds fitted on a random half of the test films,
threshold metrics reported on the other 12,798):

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
model scores 16/53: more data improved *where* the model looks, not only what
it predicts.

**Triage** (any finding vs none, 46/54 split, the one place accuracy means
something): ensemble AUROC 0.761, accuracy 0.71, sensitivity 0.71 / specificity 0.72.

**Report generation** (MedGemma 4B via Ollama; 200 films × 3 briefings = 600
reports): giving the model the classifier's probabilities alone made it assert
2.9× more findings than from the image and leave no normal film clean; adding
the Grad-CAM zone cut unsupported assertions by **1.07 per report (95 % CI
0.80–1.37)** and reduced false findings against the ground truth from 5.0 to
3.9. Grounding helps; no condition produces a draft fit to go out unread.

---

## 7. Method

- **Multi-label, not multi-class.** 14 independent sigmoids, weighted BCE with
  `pos_weight` capped at 20 (uncapped, Hernia's would be ~430), label smoothing
  0.05. Never softmax; never plain accuracy as the headline.
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
- **Thresholds.** Per finding, the score that maximises F1 on a fitting set
  (never the evaluation set); stored in `outputs/metrics/thresholds_*.csv` and
  read by the app.

---

## 8. Results in detail

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
+0.12, Cardiomegaly +0.12, Edema +0.11, Nodule +0.10), achieved at a *lower*
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
hardware: the errors are radiological confusions, not annotation shortcuts.

*Caveat:* 111 configurations ranked on 53 boxes is optimistic. Confirmation on
all 984 boxes (a 30-minute Kaggle job) is the next task.

---

## 9. Phase 4: MedGemma report generation

MedGemma 4B-it runs locally through Ollama (`medgemma:latest`, Q4_K_M, 100 %
GPU-resident on 8 GB). It is a *writer*, not a second classifier: given the
film and (depending on condition) the classifier's evidence, it returns a JSON
block (each of the 14 findings present / absent / uncertain, plus locations)
followed by a Findings / Impression / Recommendation draft. `src.medgemma`
builds the prompts and parses replies; `src.report_study` runs and scores the
experiment.

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
- Localisation agreement of 93 % in C is largely compliance (the model repeats
  the zone it was given) and is reported as such.

Files: `outputs/reports/reports.jsonl` (all 600), `outputs/reports/examples/`,
`outputs/metrics/report_study_*.csv`, `outputs/plots/report_study.png`.

---

## 10. The application

Streamlit, three pages, dark clinical-workstation theme (navy ground, cyan
accent, glowing film frames, monospace numerics), one fixed colour per finding
used everywhere (bars, heatmaps, contours, report table, legend).

- **Home**: an animated hero built from *real* pipeline output on four held-out
  films: the film breathes, a scan line sweeps, each flagged finding's LayerCAM
  blooms in its colour with its contour and a probability counting up, and the
  MedGemma impression types itself out. Below it: why the project exists, the
  pipeline, the numbers, the data, per-finding performance, the grounding
  study, how to use the app, limitations, and the stack. Respects
  `prefers-reduced-motion`.
- **Analyse**: upload a film, or pick a held-out NIH film (all 5,606 if the
  dataset is present, otherwise the 11 demo films in `data/demo_films`).
  Probability bars with per-finding thresholds and the uncalibrated-score
  warning; per-finding heatmaps (alpha fill + 50 %-of-peak contour + peak
  mark), overlay any subset, film tint and opacity controls; attention zone per
  finding; **hit / miss / IoU / coverage when the film has a radiologist box**
  (recognised by content hash), otherwise the per-finding peak-in-box rate from
  our 53-box evaluation, never an invented metric. "Generate draft report"
  calls MedGemma under the grounded condition and shows the prose, the
  per-finding call table and the report scores (agreement with classifier,
  unsupported / omitted findings, localisation agreement, and precision /
  recall against NIH labels when known). Downloads: **JSON** (everything),
  **PDF** (film, heatmaps, findings, report, scores, disclaimer), **PNG**.
- **Dashboard**: the deployed model's final metrics (AUROC, AUPRC, accuracy,
  precision, recall, F1, per finding), the adopted explanation method's
  results on the 53 boxes, and the heatmap-to-MedGemma metrics from the study.
  Everything is read live from `outputs/metrics` at page load.

All three pages pass Streamlit's `AppTest` harness end to end, with and
without the dataset present.

---

## 11. Status

- [x] Data pipeline, patient-grouped folds, distribution analysis
- [x] Classifier: 224 px study (null result), 320 px model, RAD-DINO probe, ensemble
- [x] Full-data model on Kaggle; fair shared-test comparison
- [x] Grad-CAM: configuration sweep, bbox-validated localisation
- [x] MedGemma report generation + grounding study (600 reports)
- [x] Streamlit interface (Home with animated hero, Analyse, Dashboard; JSON / PDF / PNG export)
- [x] Report (`Project_Briefing.docx`) with all figures
- [x] Weights, results and demo films in the repository; runs from a bare clone
- [ ] Confirm Grad-CAM configuration on all 984 boxes (Kaggle)
- [ ] Optional: full-data model at 320 px, hflip ablation

## Citation

Wang X. et al., *ChestX-ray8: Hospital-scale Chest X-ray Database and
Benchmarks on Weakly-Supervised Classification and Localization of Common
Thorax Diseases*, CVPR 2017.
Pérez-García F. et al., *RAD-DINO: Exploring Scalable Medical Image Encoders
Beyond Text Supervision*, 2024. Selvaraju R. R. et al., *Grad-CAM*, ICCV 2017.
Jiang P.-T. et al., *LayerCAM*, IEEE TIP 2021. Sellergren A. et al.,
*MedGemma Technical Report*, 2025.
