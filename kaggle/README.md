# Training on the full NIH dataset — on Kaggle, not on your laptop

The full ChestX-ray14 set is 45 GB. Kaggle already hosts it, so we train there
and bring back only a 30 MB checkpoint plus the test-set predictions.

## One-time setup (5 minutes)

1. Sign in to kaggle.com → **Create → New Notebook**.
2. Right panel → **Add Input** → **Datasets** tab (not Notebooks) → search `nih-chest-xrays/data` → Add.
   It mounts at `/kaggle/input/datasets/nih-chest-xrays/data/` (older notebooks: `/kaggle/input/data/`);
   the script finds either (112,120 PNGs in `images_001` … `images_012`,
   plus `Data_Entry_2017.csv`, `train_val_list.txt`, `test_list.txt`).
3. Right panel → **Settings**:
   - Accelerator: **GPU T4 x2**. Do NOT pick P100: Kaggle's current PyTorch build
     dropped Pascal GPUs (sm_60), so the P100 is detected but every CUDA op fails.
     The script trains in mixed precision, which the T4's tensor cores accelerate.
   - Persistence: **Files only** (keeps `/kaggle/working` if you need a second session)
   - Internet: **On** (torchvision needs to fetch the ImageNet weights once)
4. Phone verification is required for GPU access if you have not done it before.

## Run

Upload `train_full_nih.py` via **File → Upload**, or paste its contents into a
cell prefixed with `%%writefile train_full_nih.py`. Then in a cell:

```
!python train_full_nih.py --img 224 --epochs 8 --max-hours 10.5
```

What to expect:

| Phase | Time (one T4, AMP) |
|---|---|
| Index images + build 256 px cache in `/tmp` | ~15 min |
| Epoch 1 (head only) | ~10 min |
| Epochs 2–8 (full fine-tune) | ~15–20 min each |
| Test inference with TTA (25,596 films) | ~5 min |
| **Total** | **~2.5–3 h** |

The script stops itself before the `--max-hours` budget and writes the best
checkpoint after every epoch, so a killed session still leaves usable outputs.

Optional: `--img 320` matches the local run's resolution. Roughly 2× slower per
epoch; use `--epochs 6` to stay inside one session, or run two sessions with
`--resume` (add the first session's output as an input dataset first).

## Bring the results home

From the notebook's **Output** tab download these four files:

```
densenet121_full224.pt      best checkpoint (~30 MB)
full_test_preds.npz         predictions on the official 25,596-image test set
history.csv
summary.json
```

Put them in one folder, then locally:

```
python scripts/import_kaggle_run.py --dir path/to/downloaded/folder
```

That registers the model alongside the sample-trained ones and builds the
paired comparison described below.

## How the comparison stays fair

The two models were trained on different data and evaluated on different
splits, so their headline numbers are not directly comparable. The import
script therefore scores every model on the **same 1,293 images**: the films
that are both in our 5,606-image sample *and* in NIH's official test list.

- The sample-trained models score them out-of-fold (each image predicted by
  the one fold that never trained on it).
- The full-data model never trained on them at all (they are in `test_list.txt`).

That set has 685 patients and is enriched for abnormal films (61% vs 46%
overall), so absolute numbers will differ from the 5-fold figures — but the
comparison between models on it is clean. Per-class AUROC is reported only for
classes with at least 10 positives in the set; Hernia has 1 and is skipped.

## Sanity check before a long run

```
!python train_full_nih.py --limit 2000 --epochs 2 --max-hours 0.5
```

Finishes in ~25 minutes (most of it cache building) and exercises every code
path, including the test-set write-out.
