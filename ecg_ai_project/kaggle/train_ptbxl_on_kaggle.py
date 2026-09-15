"""Kaggle-ready training script: trains the spatial CNN + temporal model on
the real PTB-XL 12-lead ECG dataset.

How to run this on Kaggle (see kaggle/README.md for details):

1. Create a Kaggle Notebook.
2. Add Data:
   - This repository (Add Data -> GitHub -> this repo's URL), so the
     `ecg_ai` package is available under /kaggle/input/.
   - A PTB-XL Kaggle Dataset (search "ptb-xl" in Kaggle Datasets), so its
     files are available under /kaggle/input/.
3. Turn on internet + (ideally) a GPU accelerator in notebook settings.
4. Either upload this file as-is and run it as a script, or copy the
   sections marked "# %%" into separate notebook cells.
5. Run. Checkpoints are written to /kaggle/working/, which Kaggle persists
   as notebook output.

This script trains on real data but with a heuristic label-derivation
pipeline (see ecg_ai/data/ptbxl.py) -- read that module's docstrings before
treating its output as more than a demonstration of the training pipeline
on real waveforms. Every report the resulting model produces still carries
the "not a validated diagnostic device" disclaimer baked into ecg_ai's
report classes.
"""

# %% Bootstrap: make the ecg_ai package importable from a Kaggle notebook.
import glob
import importlib
import os
import subprocess
import sys


def _find_and_add_ecg_ai_to_path() -> None:
    try:
        import ecg_ai  # noqa: F401
        return
    except ImportError:
        pass

    candidates = glob.glob("/kaggle/input/**/ecg_ai/__init__.py", recursive=True)
    for init_file in candidates:
        repo_root = os.path.dirname(os.path.dirname(init_file))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        try:
            import ecg_ai  # noqa: F401
            print(f"[bootstrap] found ecg_ai package at {repo_root}")
            return
        except ImportError:
            continue

    raise ImportError(
        "Could not locate the ecg_ai package. In the Kaggle notebook, "
        "either (a) Add Data -> GitHub and attach this repository so it "
        "lands under /kaggle/input/, or (b) run `!git clone <repo-url>` "
        "into the working directory before this script, or (c) add this "
        "repo as a Kaggle 'Utility script'."
    )


_find_and_add_ecg_ai_to_path()

for _pkg in ["pandas", "wfdb"]:
    try:
        importlib.import_module(_pkg)
    except ImportError:
        print(f"[bootstrap] installing missing dependency: {_pkg}")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", _pkg], check=True)


# %% Imports (after bootstrap, so ecg_ai + its optional deps are available).
import torch

from ecg_ai.data.ptbxl import PTBXLSerialDataset, PTBXLSpatialDataset, find_ptbxl_root
from ecg_ai.inference.predict import ECGInterpreter
from ecg_ai.training import train_spatial, train_temporal


# %% Configuration -- edit these for your run, or override via env vars so
# the same script works unattended (e.g. a scheduled Kaggle notebook run).
PTBXL_ROOT = os.environ.get("PTBXL_ROOT") or find_ptbxl_root()
USE_HIGH_RES = True  # 500 Hz records; set False to use the 100 Hz records
TRAIN_FOLDS = list(range(1, 9))  # PTB-XL convention: folds 1-8 train
VAL_FOLDS = [9]                  # fold 9 validation, fold 10 held out as test
SPATIAL_EPOCHS = int(os.environ.get("SPATIAL_EPOCHS", 10))
TEMPORAL_EPOCHS = int(os.environ.get("TEMPORAL_EPOCHS", 10))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 32))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 2))
OUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."

print(f"PTB-XL root: {PTBXL_ROOT}")
print(f"device: {'cuda' if torch.cuda.is_available() else 'cpu'}")


# %% Stage 1: train the spatial classifier on single PTB-XL ECGs.
spatial_train_ds = PTBXLSpatialDataset(root=PTBXL_ROOT, folds=TRAIN_FOLDS, use_high_res=USE_HIGH_RES)
spatial_val_ds = PTBXLSpatialDataset(root=PTBXL_ROOT, folds=VAL_FOLDS, use_high_res=USE_HIGH_RES)
print(f"spatial train/val records: {len(spatial_train_ds)} / {len(spatial_val_ds)}")

spatial_checkpoint = os.path.join(OUT_DIR, "spatial.pt")
train_spatial.train(
    epochs=SPATIAL_EPOCHS,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    train_dataset=spatial_train_ds,
    val_dataset=spatial_val_ds,
    out_path=spatial_checkpoint,
)


# %% Stage 2: train the temporal (serial-ECG evolution) model, initialized
# from the spatial encoder just trained above.
temporal_train_ds = PTBXLSerialDataset(root=PTBXL_ROOT, folds=TRAIN_FOLDS, use_high_res=USE_HIGH_RES)
temporal_val_ds = PTBXLSerialDataset(root=PTBXL_ROOT, folds=VAL_FOLDS, use_high_res=USE_HIGH_RES)
print(f"temporal train/val patient sequences: {len(temporal_train_ds)} / {len(temporal_val_ds)}")

temporal_checkpoint = os.path.join(OUT_DIR, "temporal.pt")
if len(temporal_train_ds) == 0:
    print("No patients with >=2 PTB-XL records found in the training folds -- "
          "skipping temporal training. (PTB-XL has relatively few repeat "
          "patients; this is expected on small fold subsets.)")
else:
    train_temporal.train(
        epochs=TEMPORAL_EPOCHS,
        batch_size=min(BATCH_SIZE, 8),  # serial sequences are much larger per-item
        num_workers=NUM_WORKERS,
        train_dataset=temporal_train_ds,
        val_dataset=temporal_val_ds,
        spatial_checkpoint=spatial_checkpoint,
        out_path=temporal_checkpoint,
    )


# %% Sanity check: run the trained model(s) through ECGInterpreter on one
# validation example and print the resulting report(s).
has_temporal = os.path.exists(temporal_checkpoint)
interpreter = ECGInterpreter.from_checkpoints(
    spatial_checkpoint=spatial_checkpoint,
    temporal_checkpoint=temporal_checkpoint if has_temporal else None,
)

sample_ecg, _, _ = spatial_val_ds[0]
report = interpreter.interpret(sample_ecg.numpy(), fs=500.0 if USE_HIGH_RES else 100.0)
print("\n" + "=" * 70)
print("Sample single-ECG report (validation set)")
print("=" * 70)
print(report.to_text())

if has_temporal and len(temporal_val_ds) > 0:
    seq = temporal_val_ds.sequences[0]
    studies = [
        (temporal_val_ds[0][0][i].numpy(), 500.0 if USE_HIGH_RES else 100.0,
         temporal_val_ds[0][1][i].item())
        for i in range(temporal_val_ds[0][0].shape[0])
    ]
    latest_report, evolution_report = interpreter.interpret_serial(studies)
    print("\n" + "=" * 70)
    print("Sample serial-ECG evolution report (validation set)")
    print("=" * 70)
    print(evolution_report.to_text())

print(f"\nCheckpoints saved to: {OUT_DIR}")
