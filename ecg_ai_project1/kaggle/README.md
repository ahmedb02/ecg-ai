# Running this project on Kaggle

`train_ptbxl_on_kaggle.py` trains the spatial CNN and temporal model from
`ecg_ai` on the real [PTB-XL](https://physionet.org/content/ptb-xl/)
12-lead ECG dataset, using a Kaggle-friendly bootstrap so it runs without
`pip install -e .` or any setup beyond attaching two Kaggle Datasets.

## Setup (one time, per notebook)

1. **Create a Kaggle Notebook.**
2. **Add Data -> this repository.** Kaggle's "Add Data" dialog has a GitHub
   option — point it at this repo (or upload the repo as a zipped Dataset
   if you'd rather not link GitHub). Either way it lands under
   `/kaggle/input/<something>/`, and the script finds it automatically.
3. **Add Data -> a PTB-XL dataset.** Search "ptb-xl" in Kaggle Datasets and
   attach one of the public mirrors of the PhysioNet PTB-XL release (it
   must keep the standard layout: `ptbxl_database.csv`, `scp_statements.csv`,
   `records100/`, `records500/`). This also lands under `/kaggle/input/`.
4. **Notebook settings:** turn on Internet (needed to `pip install wfdb`,
   which isn't preinstalled on Kaggle) and, ideally, a GPU accelerator.
5. Copy `train_ptbxl_on_kaggle.py` into the notebook (as a single code cell,
   or split at the `# %%` markers into one cell per section) and run it.

## What it does

1. Locates the `ecg_ai` package and the PTB-XL root under `/kaggle/input/`.
2. Installs `pandas`/`wfdb` if missing.
3. Trains `SpatialECGClassifier` on PTB-XL's official folds 1-8, validating
   on fold 9, and saves `spatial.pt`.
4. Groups PTB-XL records by `patient_id` into serial sequences (patients
   with >= 2 recordings), derives a weak evolution label per sequence (see
   `ecg_ai/data/ptbxl.py::infer_evolution_label`), and fine-tunes
   `TemporalECGModel` (initialized from the just-trained spatial encoder),
   saving `temporal.pt`.
5. Runs both models through `ECGInterpreter` on one validation example and
   prints the resulting single-ECG report and serial-evolution report, as a
   sanity check that training actually produced something coherent.

Checkpoints are written to `/kaggle/working/` (or the current directory if
`/kaggle/working` doesn't exist, e.g. when testing locally), which Kaggle
persists as notebook output you can download or use in a follow-up notebook.

## Configuration

Edit the constants in the "Configuration" section of the script, or set
environment variables before running (useful for a scheduled/unattended
Kaggle notebook run): `PTBXL_ROOT`, `SPATIAL_EPOCHS`, `TEMPORAL_EPOCHS`,
`BATCH_SIZE`, `NUM_WORKERS`.

## Caveats specific to training on PTB-XL

- The `scp_codes` -> finding mapping (`SCP_CODE_TO_FINDING` in
  `ecg_ai/data/ptbxl.py`) is a best-effort clinical approximation, not an
  exhaustive or clinically validated coding of every PTB-XL statement.
  Extend it if you need finer coverage.
- PTB-XL was not collected with serial-evolution comparison in mind, so the
  temporal model's training labels are *derived* (via
  `infer_evolution_label`) rather than clinically annotated. Relatively few
  PTB-XL patients have multiple recordings, so the temporal training set
  will be much smaller than the spatial one — this stage is best understood
  as validating the training pipeline on real waveforms, not as producing
  a clinically meaningful evolution detector on its own.
- As with everything in this repo: output carries a "not a validated
  diagnostic device" disclaimer for a reason. Treat a model trained by this
  script as a research baseline to iterate on, not a finished product.
