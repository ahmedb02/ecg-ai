# ecg_ai

A CNN-based pipeline for 12-lead ECG interpretation: given a single ECG it
produces a set of findings and a ranked differential diagnosis (DDx); given
a *series* of ECGs from the same patient over time, it additionally detects
and explains dynamic changes / evolution (e.g. an evolving STEMI, a new
conduction block, resolving ischemia). A web app (`backend/` + `frontend/`)
lets you upload a **photo or scan of a printed ECG** and get the same
findings/DDx/evolution report in the browser — see
[Web app: upload an ECG image](#web-app-upload-an-ecg-image).

> **Disclaimer:** This is a research/educational scaffold, not a validated
> medical device. It ships with **no real training data** and **no trained
> weights** — see [Using real data](#using-real-data) before relying on it
> for anything beyond exploring the architecture. Every generated report
> carries this same disclaimer and must be independently reviewed by a
> qualified clinician before informing any real decision.

## How it's structured

```
ecg_ai/
  clinical/       finding taxonomy, rule-based DDx knowledge base,
                   report + evolution narrative generation
  models/         the CNN architectures (spatial + temporal)
  data/           preprocessing, synthetic ECG generator, PyTorch datasets
  imaging/        digitizes a photo/scan of a printed ECG into a signal array
  training/       training loops (synthetic data by default)
  inference/      ECGInterpreter: raw signal(s) -> clinical report
backend/          FastAPI app: image upload -> digitize -> interpret -> JSON
frontend/         plain HTML/CSS/JS single-page app (no build step)
examples/demo.py  runnable end-to-end walkthrough (signal-based)
tests/            unit tests covering models, data, imaging, API, and the full pipeline
```

### 1. Spatial CNN (single ECG -> findings + DDx)

`ecg_ai/models/spatial_cnn.py`

A 12-lead ECG isn't 12 interchangeable channels — each lead is a different
electrical "view" of the heart (inferior, lateral, anteroseptal, ...). The
spatial encoder respects that:

1. **Per-lead 1D CNN** (shared weights, residual + SE blocks) extracts a
   temporal feature embedding independently for each of the 12 leads.
2. **Lead self-attention** lets leads inform each other — e.g. relating
   inferior ST elevation to reciprocal lateral ST depression — producing a
   `(12, D)` spatial feature map, analogous to a spatial map in an image CNN
   where "lead" plays the role of spatial position.
3. **Anatomical region pooling** groups leads into clinical regions
   (inferior / lateral / anteroseptal / septal / anterior) for localized
   findings.
4. Two classification heads: a **global head** (whole-ECG findings — rhythm,
   conduction, hypertrophy, etc.) and a **region head** (per-region
   ischemia/infarct localization), both multi-label.

`ecg_ai/clinical/ddx.py` then turns thresholded findings into a ranked
differential diagnosis using a small, transparent, auditable knowledge base
(e.g. "STEMI pattern, inferior leads" → *acute inferior STEMI (RCA/LCx
occlusion)*, *pericarditis (mimic)*, ...). DDx generation is rule-based
rather than learned deliberately — the CNN does the perceptual task (is this
pattern present, and where); the mapping from an established ECG pattern to
its differential is well-codified medical knowledge, and keeping it as
explicit rules keeps every suggestion traceable to a stated rationale
instead of an opaque score.

### 2. Temporal model (serial ECGs -> evolution)

`ecg_ai/models/temporal.py`

Takes a time-ordered sequence of ECGs from one patient (irregularly spaced —
ED arrival, +1hr, +3hr, next day, ...) and:

1. Encodes each ECG with the **same spatial encoder** (shared weights, so
   the model reuses one set of waveform-morphology detectors instead of
   learning them again for the temporal task).
2. Feeds the resulting sequence of per-study embeddings through a
   **Transformer with a continuous sinusoidal time encoding** (so it's aware
   of the actual elapsed hours between studies, not just step order).
3. Predicts an **evolution label** at the latest study relative to the
   earlier ones: new/evolving STEMI, resolving/worsening ischemia, new or
   resolved conduction abnormality, new or resolved arrhythmia, or the
   classic infarct-evolution trajectory (hyperacute T → ST elevation → Q
   waves).
4. Computes an interpretable **per-region embedding delta** between the
   earliest and latest study.

`ecg_ai/clinical/evolution.py` combines that learned label with a
transparent set-difference of the discrete findings at baseline vs. latest
(new / resolved / persistent) into a plain-language narrative, so a
clinician can see *why* the model called an evolution, not just the label.

### 3. Putting it together

`ecg_ai/inference/predict.py::ECGInterpreter` is the entry point:

```python
from ecg_ai.inference.predict import ECGInterpreter

interpreter = ECGInterpreter.from_checkpoints(
    spatial_checkpoint="checkpoints/spatial.pt",
    temporal_checkpoint="checkpoints/temporal.pt",  # optional, needed for interpret_serial
)

# Single ECG -> findings + DDx
report = interpreter.interpret(ecg, fs=500)          # ecg: (12, num_samples) numpy array
print(report.to_text())

# Serial ECGs -> latest interpretation + evolution report
studies = [(ecg_t0, 500.0, 0.0), (ecg_t1, 500.0, 2.0), (ecg_t2, 500.0, 72.0)]  # (signal, fs, hours)
latest_report, evolution_report = interpreter.interpret_serial(studies)
print(evolution_report.to_text())
```

Run `python -m examples.demo` for a full runnable walkthrough (with random,
untrained weights — see below).

## Quickstart

```bash
pip install -r requirements.txt
pytest -q                     # model shapes, DDx rules, imaging, API, full pipeline
python -m examples.demo       # end-to-end demo on synthetic ECGs
```

## Web app: upload an ECG image

`backend/` (FastAPI) + `frontend/` (plain HTML/CSS/JS, no build step) turn
the pipeline into a small web app: upload a photo or scan of a printed
12-lead ECG, optionally drag a crop box over just the lead-panel grid, and
get the findings/DDx report in the browser — plus, on the "Compare serial
ECGs" tab, upload several ECGs with their date/time to get the evolution
report.

```bash
pip install -r requirements.txt
uvicorn backend.app:app --reload --port 8000
# open http://localhost:8000
```

Without `SPATIAL_CHECKPOINT`/`TEMPORAL_CHECKPOINT` env vars pointing at
trained weights (see Training/Kaggle above), it still runs — on random
weights, clearly flagged as "DEMO MODE" in the UI and every API response
— so you can try the whole upload → digitize → interpret flow immediately.
Click "Try a synthetic example" on the Single ECG tab if you don't have a
real ECG image handy.

```bash
SPATIAL_CHECKPOINT=checkpoints/spatial.pt \
TEMPORAL_CHECKPOINT=checkpoints/temporal.pt \
uvicorn backend.app:app --port 8000
```

A `Dockerfile` is included for deployment (`docker build -t ecg_ai . &&
docker run -p 8000:8000 ecg_ai`; mount/copy checkpoints in and set the same
env vars to serve trained weights).

### How image digitization works

`ecg_ai/imaging/digitize.py` turns a photo/scan into a `(12, T)` signal
array the CNN can consume:

1. Isolate ink pixels (the trace) from the paper grid and background using
   color + local-contrast thresholding.
2. **Self-calibrate the paper's mm grid spacing directly from the image**
   (via the dominant periodicity of grid-line-colored pixels) — this works
   regardless of the photo's actual resolution/DPI, since it never needs to
   know that.
3. Split into the standard 3-row × 4-column lead panel layout, trace each
   panel's ink column-by-column, and convert pixel position to
   (time, amplitude) using that calibration.

Every response includes a `digitization` block with the calibration method
used, any warnings, and an **overlay image** (extracted trace drawn back
onto your photo) — check that overlay before trusting a result, since this
is a best-effort classical (non-learned) digitizer, not a validated tool.
Two things meaningfully affect quality: cropping out headers/margins
before analyzing (drag the crop box in the UI), and image contrast/focus.

One inherent limitation, regardless of image quality: a standard printed
ECG shows each lead over a *different* few seconds of real time (one
column of panels per short interval), not all 12 leads simultaneously like
a true digital acquisition (or this repo's synthetic generator) — so
digitized leads aren't perfectly time-synchronized with each other. Most
findings here are per-lead morphology + region grouping, which this
doesn't much affect; anything relying on precise cross-lead timing is
weaker on digitized-from-print input.

## Training

No real ECG dataset is bundled with this repo, so the training scripts
default to an **on-the-fly synthetic ECG generator**
(`ecg_ai/data/synthetic.py`) — it's a parametric waveform model (Gaussian
P/QRS/T components with rough per-lead morphology scaling) that can produce
structurally plausible 12-lead ECGs tagged with conditions like `stemi`,
`afib`, `lbbb`, `lvh`, etc., and serial sequences tagged with an evolution
scenario. This is enough to exercise every shape and code path (train, save
a checkpoint, run inference, generate a report) but **is not real training
data** — treat models trained on it purely as a plumbing check, not as
something with real diagnostic signal.

```bash
# 1. Train the spatial classifier
python -m ecg_ai.training.train_spatial --epochs 20 --out checkpoints/spatial.pt

# 2. Train the temporal model, initialized from the spatial encoder's weights
python -m ecg_ai.training.train_temporal \
    --epochs 20 --spatial-checkpoint checkpoints/spatial.pt --out checkpoints/temporal.pt
```

## Using real data

**PTB-XL is wired up already** — `ecg_ai/data/ptbxl.py` provides
`PTBXLSpatialDataset` and `PTBXLSerialDataset` (real PTB-XL waveforms +
diagnosis codes mapped onto this repo's finding taxonomy), and
`kaggle/train_ptbxl_on_kaggle.py` is a ready-to-run Kaggle notebook script
that trains both models on it end to end — see `kaggle/README.md`. That's
the fastest path to training on real 12-lead ECGs without writing a new
dataset loader.

To train on a different real dataset (MIMIC-IV-ECG, a private clinical
dataset, ...), implement a `torch.utils.data.Dataset`
that yields the same tensors the synthetic datasets do, and pass it to the
training scripts in place of `SyntheticSpatialDataset` / `SyntheticTemporalDataset`:

- **Spatial**: `(ecg: FloatTensor[12, T], global_vec: FloatTensor[NUM_FINDINGS],
  region_vec: FloatTensor[NUM_REGIONS, num_region_findings])`. Use
  `ecg_ai.data.dataset.labels_to_vectors(global_finding_keys, [(finding_key, region), ...])`
  to build the label tensors from your dataset's diagnosis codes, mapped onto
  the taxonomy in `ecg_ai/clinical/labels.py` (extend that taxonomy as needed
  — e.g. add SNOMED-CT-mapped PTB-XL codes).
- **Temporal**: a sequence of raw `(ecg, fs, timestamp)` per patient, collated
  with `ecg_ai.data.dataset.collate_temporal_batch`; you'll need to derive
  ground-truth evolution labels from serial reads (e.g. two priors read as
  "normal" then "STEMI" → `new_or_evolving_stemi`).
- Run every raw ECG through `ecg_ai.data.preprocessing.preprocess_ecg` (or
  your own equivalent) so lead order, sampling rate, and signal length match
  `ecg_ai/config.py::DEFAULT_SIGNAL_CONFIG` (500 Hz, 10 s, lead order in
  `ecg_ai/clinical/labels.py::LEAD_NAMES`).

`ecg_ai/data/ptbxl.py` needs `pandas` and `wfdb`, which aren't in the base
`requirements.txt` to keep the core package lightweight — install them with
`pip install -r requirements-kaggle.txt` (or just `pandas wfdb`) to use it
locally.

## Finding taxonomy

Defined in `ecg_ai/clinical/labels.py`: rhythm (NSR, AFib, VT/VFib, ...),
conduction (LBBB/RBBB, AV blocks, WPW), ischemia/infarction — regionalized
by anatomical territory (inferior/lateral/anteroseptal/septal/anterior),
hypertrophy/enlargement, and other patterns (pericarditis, long QT, Brugada,
hyperkalemia, low voltage, paced rhythm). Extending it means adding a
`Finding` entry, a DDx knowledge-base entry in `ecg_ai/clinical/ddx.py`, and
(if you're training on synthetic data) a scenario in
`ecg_ai/data/synthetic.py::make_case`.
