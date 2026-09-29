# Satellite Change Detection — India

Streamlit app that compares two real Sentinel-2 acquisitions of any location in
India and highlights where the ground changed, using a Siamese U-Net trained on
LEVIR-CD.

Nothing in the app is simulated. If Earth Engine is not connected, no trained
checkpoint exists, or no labelled test data has been evaluated, the app says so
instead of displaying a number.

## Install

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Google Earth Engine

Imagery comes from `COPERNICUS/S2_SR_HARMONIZED`, which requires an Earth Engine
account and a Cloud project registered for Earth Engine.

```bash
earthengine authenticate           # once per machine
export GEE_PROJECT_ID=your-project-id
```

A service account works too:

```bash
export GEE_SERVICE_ACCOUNT=ee-runner@your-project.iam.gserviceaccount.com
export GEE_PRIVATE_KEY_FILE=/path/to/key.json
```

The project id can also be typed into the sidebar at runtime.

## Run

```bash
python -m streamlit run app.py
```

Flow: **Location → Dates → Cloud threshold → Run detection → Before/After →
Change overlay → Metrics → Downloads.**

Location can be chosen four ways: state → district → city (all 36 states/UTs and
~750 districts are bundled in `resources/india_districts.json`, cities are looked
up live from OpenStreetMap and cached), free-text search, a click on the
interactive map, or manual latitude/longitude. The AOI is a 1–20 km radius.

For each date the app searches a window around it, scores every candidate scene
by cloud cover *inside the AOI* (Sentinel-2 SCL classes 3, 8, 9, 10) and reports
the real acquisition date and cloud percentage of the scene it used. Before and
after images are rendered from the same rectangle, CRS, band stretch and pixel
grid, so they are pixel aligned.

## Train

```bash
python download_data.py                                   # LEVIR-CD -> data/LEVIR-CD
python -m src.train --data data/LEVIR-CD --epochs 40
```

Training uses Dice + BCE with a `pos_weight` derived from the actual share of
changed pixels in the training labels (~4%), flip/rotate/date-swap augmentation
with independent photometric jitter per date, model selection on validation F1,
and a decision threshold tuned on the validation split. The tuned threshold is
stored in the checkpoint and is the default used by the app.

On a CPU-only box, reduce the cost: `--size 128 --train-limit 2400 --epochs 10`.

## Evaluate

```bash
python -m src.evaluate --data data/LEVIR-CD
```

Evaluates the held-out `test` split and writes `outputs/test_metrics.json`,
`outputs/confusion_matrix.png` and `outputs/qualitative_examples.png`:
accuracy, balanced accuracy, precision, recall, F1, IoU, Dice, specificity and
the pixel confusion matrix. Only ~4% of LEVIR-CD pixels are changed, so raw
pixel accuracy is high for any model — F1 and IoU are the meaningful numbers.

These metrics describe the model on LEVIR-CD (0.5 m aerial imagery). No labelled
ground truth exists for an arbitrary Sentinel-2 AOI, so the change percentage
reported for a live AOI cannot be scored, and the app states this.

## Layout

```
app.py                 Streamlit UI
config.py              paths, GEE/Sentinel-2 constants, defaults
download_data.py       LEVIR-CD fetch + export
resources/             bundled state/district registry
src/india_locations.py states, districts, city lookup, search, geocoding
src/satellite.py       Earth Engine scene selection and rendering
src/model.py           Siamese U-Net + Dice/BCE loss
src/dataset.py         LEVIR-CD dataset and loaders
src/train.py           training, threshold tuning, checkpointing
src/evaluate.py        held-out test metrics
src/metrics.py         confusion matrix and metric definitions
src/inference.py       checkpoint loading and change detection
src/utils.py           mask post-processing, areas, overlays
src/report.py          PDF report
```
