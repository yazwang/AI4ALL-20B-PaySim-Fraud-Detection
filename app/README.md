# App

A Streamlit app for exploring the PaySim fraud-detection models trained in the
project notebook.

## Features

- **Predict** — enter a single transaction or upload a CSV, and get a fraud
  probability plus a flag at the selected decision threshold. If the upload
  includes an `isFraud` column, the app also reports precision, recall, F1,
  and a confusion matrix for that upload.
- **Dataset Insights** — class distribution and fraud rate by transaction
  type from the PaySim sample.
- **Model Performance** — model comparison table, confusion matrices at the
  default and validation-selected thresholds, precision-recall curves, and
  feature importance for Random Forest and Logistic Regression.

## Run

Live demo: [open the deployed Streamlit app](https://ai4all-20b-paysim-fraud-detection-mztvgddw2ckpfplbvb9vmo.streamlit.app/)

To run locally:

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

Run from the repository root as shown; the model bundle is loaded relative to
`app/app.py`, not the current working directory.

## Model bundle

The app loads `fraud_detection_bundle.pkl` from this directory. It contains
the exported models and evaluation artifacts used by the UI. The bundle was
generated with scikit-learn 1.6.1, and `requirements.txt` pins that version so
the pickled estimators load cleanly. The notebook's export cell that produced
this bundle is not yet in the repository (tracked as a follow-up), so the
bundle is not currently reproducible from source.

Predictions are exploratory: the models are trained on the synthetic PaySim
development sample and are not a validated production fraud system.
