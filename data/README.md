# Data

This project uses the PaySim mobile-payment dataset for fraud-detection modeling.

The full PaySim CSV dataset is not stored in this GitHub repository. The dataset is approximately 470 MB, which is too large for normal repository storage and should remain outside Git history.

## Authoritative Dataset Source

The team's shared, authoritative copy of the PaySim CSV is stored in the project Google Drive folder:

- Shared Drive folder: https://drive.google.com/drive/folders/1fXSw6mDSg26FKkuGm8Iw_fMegIbRdsr3?usp=sharing
- Direct PaySim CSV link used by the final Colab notebook: https://drive.google.com/file/d/1A6BxEjk_ryaoFNMVxoMaH3XViXDZst9_/view?usp=sharing

The final Colab notebook loads the real PaySim CSV (6,362,620 rows x 11 columns) from Google Drive at `/content/drive/MyDrive/PaySim_DS.csv` unless `DATA_PATH` is updated.

## Dataset Citation

The PaySim dataset was originally described in:

- Lopez-Rojas, E. A., Elmir, A., & Axelsson, S. (2016). "PaySim: A financial mobile money simulator for fraud detection."
  https://www.msc-les.org/proceedings/emss/2016/EMSS2016_249.pdf
- Kaggle. "Synthetic Financial Datasets for Fraud Detection" (the public PaySim distribution).
  https://www.kaggle.com/datasets/ealaxi/paysim1

## Local Reproduction

For local reproduction, place the CSV file here:

```text
data/PaySim_DS.csv
```

If the real CSV is not available, the notebook falls back to a small synthetic
development sample so the pipeline can be checked from top to bottom. Any
results produced under that fallback are a pipeline sanity check only and are
not the project's final metrics.
