# AI4ALL Group 20B PaySim Fraud Detection

## Project Overview

This repository contains the cleaned final notebook for the AI4ALL Group 20B PaySim fraud-detection project. The project explores whether supervised machine-learning models can identify fraudulent mobile-money transactions from the PaySim synthetic transaction dataset.

The current repository is intentionally lightweight: it includes the final notebook, documentation, and setup files, but it does not include the full PaySim CSV dataset because the dataset is too large for GitHub.

## Project Scope and Positioning

Machine-learning fraud detection using the PaySim dataset has been explored in prior research and educational projects, including work involving ensemble methods, explainable AI, anomaly detection, and graph-based approaches.

This project does not claim to introduce a novel fraud-detection algorithm. Instead, it develops an end-to-end educational workflow for comparing supervised machine-learning models under severe class imbalance. The project emphasizes leakage-aware feature engineering, validation-based threshold selection, comparison with the original `isFlaggedFraud` rule, false-positive trade-offs, and responsible interpretation of results obtained from synthetic data.

## Problem Statement

Mobile-payment fraud is rare compared with normal transaction activity, which makes fraud detection an imbalanced-classification problem. The goal of this project is to compare model-generated fraud flags against the original PaySim `isFlaggedFraud` rule and evaluate whether machine-learning models can improve recall while keeping false positives low.

## Dataset Description

The project uses the PaySim mobile-payment dataset, a synthetic dataset based on mobile-money transaction behavior. The notebook expects transaction-level records with fields such as:

- `step`
- `type`
- `amount`
- `nameOrig`
- `oldbalanceOrg`
- `newbalanceOrig`
- `nameDest`
- `oldbalanceDest`
- `newbalanceDest`
- `isFraud`
- `isFlaggedFraud`

The full dataset has approximately 6.3 million rows and is about 470 MB, so it is not stored in this repository. Before final submission, the team should document the authoritative dataset source and access instructions.

## Project Workflow

The final notebook follows this workflow:

1. Import required libraries and define development settings.
2. Load the PaySim CSV from Google Drive when running in Colab.
3. Perform quick data checks and exploratory review.
4. Create safe engineered features.
5. Separate the fraud target from the original PaySim rule baseline.
6. Build a small stratified development sample.
7. Split the sample into train, validation, and test sets.
8. Train and evaluate Logistic Regression and Random Forest models.
9. Select model thresholds using validation data only.
10. Evaluate final performance on the untouched test set.
11. Compare model results with the original `isFlaggedFraud` baseline.

## Safe Feature Engineering

The notebook uses engineered features that are available from transaction information without using the target label as an input. Current engineered features include:

- `amount_log`
- `orig_balance_error`
- `dest_balance_error`
- `orig_account_emptied`
- `hour_of_day`
- `day`

The notebook excludes `isFraud`, `isFlaggedFraud`, raw account identifiers, and account-frequency features from the predictor set for the current cleaned implementation.

## Models Currently Included

The cleaned final notebook currently includes:

- Original PaySim `isFlaggedFraud` baseline
- Logistic Regression
- Random Forest
- Gradient Boosting placeholder for the assigned owner

Gradient Boosting has not yet been implemented in the final cleaned notebook.

## Evaluation Metrics

The project compares models using:

- Precision
- Recall
- F1 score
- PR-AUC / Average Precision
- False-positive rate
- Number of transactions flagged
- Percentage of transactions flagged
- Confusion matrix and classification report

These metrics are especially important because fraud is rare and accuracy alone can be misleading.

## Threshold-Based Flagging

The notebook evaluates both default and validation-selected decision thresholds. Thresholds are chosen using validation-set F1 score, and the untouched test set is used only after threshold selection is complete.

This design helps avoid tuning directly on the test set.

## Current Development-Sample Findings

The saved notebook outputs are based on a 100,000-row stratified development sample from the full PaySim dataset. The test set contains 15,000 transactions, including 19 fraud cases.

Random Forest achieved perfect performance on this current development sample: it identified all 19 fraud cases with no false-positive fraud flags at both the default threshold of 0.50 and the validation-selected threshold of 0.30. Its precision, recall, F1, and PR-AUC / Average Precision were all 1.000000 on this test split.

This result should be interpreted carefully. It does not prove that Random Forest will generalize perfectly to the full PaySim dataset or to new transaction data. The result is based only on the current development sample containing 19 fraud cases in the test set.

Logistic Regression found 11 of the 19 fraud cases. At the default threshold of 0.50, it flagged 12 transactions, with precision 0.916667, recall 0.578947, F1 0.709677, and PR-AUC / Average Precision 0.671791. Lowering the threshold to the validation-selected value of 0.15 did not improve recall on the test set and reduced precision.

The original `isFlaggedFraud` baseline did not flag any transactions in this test set and did not catch any of the 19 fraud cases.

## Extracted Reusable Components

The `src/` folder contains reusable safe feature-engineering and shared
flagging/evaluation utilities extracted from the final notebook. These
components are separated from the model-training sections so they can be
reviewed, tested, and reused independently.

The saved, data-supported findings are available in
`docs/key_findings.md`. Team authorship and contribution details are
documented in `CONTRIBUTIONS.md`.

## Repository Structure

```text
AI4ALL-PaySim-Fraud-Detection/
├── README.md
├── CONTRIBUTIONS.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── AI4ALL_Group20B_PaySim_Fraud_Detection_FINAL.ipynb
├── docs/
│   ├── 20B Project Proposal.pdf
│   └── key_findings.md
├── src/
│   ├── evaluation.py
│   └── feature_engineering.py
├── data/
│   └── README.md
└── app/
    ├── app.py
    ├── fraud_detection_bundle.pkl
    └── README.md
```

## Setup and Reproduction

1. Clone or open this repository locally.
2. Create and activate a Python virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Place the PaySim CSV outside Git history at:

```text
data/PaySim_DS.csv
```

5. Open the final notebook:

```text
notebooks/AI4ALL_Group20B_PaySim_Fraud_Detection_FINAL.ipynb
```

6. If running in Google Colab, mount Google Drive and update `DATA_PATH` in the notebook if needed.
7. Run the notebook from top to bottom.

The notebook currently uses a small stratified sample first. Full-dataset training is intentionally blocked in the cleaned version until the project owner approves a full run.

## Limitations

- The full PaySim dataset is not included in this repository.
- Current saved findings are based on a development sample, not a complete final full-dataset experiment.
- The Random Forest result is perfect only on the current development sample containing 19 fraud cases in the test set.
- Gradient Boosting is still a placeholder.
- Dataset source documentation must be finalized before submission.
- The `fraud_detection_bundle.pkl` used by the Streamlit app was generated with scikit-learn 1.6.1; the export cell that produced it is not yet in the notebook, so the bundle is not yet reproducible from this repository.

## Streamlit Deployment

The `app/` directory contains a Streamlit app (`app/app.py`) that deploys the
Logistic Regression and Random Forest models on the PaySim data. It supports
single-transaction input or CSV upload, fraud probability flagging with an
adjustable decision threshold, dataset insights, and model-performance views
(confusion matrices, PR curves, feature importance).

To run it locally:

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

The app loads `app/fraud_detection_bundle.pkl`, which stores the exported
models and evaluation artifacts. The bundle was generated with
scikit-learn 1.6.1, so `requirements.txt` pins that version. Predictions are
exploratory and based on the synthetic PaySim development sample.

## Team

AI4ALL Ignite Fellowship, Group 20B.

Team contribution details are drafted in `CONTRIBUTIONS.md` and should be finalized only with confirmed owner names and confirmed task history.

## Acknowledgments

The team thanks the instructors of AI4ALL Ignite Fellowship Group 20B for their guidance and support. The team also acknowledges AI4ALL for providing the project structure and learning community.
