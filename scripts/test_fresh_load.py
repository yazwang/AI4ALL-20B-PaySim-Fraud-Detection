#!/usr/bin/env python
"""Fresh-process smoke test for the native XGBoost deployment artifact.

Runs in a NEW Python process (deliberately separate from
scripts/validate_xgboost.py) to prove that app/xgboost_model.json loads under
the pinned dependency versions and generates predictions through the same
feature-preparation path the Streamlit app uses.

Checks:
1. The native JSON loads under xgboost==2.1.4 with enable_categorical=True.
2. Single-transaction feature preparation succeeds for every PaySim type.
3. predict_proba + thresholding succeed (no dtype/categorical errors).
4. On the authoritative 15,000-row test set, the loaded model reproduces the
   validated XGBoost results (19/19 fraud caught, confusion matrix TN/FP/FN/TP).

Usage:
    python scripts/test_fresh_load.py [--data PATH_TO_PaySim_DS.csv]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.validate_xgboost import (  # noqa: E402
    REPO_ROOT as _ROOT,
    load_real_paysim,
    reproduce_sample_and_split,
)

BUNDLE_PATH = REPO_ROOT / "app" / "fraud_detection_bundle.pkl"
XGBOOST_MODEL_PATH = REPO_ROOT / "app" / "xgboost_model.json"

TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]


def engineer_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Mirror the app's engineer_features (== notebook cell 13)."""

    df = raw_df.copy()
    df["amount_log"] = np.log1p(df["amount"])
    df["orig_balance_error"] = (
        df["oldbalanceOrg"] - df["newbalanceOrig"] - df["amount"]
    ).abs()
    df["dest_balance_error"] = (
        df["newbalanceDest"] - df["oldbalanceDest"] - df["amount"]
    ).abs()
    df["orig_account_emptied"] = (
        (df["oldbalanceOrg"] > 0) & (df["newbalanceOrig"] == 0)
    ).astype(int)
    df["hour_of_day"] = df["step"] % 24
    df["day"] = df["step"] // 24
    return df


def prepare_model_input(raw_df: pd.DataFrame, info: dict) -> pd.DataFrame:
    """Mirror the app's prepare_model_input for the native-categorical path."""

    feature_names = info["feature_names"]
    impute_means = info.get("impute_means", {})
    type_categories = info.get("type_categories", TRANSACTION_TYPES)

    df = engineer_features(raw_df)
    df["type"] = pd.Categorical(df["type"], categories=type_categories)

    X = pd.DataFrame(index=df.index)
    for col in feature_names:
        if col in df.columns:
            X[col] = df[col]
        else:
            X[col] = impute_means.get(col, 0)

    X["type"] = pd.Categorical(X["type"], categories=type_categories)
    return X[feature_names]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default=str(REPO_ROOT / "data" / "PaySim_DS.csv"),
        help="Path to the real PaySim CSV (default: data/PaySim_DS.csv).",
    )
    args = parser.parse_args()

    print(f"xgboost version: {xgb.__version__}  (requirements pins 2.1.4)")
    assert xgb.__version__ == "2.1.4", "xgboost version mismatch in fresh process"

    bundle = joblib.load(BUNDLE_PATH)
    info = bundle["models"]["xgboost"]
    assert info.get("model_source") == "native_json"

    model = xgb.XGBClassifier(enable_categorical=True)
    model.load_model(str(XGBOOST_MODEL_PATH))
    print(f"Loaded native model from {XGBOOST_MODEL_PATH.name}")

    # 1/2/3. Single-transaction path for every PaySim type.
    for txn_type in TRANSACTION_TYPES:
        row = pd.DataFrame([{
            "step": 1, "type": txn_type, "amount": 1000.0,
            "oldbalanceOrg": 5000.0, "newbalanceOrig": 4000.0,
            "oldbalanceDest": 0.0, "newbalanceDest": 1000.0,
        }])
        X = prepare_model_input(row, info)
        assert "type" in X and X["type"].dtype.name == "category"
        proba = model.predict_proba(X)[:, 1]
        flag = int((proba >= 0.5).astype(int)[0])
        print(f"single transaction [{txn_type:9s}]: proba={proba[0]:.6f} flag={flag}")
    print("Single-transaction feature prep + predict_proba + thresholding: OK")

    # 4. Batch: reproduce the authoritative test set and confirm the loaded
    #    model reproduces the validated confusion matrix.
    print(f"Loading real PaySim data from {args.data} ...")
    df = load_real_paysim(args.data)
    (
        _X_train, _X_val, X_test, _y_train, _y_val, y_test,
        _baseline_test, _X_model, _y_model,
    ) = reproduce_sample_and_split(df)

    X_test_prep = prepare_model_input(X_test, info)
    proba = model.predict_proba(X_test_prep)[:, 1]
    selected_thr = float(info["selected_threshold"])
    pred = (proba >= selected_thr).astype(int)

    n_test = int(len(y_test))
    n_fraud = int(y_test.sum())
    tn = int(((y_test == 0) & (pred == 0)).sum())
    fp = int(((y_test == 0) & (pred == 1)).sum())
    fn = int(((y_test == 1) & (pred == 0)).sum())
    tp = int(((y_test == 1) & (pred == 1)).sum())
    print(f"Test rows={n_test}  fraud={n_fraud}  flagged={int(pred.sum())}")
    print(f"Confusion matrix: TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"Threshold: {selected_thr}")

    assert n_test == 15_000, "fresh-process test set != 15,000 rows"
    assert n_fraud == 19, "fresh-process test fraud != 19 cases"
    assert tp == 19, "loaded XGBoost did not reproduce TP=19"
    assert (fp, fn) == (3, 0), "loaded XGBoost did not reproduce FP=3 / FN=0"
    print("FRESH-PROCESS LOAD TEST PASSED")


if __name__ == "__main__":
    main()
