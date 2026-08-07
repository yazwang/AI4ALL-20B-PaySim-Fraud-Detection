#!/usr/bin/env python
"""Revalidate and export XGBoost on the authoritative real-PaySim pipeline.

This script reproduces the exact sampling and splitting logic that produced the
final Logistic Regression and Random Forest results (100,000-row stratified
development sample; train/validation/test 70/15/15; final test set of 15,000
transactions containing 19 fraud cases), then trains XGBoost -- the team's
boosted-tree implementation originally contributed by Emmanuel A. Opoku -- on
the training split only.

Validation discipline:
- XGBoost is trained on the TRAINING split only.
- The VALIDATION split is used for early stopping and decision-threshold
  selection (same methodology as LR/RF: maximize validation F1).
- The untouched 15,000-row TEST set is used exactly once, for final evaluation.

After a successful run the script writes the deployment artifacts:
- app/xgboost_model.json          -- native XGBoost JSON (cross-version safe)
- app/fraud_detection_bundle.pkl  -- LR/RF models plus XGBoost
                                      metadata/evaluation artifacts

Before writing the bundle it verifies that the Logistic Regression and Random
Forest models reproduced here behave identically to the shipped bundle on the
15,000-row test set, so existing LR/RF behavior is preserved.

Usage:
    python scripts/validate_xgboost.py [--data PATH_TO_PaySim_DS.csv]

Expected data:
    data/PaySim_DS.csv  (the real ~6.36M-row PaySim CSV)

The real CSV must be used. This script never falls back to synthetic data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation import (  # noqa: E402
    DEFAULT_THRESHOLD,
    build_model_comparison,
    evaluate_is_flagged_fraud_baseline,
    evaluate_model,
    make_fraud_flags,
    select_threshold_from_validation,
)
from src.feature_engineering import (  # noqa: E402
    add_safe_features,
    split_target_baseline_and_predictors,
)

RANDOM_STATE = 42
SAMPLE_SIZE = 100_000

EXPECTED_FULL_ROWS = 6_362_620
EXPECTED_FULL_COLS = 11
EXPECTED_SAMPLE_ROWS = 100_000
EXPECTED_TEST_ROWS = 15_000
EXPECTED_TEST_FRAUD = 19

XGBOOST_MODEL_PATH = REPO_ROOT / "app" / "xgboost_model.json"
BUNDLE_PATH = REPO_ROOT / "app" / "fraud_detection_bundle.pkl"

# Authoritative final-test metrics from the real-PaySim run (documented in
# docs/key_findings.md and the notebook's model-comparison table). Reproducing
# these exactly is the proof that this script reuses the same pipeline.
AUTHORITATIVE_LR_DEFAULT = {
    "Precision": 11 / 12,
    "Recall": 11 / 19,
    "F1": 0.7096774193548387,
    "PR-AUC / Average Precision": 0.6717910845153309,
    "Number Flagged": 12,
}
AUTHORITATIVE_LR_SELECTED = {
    "Precision": 11 / 13,
    "Recall": 11 / 19,
    "F1": 0.6875,
    "PR-AUC / Average Precision": 0.6717910845153309,
    "Number Flagged": 13,
}
AUTHORITATIVE_RF = {
    "Precision": 1.0,
    "Recall": 1.0,
    "F1": 1.0,
    "PR-AUC / Average Precision": 1.0,
    "Number Flagged": 19,
}


def check_dependency_versions() -> None:
    """Assert the installed library versions match requirements.txt pins."""

    import sklearn

    checks = {
        "scikit-learn": (sklearn.__version__, "1.6.1"),
        "xgboost": (xgb.__version__, "2.1.4"),
        "pandas": (pd.__version__, "2.3.3"),
        "numpy": (np.__version__, "2.0.2"),
    }
    for name, (installed, pinned) in checks.items():
        if installed != pinned:
            raise SystemExit(
                f"Dependency mismatch for {name}: installed {installed}, "
                f"requirements.txt pins {pinned}. Install the pinned versions "
                "before exporting model artifacts."
            )


def load_real_paysim(data_path: str | Path) -> pd.DataFrame:
    """Load the real PaySim CSV. Fails instead of falling back to synthetic data."""

    df = pd.read_csv(data_path)
    if df.shape != (EXPECTED_FULL_ROWS, EXPECTED_FULL_COLS):
        raise SystemExit(
            f"Expected the real PaySim CSV with {EXPECTED_FULL_ROWS} rows x "
            f"{EXPECTED_FULL_COLS} columns, got {df.shape}. Refusing to run on "
            f"anything other than the real dataset. Place the CSV at "
            f"data/PaySim_DS.csv and re-run."
        )
    return df


def reproduce_sample_and_split(df: pd.DataFrame):
    """Reproduce the notebook's feature engineering, sampling, and split.

    Mirrors cells 13, 15, 17, and 19 of
    notebooks/AI4ALL_Group20B_PaySim_Fraud_Detection_FINAL.ipynb.
    """

    df_fe = add_safe_features(df)
    X, y, baseline_flag = split_target_baseline_and_predictors(df_fe)

    # Cell 17: small stratified sample for the development run.
    sample_size = min(SAMPLE_SIZE, len(df_fe))
    sample_index, _ = train_test_split(
        df_fe.index,
        train_size=sample_size,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=y,
    )
    X_model = X.loc[sample_index].copy()
    y_model = y.loc[sample_index].copy()
    baseline_model = baseline_flag.loc[sample_index].copy()

    # Cell 19: train / validation / test split.
    X_train, X_temp, y_train, y_temp, baseline_train, baseline_temp = train_test_split(
        X_model,
        y_model,
        baseline_model,
        train_size=0.70,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=y_model,
    )
    X_val, X_test, y_val, y_test, baseline_val, baseline_test = train_test_split(
        X_temp,
        y_temp,
        baseline_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=y_temp,
    )

    assert len(y_model) == EXPECTED_SAMPLE_ROWS
    assert len(y_train) == 70_000
    assert len(y_val) == EXPECTED_TEST_ROWS
    assert len(y_test) == EXPECTED_TEST_ROWS
    assert int(y_test.sum()) == EXPECTED_TEST_FRAUD

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        baseline_test,
        X_model,
        y_model,
    )


# ---------------------------------------------------------------------------
# Shared preprocessing helpers (mirror notebook cell 21).
# ---------------------------------------------------------------------------


def prepare_numeric_features(X_train_data: pd.DataFrame, X_other_data: pd.DataFrame):
    """Keep numeric columns and fill missing values using training-set means."""

    numeric_cols = X_train_data.select_dtypes(include="number").columns.tolist()
    X_train_numeric = X_train_data[numeric_cols].copy()
    X_other_numeric = X_other_data[numeric_cols].copy()

    for col in numeric_cols:
        mean_val = X_train_numeric[col].mean()
        X_train_numeric[col] = X_train_numeric[col].fillna(mean_val)
        X_other_numeric[col] = X_other_numeric[col].fillna(mean_val)

    return X_train_numeric, X_other_numeric


def fit_tree_preprocessor(X_train_data: pd.DataFrame):
    """One-hot encode training data and remember the final training columns."""

    X_train_tree = pd.get_dummies(X_train_data.copy(), columns=["type"], drop_first=False)
    for col in X_train_tree.columns:
        X_train_tree[col] = X_train_tree[col].fillna(X_train_tree[col].mean())
    return X_train_tree, X_train_tree.columns.tolist(), X_train_tree.mean(numeric_only=True)


def transform_tree_features(
    X_other_data: pd.DataFrame,
    training_columns: list[str],
    training_means,
) -> pd.DataFrame:
    """Apply the training one-hot columns and training means to validation/test data."""

    X_other_tree = pd.get_dummies(X_other_data.copy(), columns=["type"], drop_first=False)
    X_other_tree = X_other_tree.reindex(columns=training_columns, fill_value=0)
    for col in training_columns:
        X_other_tree[col] = X_other_tree[col].fillna(training_means[col])
    return X_other_tree


# ---------------------------------------------------------------------------
# LR / RF reproduction (mirror notebook cells 25 and 27).
# ---------------------------------------------------------------------------


def train_and_evaluate_logistic_regression(
    X_train_data: pd.DataFrame,
    y_train_data: pd.Series,
    X_val_data: pd.DataFrame,
    y_val_data: pd.Series,
    X_test_data: pd.DataFrame,
    y_test_data: pd.Series,
):
    X_train_numeric, X_val_numeric = prepare_numeric_features(X_train_data, X_val_data)
    _, X_test_numeric = prepare_numeric_features(X_train_data, X_test_data)

    model_pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42, solver="liblinear"),
    )
    model_pipeline.fit(X_train_numeric, y_train_data)

    y_val_proba = model_pipeline.predict_proba(X_val_numeric)[:, 1]
    selected_threshold, lr_threshold_table = select_threshold_from_validation(
        y_val_data, y_val_proba
    )

    y_test_proba = model_pipeline.predict_proba(X_test_numeric)[:, 1]
    y_pred_default = make_fraud_flags(y_test_proba, DEFAULT_THRESHOLD)
    y_pred_selected = make_fraud_flags(y_test_proba, selected_threshold)

    default_results = evaluate_model(
        "Logistic Regression (threshold 0.50)",
        y_test_data,
        y_pred_default,
        y_test_proba,
        threshold=DEFAULT_THRESHOLD,
    )
    selected_results = evaluate_model(
        "Logistic Regression (validation-selected threshold)",
        y_test_data,
        y_pred_selected,
        y_test_proba,
        threshold=selected_threshold,
    )

    return {
        "model_object": model_pipeline,
        "feature_names": X_train_numeric.columns.tolist(),
        "impute_means": {c: float(X_train_numeric[c].mean()) for c in X_train_numeric.columns},
        "uses_type_dummies": False,
        "default_threshold": DEFAULT_THRESHOLD,
        "selected_threshold": selected_threshold,
        "threshold_table": lr_threshold_table,
        "default_results": default_results,
        "selected_results": selected_results,
        "test_scores": y_test_proba,
        "test_flags_default": y_pred_default,
        "test_flags_selected": y_pred_selected,
    }


def train_and_evaluate_random_forest(
    X_train_data: pd.DataFrame,
    y_train_data: pd.Series,
    X_val_data: pd.DataFrame,
    y_val_data: pd.Series,
    X_test_data: pd.DataFrame,
    y_test_data: pd.Series,
):
    X_train_rf = X_train_data.copy()
    X_val_rf = X_val_data.copy()
    X_test_rf = X_test_data.copy()

    drop_cols = ["nameOrig", "nameDest", "isFlaggedFraud", "orig_txn_count", "dest_txn_count"]
    X_train_rf = X_train_rf.drop(columns=[c for c in drop_cols if c in X_train_rf.columns])
    X_val_rf = X_val_rf.drop(columns=[c for c in drop_cols if c in X_val_rf.columns])
    X_test_rf = X_test_rf.drop(columns=[c for c in drop_cols if c in X_test_rf.columns])

    X_train_rf, rf_training_columns, rf_training_means = fit_tree_preprocessor(X_train_rf)
    X_val_rf = transform_tree_features(X_val_rf, rf_training_columns, rf_training_means)
    X_test_rf = transform_tree_features(X_test_rf, rf_training_columns, rf_training_means)

    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_train_rf, y_train_data)

    y_val_proba = rf_model.predict_proba(X_val_rf)[:, 1]
    selected_threshold, rf_threshold_table = select_threshold_from_validation(
        y_val_data, y_val_proba
    )

    y_test_proba = rf_model.predict_proba(X_test_rf)[:, 1]
    y_pred_default = make_fraud_flags(y_test_proba, DEFAULT_THRESHOLD)
    y_pred_selected = make_fraud_flags(y_test_proba, selected_threshold)

    default_results = evaluate_model(
        "Random Forest (threshold 0.50)",
        y_test_data,
        y_pred_default,
        y_test_proba,
        threshold=DEFAULT_THRESHOLD,
    )
    selected_results = evaluate_model(
        "Random Forest (validation-selected threshold)",
        y_test_data,
        y_pred_selected,
        y_test_proba,
        threshold=selected_threshold,
    )

    return {
        "model_object": rf_model,
        "feature_names": X_train_rf.columns.tolist(),
        "impute_means": {c: float(rf_training_means[c]) for c in X_train_rf.columns},
        "uses_type_dummies": True,
        "default_threshold": DEFAULT_THRESHOLD,
        "selected_threshold": selected_threshold,
        "threshold_table": rf_threshold_table,
        "default_results": default_results,
        "selected_results": selected_results,
        "test_scores": y_test_proba,
        "test_flags_default": y_pred_default,
        "test_flags_selected": y_pred_selected,
    }


# ---------------------------------------------------------------------------
# XGBoost (mirror notebook cell 29, plus validation-based thresholding).
# ---------------------------------------------------------------------------


def train_and_evaluate_xgboost(
    X_train_data: pd.DataFrame,
    y_train_data: pd.Series,
    X_val_data: pd.DataFrame,
    y_val_data: pd.Series,
    X_test_data: pd.DataFrame,
    y_test_data: pd.Series,
):
    """Train XGBoost on the training split, tune the threshold on validation.

    Early stopping monitors the validation split with PR-AUC ('aucpr'), and the
    decision threshold is chosen on the validation split with the same
    select_threshold_from_validation methodology used for LR/RF. The test set is
    evaluated exactly once, after the threshold is frozen.
    """

    X_train_xgb = X_train_data.copy()
    X_val_xgb = X_val_data.copy()
    X_test_xgb = X_test_data.copy()

    # Convert object (text) columns to pandas 'category' so XGBoost can read
    # them natively (enable_categorical=True), matching the notebook.
    for col in X_train_xgb.select_dtypes(include=["object"]).columns:
        X_train_xgb[col] = X_train_xgb[col].astype("category")
        X_val_xgb[col] = X_val_xgb[col].astype("category")
        X_test_xgb[col] = X_test_xgb[col].astype("category")

    # Class-imbalance handling: scale_pos_weight computed from training data only.
    neg_count = (y_train_data == 0).sum()
    pos_count = (y_train_data == 1).sum()
    imbalance_ratio = neg_count / pos_count

    xgb_model = xgb.XGBClassifier(
        scale_pos_weight=imbalance_ratio,
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        eval_metric="aucpr",
        random_state=42,
        tree_method="hist",
        enable_categorical=True,
        early_stopping_rounds=10,
        nthread=1,
    )

    xgb_model.fit(
        X_train_xgb,
        y_train_data,
        eval_set=[(X_val_xgb, y_val_data)],
        verbose=False,
    )

    # Threshold selection on the validation split only.
    y_val_proba = xgb_model.predict_proba(X_val_xgb)[:, 1]
    selected_threshold, xgb_threshold_table = select_threshold_from_validation(
        y_val_data, y_val_proba
    )

    # Evaluate exactly once on the untouched test set.
    y_test_proba = xgb_model.predict_proba(X_test_xgb)[:, 1]
    y_pred_default = make_fraud_flags(y_test_proba, DEFAULT_THRESHOLD)
    y_pred_selected = make_fraud_flags(y_test_proba, selected_threshold)

    default_results = evaluate_model(
        "XGBoost (threshold 0.50)",
        y_test_data,
        y_pred_default,
        y_test_proba,
        threshold=DEFAULT_THRESHOLD,
    )
    selected_results = evaluate_model(
        "XGBoost (validation-selected threshold)",
        y_test_data,
        y_pred_selected,
        y_test_proba,
        threshold=selected_threshold,
    )

    feature_importance = {
        k: float(v) for k, v in xgb_model.get_booster().get_score(importance_type="gain").items()
    }

    return {
        "model_object": xgb_model,
        "feature_names": X_train_xgb.columns.tolist(),
        "impute_means": {
            c: float(X_train_xgb[c].mean())
            for c in X_train_xgb.columns
            if not isinstance(X_train_xgb[c].dtype, pd.CategoricalDtype)
        },
        "categorical_mode": "native",
        "type_categories": list(X_train_xgb["type"].cat.categories),
        "scale_pos_weight": float(imbalance_ratio),
        "default_threshold": DEFAULT_THRESHOLD,
        "selected_threshold": selected_threshold,
        "threshold_table": xgb_threshold_table,
        "default_results": default_results,
        "selected_results": selected_results,
        "test_scores": y_test_proba,
        "test_flags_default": y_pred_default,
        "test_flags_selected": y_pred_selected,
        "feature_importance": feature_importance,
    }


def assert_results_match(results: dict, expected: dict, label: str) -> None:
    """Assert one model's test metrics match the documented authoritative values."""

    problems = []
    for metric, value in expected.items():
        actual = results[metric]
        if metric == "Number Flagged":
            if int(actual) != int(value):
                problems.append(f"{metric}: got {actual}, expected {value}")
        else:
            if not np.isclose(float(actual), float(value), atol=1e-6):
                problems.append(f"{metric}: got {actual}, expected {value}")
    if problems:
        raise SystemExit(f"{label} did not reproduce the authoritative metrics: {problems}")


def prepare_shipped_test_features(key: str, X_test: pd.DataFrame, shipped_entry: dict):
    """Build the feature matrix a shipped bundle model expects on the test set."""

    if key == "logistic_regression":
        _, X_test_numeric = prepare_numeric_features(X_test, X_test)
        return X_test_numeric
    X_test_rf = X_test.copy()
    drop_cols = ["nameOrig", "nameDest", "isFlaggedFraud", "orig_txn_count", "dest_txn_count"]
    X_test_rf = X_test_rf.drop(columns=[c for c in drop_cols if c in X_test_rf.columns])
    return transform_tree_features(
        X_test_rf,
        shipped_entry["feature_names"],
        shipped_entry["impute_means"],
    )


def verify_lr_rf_reproduce_ship(
    shipped: dict,
    lr_output: dict,
    rf_output: dict,
    X_test: pd.DataFrame,
) -> None:
    """Verify reproduced LR/RF test predictions match the shipped bundle.

    Stops before writing anything if the reproduced models differ from the
    shipped LR/RF models, so existing LR/RF behavior is never silently changed.
    """

    for key, output in [
        ("logistic_regression", lr_output),
        ("random_forest", rf_output),
    ]:
        shipped_entry = shipped["models"][key]
        if float(output["selected_threshold"]) != float(shipped_entry["selected_threshold"]):
            raise SystemExit(
                f"{key}: reproduced validation-selected threshold "
                f"{output['selected_threshold']} differs from shipped "
                f"{shipped_entry['selected_threshold']}. Refusing to overwrite the bundle."
            )
        shipped_X_test = prepare_shipped_test_features(key, X_test, shipped_entry)
        shipped_proba = shipped_entry["model"].predict_proba(shipped_X_test)[:, 1]
        if not np.allclose(shipped_proba, output["test_scores"], atol=1e-6):
            raise SystemExit(
                f"{key}: reproduced test probabilities differ from the shipped "
                "bundle. Refusing to overwrite the bundle."
            )
        print(f"Verified: reproduced {key} test predictions match the shipped bundle.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default=str(REPO_ROOT / "data" / "PaySim_DS.csv"),
        help="Path to the real PaySim CSV (default: data/PaySim_DS.csv).",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Run the full validation but do not write bundle/model artifacts.",
    )
    args = parser.parse_args()

    check_dependency_versions()

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(
            f"Real PaySim CSV not found at {data_path}. Place PaySim_DS.csv at "
            f"data/PaySim_DS.csv (or pass --data) and re-run. The real CSV "
            "is required; this script never falls back to synthetic data."
        )

    print(f"Loading real PaySim data from {data_path} ...")
    df = load_real_paysim(data_path)
    print(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns.")

    print("Reproducing sampling and split ...")
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        baseline_test,
        _X_model,
        _y_model,
    ) = reproduce_sample_and_split(df)
    print(
        f"Sample={len(_y_model):,}  Train={len(y_train):,}  "
        f"Val={len(y_val):,}  Test={len(y_test):,}  "
        f"Test fraud={int(y_test.sum())}"
    )
    assert len(y_test) == EXPECTED_TEST_ROWS, "test rows != 15,000"
    assert int(y_test.sum()) == EXPECTED_TEST_FRAUD, "test fraud != 19"

    # -------------------- LR / RF reproduction --------------------
    print("Reproducing Logistic Regression on the training split ...")
    lr_output = train_and_evaluate_logistic_regression(
        X_train, y_train, X_val, y_val, X_test, y_test
    )
    print("Reproducing Random Forest on the training split ...")
    rf_output = train_and_evaluate_random_forest(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    for label, output, expected_default, expected_selected in [
        (
            "Logistic Regression",
            lr_output,
            AUTHORITATIVE_LR_DEFAULT,
            AUTHORITATIVE_LR_SELECTED,
        ),
        ("Random Forest", rf_output, AUTHORITATIVE_RF, AUTHORITATIVE_RF),
    ]:
        assert_results_match(output["default_results"], expected_default, f"{label} (default)")
        assert_results_match(output["selected_results"], expected_selected, f"{label} (selected)")
        print(
            f"{label}: selected threshold={output['selected_threshold']}, "
            f"default PR-AUC={output['default_results']['PR-AUC / Average Precision']:.6f}, "
            f"flagged={output['selected_results']['Number Flagged']}"
        )

    # Verify the reproduced split and LR/RF predictions match the shipped bundle.
    if not BUNDLE_PATH.exists():
        raise SystemExit(
            f"Missing {BUNDLE_PATH}. Refusing to regenerate without the shipped "
            "bundle to verify LR/RF predictions against."
        )
    shipped = joblib.load(BUNDLE_PATH)
    verify_lr_rf_reproduce_ship(shipped, lr_output, rf_output, X_test)

    # -------------------- XGBoost --------------------
    print("Training XGBoost on the training split (early stopping on validation) ...")
    xgb_output = train_and_evaluate_xgboost(
        X_train, y_train, X_val, y_val, X_test, y_test
    )
    xgb_default = xgb_output["default_results"]
    xgb_selected = xgb_output["selected_results"]

    print()
    print("=" * 70)
    print("XGBoost final test-set evaluation (validated)")
    print("=" * 70)
    print(f"Test rows = {len(y_test):,}  (asserted = {EXPECTED_TEST_ROWS:,})")
    print(f"Test fraud = {int(y_test.sum())}  (asserted = {EXPECTED_TEST_FRAUD})")
    print(f"Selected threshold (validation F1): {xgb_output['selected_threshold']}")
    print(f"Default threshold results  : {xgb_default}")
    print(f"Selected threshold results : {xgb_selected}")

    cm = np.asarray(xgb_selected["Confusion Matrix"])
    tn, fp, fn, tp = cm.ravel()
    print(f"Confusion matrix (selected threshold): TN={tn} FP={fp} FN={fn} TP={tp}")

    # -------------------- Export --------------------
    if args.no_write:
        print("\n--no-write set; skipping artifact export.")
        return

    XGBOOST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    xgb_output["model_object"].save_model(str(XGBOOST_MODEL_PATH))
    print(f"\nSaved native XGBoost model -> {XGBOOST_MODEL_PATH}")

    # Reload the native artifact in this process and prove identical predictions.
    reloaded = xgb.XGBClassifier(enable_categorical=True)
    reloaded.load_model(str(XGBOOST_MODEL_PATH))
    X_test_xgb = X_test.copy()
    for col in X_test_xgb.select_dtypes(include=["object"]).columns:
        X_test_xgb[col] = X_test_xgb[col].astype("category")
    reloaded_pred = reloaded.predict(X_test_xgb)
    original_pred = xgb_output["model_object"].predict(X_test_xgb)
    if not np.array_equal(reloaded_pred, original_pred):
        raise SystemExit("Native model reload produced different predictions; aborting export.")
    print("Verified: reloaded native XGBoost model reproduces identical predictions.")

    # Build the model comparison list (baseline + LR/RF/XGBoost at both thresholds).
    baseline_results = evaluate_is_flagged_fraud_baseline(y_test, baseline_test)
    model_results = [
        baseline_results,
        lr_output["default_results"],
        lr_output["selected_results"],
        rf_output["default_results"],
        rf_output["selected_results"],
        xgb_output["default_results"],
        xgb_output["selected_results"],
    ]
    model_comparison = build_model_comparison(model_results).to_dict("records")

    # Assemble the deployment bundle.
    bundle = {}
    bundle["_export_info"] = {
        "generated_by": "scripts/validate_xgboost.py",
        "pipeline": "real PaySim CSV -> 100,000-row stratified sample -> 70/15/15 split",
        "test_rows": int(len(y_test)),
        "test_fraud_cases": int(y_test.sum()),
        "xgboost_serialization": "native JSON (app/xgboost_model.json)",
        "sklearn": "1.6.1",
        "xgboost": "2.1.4",
    }
    bundle["models"] = {}
    for key, output in [
        ("logistic_regression", lr_output),
        ("random_forest", rf_output),
    ]:
        bundle["models"][key] = {
            "model": output["model_object"],
            "feature_names": output["feature_names"],
            "impute_means": output["impute_means"],
            "uses_type_dummies": output["uses_type_dummies"],
            "default_threshold": output["default_threshold"],
            "selected_threshold": output["selected_threshold"],
        }
    bundle["models"]["xgboost"] = {
        "model_source": "native_json",
        "model_path": "xgboost_model.json",
        "feature_names": xgb_output["feature_names"],
        "impute_means": xgb_output["impute_means"],
        "categorical_mode": "native",
        "type_categories": xgb_output["type_categories"],
        "scale_pos_weight": xgb_output["scale_pos_weight"],
        "default_threshold": xgb_output["default_threshold"],
        "selected_threshold": xgb_output["selected_threshold"],
    }

    bundle["model_comparison"] = model_comparison
    bundle["confusion_matrices"] = {
        key: {
            "default": np.asarray(output["default_results"]["Confusion Matrix"]).tolist(),
            "selected": np.asarray(output["selected_results"]["Confusion Matrix"]).tolist(),
        }
        for key, output in [
            ("logistic_regression", lr_output),
            ("random_forest", rf_output),
            ("xgboost", xgb_output),
        ]
    }
    bundle["pr_curve_data"] = {}
    bundle["pr_auc"] = {}
    for key, output in [
        ("logistic_regression", lr_output),
        ("random_forest", rf_output),
        ("xgboost", xgb_output),
    ]:
        pr_precision, pr_recall, _ = precision_recall_curve(y_test, output["test_scores"])
        bundle["pr_curve_data"][key] = {
            "precision": pr_precision.tolist(),
            "recall": pr_recall.tolist(),
        }
        bundle["pr_auc"][key] = float(
            output["default_results"]["PR-AUC / Average Precision"]
        )
    bundle["threshold_tables"] = {
        key: output["threshold_table"].to_dict("records")
        for key, output in [
            ("logistic_regression", lr_output),
            ("random_forest", rf_output),
            ("xgboost", xgb_output),
        ]
    }
    bundle["feature_importance"] = {
        "random_forest": dict(
            zip(rf_output["feature_names"], rf_output["model_object"].feature_importances_.tolist())
        ),
        "logistic_regression": {
            k: float(v)
            for k, v in zip(
                lr_output["feature_names"],
                np.abs(lr_output["model_object"].steps[-1][1].coef_).ravel().tolist(),
            )
        },
        "xgboost": xgb_output["feature_importance"],
    }
    bundle["eda"] = {
        "class_distribution": {int(k): int(v) for k, v in df["isFraud"].value_counts().items()},
        "fraud_rate_by_type": {
            str(k): float(v) for k, v in df.groupby("type")["isFraud"].mean().items()
        },
    }

    joblib.dump(bundle, BUNDLE_PATH)
    print(f"Saved deployment bundle -> {BUNDLE_PATH}")
    print("XGBoost validation and export complete.")


if __name__ == "__main__":
    main()
