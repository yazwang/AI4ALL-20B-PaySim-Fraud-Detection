#!/usr/bin/env python
"""Smoke test for the Streamlit app and all three selectable models.

Verifies, for each of Logistic Regression, Random Forest, and XGBoost:
- the model loads (LR/RF from the bundle, XGBoost from its native JSON)
- single-transaction feature preparation succeeds
- predict_proba succeeds
- thresholding succeeds
- no dtype/categorical errors occur

Also boots the Streamlit app with streamlit.testing.v1.AppTest to catch any
bundle/model-loading exceptions during startup, and switches the model selector
to XGBoost to exercise its performance tab.

Usage:
    python scripts/smoke_test_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "app"))
sys.path.insert(0, str(REPO_ROOT))

import app as appmod  # noqa: E402

EXPECTED_MODELS = ["random_forest", "logistic_regression", "xgboost"]


def single_transaction() -> pd.DataFrame:
    return pd.DataFrame([{
        "step": 1, "type": "CASH_OUT", "amount": 1000.0,
        "oldbalanceOrg": 5000.0, "newbalanceOrig": 4000.0,
        "oldbalanceDest": 0.0, "newbalanceDest": 1000.0,
    }])


def batch_transactions() -> pd.DataFrame:
    rows = [
        {"step": 1, "type": "CASH_OUT", "amount": 1000.0, "oldbalanceOrg": 5000.0,
         "newbalanceOrig": 4000.0, "oldbalanceDest": 0.0, "newbalanceDest": 1000.0},
        {"step": 2, "type": "TRANSFER", "amount": 300000.0, "oldbalanceOrg": 100000.0,
         "newbalanceOrig": 0.0, "oldbalanceDest": 5000.0, "newbalanceDest": 305000.0},
        {"step": 3, "type": "PAYMENT", "amount": 250.0, "oldbalanceOrg": 1000.0,
         "newbalanceOrig": 750.0, "oldbalanceDest": 900.0, "newbalanceDest": 1150.0},
    ]
    return pd.DataFrame(rows)


def main() -> None:
    available = appmod.available_models
    assert set(available) == set(EXPECTED_MODELS), (
        f"available models {available} != expected {EXPECTED_MODELS}"
    )
    print(f"Available models: {available}")

    for model_key in available:
        info = appmod.bundle["models"][model_key]
        X_single = appmod.prepare_model_input(single_transaction(), model_key)
        pred, proba = appmod.predict(single_transaction(), model_key, 0.5)
        assert len(pred) == 1 and len(proba) == 1
        assert proba[0] >= 0.0 and proba[0] <= 1.0
        assert pred[0] in (0, 1)

        X_batch = appmod.prepare_model_input(batch_transactions(), model_key)
        pred_batch, proba_batch = appmod.predict(batch_transactions(), model_key, 0.5)
        assert len(pred_batch) == 3 and len(proba_batch) == 3
        print(
            f"{model_key:20s} selected_thr={info['selected_threshold']} "
            f"single_proba={proba[0]:.6f} batch_flagged={int(pred_batch.sum())}"
        )

    print("Direct prediction smoke test for LR / RF / XGBoost: PASSED")

    # Streamlit startup smoke test. Run in a FRESH subprocess: importing
    # app.py in bare mode above (for the direct tests) must not share the
    # process with an AppTest run, or Streamlit's form state leaks and reports
    # a spurious "Forms cannot be nested" error.
    import subprocess
    import sys

    app_script = REPO_ROOT / "app" / "app.py"
    apptest_code = (
        "from streamlit.testing.v1 import AppTest; "
        f"at = AppTest.from_file(r'{app_script}'); "
        "at.run(timeout=180); "
        "assert not at.exception, [str(e) for e in at.exception]; "
        "sb = at.sidebar.selectbox[0]; "
        "assert 'XGBoost' in sb.options, sb.options; "
        "assert len(sb.options) == 3, sb.options; "
        "sb.select('XGBoost'); "
        "at.run(timeout=180); "
        "assert not at.exception, [str(e) for e in at.exception]; "
        "print('STREAMLIT SMOKE TEST PASSED'); "
        "print('model selector:', sb.options)"
    )
    result = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", apptest_code],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr[-4000:])
        raise SystemExit("Streamlit smoke test failed (see stderr above).")


if __name__ == "__main__":
    main()
