"""Safe feature engineering helpers for the PaySim fraud project.

These helpers extract Esther Wang's confirmed feature-engineering work from the
final notebook into a reusable Python module. They do not train models and they
do not load the PaySim dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TARGET_COLUMN = "isFraud"
BASELINE_COLUMN = "isFlaggedFraud"

# These columns are intentionally excluded from model predictors.
# - isFraud is the target we want to predict.
# - isFlaggedFraud is the original PaySim baseline rule, not a model input.
# - Raw account identifiers can encourage memorization instead of learning
#   general transaction patterns.
# - Account-frequency features were kept for EDA in the notebook, but excluded
#   from this first cleaned model feature set.
EXCLUDED_PREDICTOR_COLUMNS = [
    TARGET_COLUMN,
    BASELINE_COLUMN,
    "nameOrig",
    "nameDest",
    "orig_txn_count",
    "dest_txn_count",
]


def add_safe_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with safe engineered PaySim features added.

    The added features are based only on transaction fields that are available
    before modeling. This function does not use the fraud target as an input
    feature.

    Added columns:
    - amount_log
    - orig_balance_error
    - dest_balance_error
    - orig_account_emptied
    - hour_of_day
    - day
    """

    df_fe = df.copy()

    # Log-transform amount to reduce the effect of very large transactions.
    df_fe["amount_log"] = np.log1p(df_fe["amount"])

    # Balance error features compare the transaction amount with balance changes.
    df_fe["orig_balance_error"] = (
        df_fe["oldbalanceOrg"] - df_fe["newbalanceOrig"] - df_fe["amount"]
    ).abs()

    df_fe["dest_balance_error"] = (
        df_fe["newbalanceDest"] - df_fe["oldbalanceDest"] - df_fe["amount"]
    ).abs()

    # Indicator for sender account being emptied by the transaction.
    df_fe["orig_account_emptied"] = (
        (df_fe["oldbalanceOrg"] > 0) & (df_fe["newbalanceOrig"] == 0)
    ).astype(int)

    # Time features from the PaySim simulation step.
    df_fe["hour_of_day"] = df_fe["step"] % 24
    df_fe["day"] = df_fe["step"] // 24

    return df_fe


def split_target_baseline_and_predictors(
    df_fe: pd.DataFrame,
    excluded_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Separate predictors, the fraud target, and the baseline fraud flag.

    This helper does not create a train-validation-test split. It only applies
    the notebook's intentional predictor exclusions in one place.
    """

    columns_to_exclude = excluded_columns or EXCLUDED_PREDICTOR_COLUMNS

    present_excluded_columns = [col for col in columns_to_exclude if col in df_fe.columns]

    X = df_fe.drop(columns=present_excluded_columns)
    y = df_fe[TARGET_COLUMN]
    baseline_flag = df_fe[BASELINE_COLUMN]

    return X, y, baseline_flag
