"""Shared fraud-flagging and evaluation helpers for the PaySim project.

These functions extract the shared evaluation framework from the final notebook.
They are model-agnostic: they do not train Logistic Regression, Random Forest,
or Gradient Boosting models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


DEFAULT_THRESHOLD = 0.50
THRESHOLD_GRID = np.round(np.arange(0.05, 1.00, 0.05), 2)

COMPARISON_COLUMNS = [
    "Model",
    "Threshold",
    "Precision",
    "Recall",
    "F1",
    "PR-AUC / Average Precision",
    "False Positive Rate",
    "Number Flagged",
    "Percentage Flagged",
]


def make_fraud_flags(y_score, threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
    """Convert model scores into 0/1 fraud flags at the chosen threshold."""

    return (np.asarray(y_score) >= threshold).astype(int)


def select_threshold_from_validation(
    y_true,
    y_score,
    thresholds=THRESHOLD_GRID,
) -> tuple[float, pd.DataFrame]:
    """Choose a model threshold using validation-set F1 score.

    The test set should not be used with this helper. In the notebook workflow,
    the selected threshold is chosen from validation data and then applied once
    to the untouched test set.
    """

    rows = []
    for threshold in thresholds:
        y_pred = make_fraud_flags(y_score, threshold)
        rows.append(
            {
                "Threshold": threshold,
                "Precision": precision_score(y_true, y_pred, zero_division=0),
                "Recall": recall_score(y_true, y_pred, zero_division=0),
                "F1": f1_score(y_true, y_pred, zero_division=0),
                "Number Flagged": int(y_pred.sum()),
                "Percentage Flagged": float(y_pred.mean() * 100),
            }
        )

    threshold_df = pd.DataFrame(rows)
    best_row = threshold_df.sort_values(
        by=["F1", "Recall", "Precision"],
        ascending=False,
    ).iloc[0]

    return float(best_row["Threshold"]), threshold_df


def evaluate_model(model_name, y_true, y_pred, y_score=None, threshold=None) -> dict:
    """Return a dictionary of classification metrics for one fraud detector."""

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else np.nan

    results = {
        "Model": model_name,
        "Threshold": threshold,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "False Positive Rate": false_positive_rate,
        "Number Flagged": int(np.sum(y_pred)),
        "Percentage Flagged": float(np.mean(y_pred) * 100),
        "Confusion Matrix": cm,
        "Classification Report": classification_report(y_true, y_pred, zero_division=0),
    }

    if y_score is not None:
        results["PR-AUC / Average Precision"] = average_precision_score(y_true, y_score)
    else:
        results["PR-AUC / Average Precision"] = np.nan

    return results


def evaluate_is_flagged_fraud_baseline(y_true, baseline_flag) -> dict:
    """Evaluate the original PaySim ``isFlaggedFraud`` rule as a baseline."""

    return evaluate_model(
        "Original isFlaggedFraud Baseline",
        y_true,
        baseline_flag,
        y_score=None,
        threshold="PaySim rule",
    )


def add_model_flag_columns(
    test_results_df: pd.DataFrame,
    flag_columns: dict[str, np.ndarray],
    selected_flag_column: str | None = None,
) -> pd.DataFrame:
    """Return test results with named model fraud-flag columns added.

    ``flag_columns`` should map output column names to already-created 0/1 flag
    arrays, such as flags from default and validation-selected thresholds.

    If ``selected_flag_column`` is provided, a generic ``modelFlaggedFraud``
    column is created from that named flag column.
    """

    results = test_results_df.copy()

    for column_name, flags in flag_columns.items():
        results[column_name] = flags

    if selected_flag_column is not None:
        results["modelFlaggedFraud"] = results[selected_flag_column]

    return results


def build_model_comparison(results: list[dict]) -> pd.DataFrame:
    """Create the unified model-comparison table used in the final notebook."""

    model_comparison_df = pd.DataFrame(results)
    return model_comparison_df.reindex(columns=COMPARISON_COLUMNS)
