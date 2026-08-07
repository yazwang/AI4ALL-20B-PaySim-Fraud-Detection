"""
Mobile Payment Fraud Detection — Streamlit App

Deploys the Logistic Regression, Random Forest, and XGBoost models trained in
the project notebook on the PaySim dataset. XGBoost is stored in its native
JSON format (`xgboost_model.json`) and loaded explicitly, so it does not depend
on cross-version Booster pickle compatibility.

Expects a file named `fraud_detection_bundle.pkl` in the same directory.
The bundle is generated reproducibly by `scripts/validate_xgboost.py` on the
authoritative real-PaySim pipeline with scikit-learn 1.6.1; the models are
pinned to that version in `requirements.txt` so the pickled estimators load
cleanly. The XGBoost model itself lives in `xgboost_model.json` (native
serialization, produced with xgboost==2.1.4 and loaded under the same pinned
version).
"""

import os

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

st.set_page_config(page_title="Mobile Payment Fraud Detection", layout="wide")

TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
# Types PaySim ever actually generates fraud in -- used for the fraud-rate
# chart, which is only meaningful for types where fraud occurs at all.
FRAUD_PRONE_TYPES = ["CASH_OUT", "TRANSFER"]
REQUIRED_COLS = ["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
                  "oldbalanceDest", "newbalanceDest"]

# Order reflects the notebook's Key Findings on the real-PaySim development
# sample: Random Forest was the strongest validated model, XGBoost second
# (perfect recall with slightly lower precision than RF), Logistic Regression
# third.
MODEL_LABELS = {
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "logistic_regression": "Logistic Regression",
}


@st.cache_resource
def load_bundle():
    bundle_path = os.path.join(os.path.dirname(__file__), "fraud_detection_bundle.pkl")
    bundle = joblib.load(bundle_path)
    if "xgboost" in bundle.get("models", {}):
        # XGBoost is serialized in its native JSON format (not pickled) to
        # avoid cross-version Booster incompatibility. Load it explicitly.
        xgboost_model = xgb.XGBClassifier(enable_categorical=True)
        xgboost_model.load_model(
            os.path.join(os.path.dirname(__file__), "xgboost_model.json")
        )
        bundle["models"]["xgboost"]["model"] = xgboost_model
    return bundle


bundle = load_bundle()
available_models = [k for k in MODEL_LABELS if k in bundle["models"]]


def engineer_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Mirrors Section 5 of the notebook exactly. orig_txn_count /
    dest_txn_count are intentionally NOT reproduced here — the notebook
    excludes them from the model as an account-level leakage risk, and
    they can't be computed for a single new transaction anyway."""
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


def prepare_model_input(raw_df: pd.DataFrame, model_key: str) -> pd.DataFrame:
    """Rebuilds the exact feature matrix each model was trained on.

    Two categorical-handling modes are supported, matching the notebook:
    - `uses_type_dummies=True` (Random Forest): one-hot encode 'type', same
      as transform_tree_features.
    - `categorical_mode="native"` (XGBoost): keep 'type' as a single pandas
      `category` column instead of expanding it, matching Section 13's
      `enable_categorical=True` training. `type_categories` (saved at export
      time) fixes the category set so a single new transaction — which by
      definition only has one 'type' value — still gets encoded consistently
      with what the model was trained on.

    Missing columns fall back to the training-set mean, matching
    prepare_numeric_features in the notebook."""
    info = bundle["models"][model_key]
    feature_names = info["feature_names"]
    impute_means = info.get("impute_means", {})

    df = engineer_features(raw_df)

    if info.get("uses_type_dummies", False):
        df = pd.get_dummies(df, columns=["type"])
        for t in TRANSACTION_TYPES:
            col = f"type_{t}"
            if col not in df.columns:
                df[col] = 0
    elif info.get("categorical_mode") == "native":
        type_categories = info.get("type_categories", TRANSACTION_TYPES)
        df["type"] = pd.Categorical(df["type"], categories=type_categories)

    X = pd.DataFrame(index=df.index)
    for col in feature_names:
        if col in df.columns:
            X[col] = df[col]
        else:
            X[col] = impute_means.get(col, 0)

    # Re-apply the category dtype: assigning into a fresh DataFrame
    # column-by-column above can silently reset it to plain object dtype,
    # which XGBoost's enable_categorical path won't accept.
    if info.get("categorical_mode") == "native" and "type" in X.columns:
        type_categories = info.get("type_categories", TRANSACTION_TYPES)
        X["type"] = pd.Categorical(X["type"], categories=type_categories)

    return X[feature_names]


def predict(raw_df: pd.DataFrame, model_key: str, threshold: float):
    model = bundle["models"][model_key]["model"]
    X = prepare_model_input(raw_df, model_key)
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return pred, proba


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Settings")
model_label = st.sidebar.selectbox("Model", [MODEL_LABELS[k] for k in available_models])
model_key = [k for k, v in MODEL_LABELS.items() if v == model_label][0]
model_info = bundle["models"][model_key]

default_thr = float(model_info.get("default_threshold", 0.5))
selected_thr = float(model_info.get("selected_threshold", 0.5))

threshold = st.sidebar.slider(
    "Fraud decision threshold", 0.0, 1.0, selected_thr, 0.01,
    help="Transactions with predicted fraud probability at or above this "
         "value are flagged."
)
st.sidebar.caption(
    f"Notebook default threshold: **{default_thr:.2f}** · "
    f"Validation-selected threshold: **{selected_thr:.2f}**"
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Trained on PaySim, a **synthetic** mobile-money dataset, using a "
    "stratified development sample. Predictions here are exploratory, "
    "not a validated production fraud system."
)


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------
st.title("📱 Mobile Payment Fraud Detection")
st.write(
    "Detect potentially fraudulent mobile payment transactions using models "
    "trained on the PaySim simulated dataset."
)

tab_predict, tab_data, tab_perf = st.tabs(
    ["🔍 Predict", "📊 Dataset Insights", "📈 Model Performance"]
)

# ---------------------------------------------------------------------------
# Tab 1: Predict
# ---------------------------------------------------------------------------
with tab_predict:
    mode = st.radio("Input method", ["Single transaction", "Upload CSV"], horizontal=True)

    if mode == "Single transaction":
        with st.form("single_txn_form"):
            c1, c2 = st.columns(2)
            with c1:
                step = st.number_input("Step (simulation hour)", min_value=0, value=1)
                txn_type = st.selectbox("Transaction type", TRANSACTION_TYPES)
                amount = st.number_input("Amount", min_value=0.0, value=1000.0)
            with c2:
                oldbalanceOrg = st.number_input("Sender balance before", min_value=0.0, value=5000.0)
                newbalanceOrig = st.number_input("Sender balance after", min_value=0.0, value=4000.0)
                oldbalanceDest = st.number_input("Recipient balance before", min_value=0.0, value=0.0)
                newbalanceDest = st.number_input("Recipient balance after", min_value=0.0, value=1000.0)
            submitted = st.form_submit_button("Check transaction")

        if submitted:
            row = pd.DataFrame([{
                "step": step, "type": txn_type, "amount": amount,
                "oldbalanceOrg": oldbalanceOrg, "newbalanceOrig": newbalanceOrig,
                "oldbalanceDest": oldbalanceDest, "newbalanceDest": newbalanceDest,
            }])
            pred, proba = predict(row, model_key, threshold)
            fraud_prob = proba[0]

            st.subheader("Result")
            col_a, col_b = st.columns([1, 2])
            with col_a:
                if pred[0] == 1:
                    st.error("⚠️ Flagged as FRAUD")
                else:
                    st.success("✅ Not flagged as fraud")
                st.metric("Fraud probability", f"{fraud_prob:.1%}")
            with col_b:
                st.progress(min(float(fraud_prob), 1.0))
                eng = engineer_features(row).iloc[0]
                st.caption("Signals the model saw:")
                st.write({
                    "Sender account emptied": bool(eng["orig_account_emptied"]),
                    "Sender balance mismatch": round(float(eng["orig_balance_error"]), 2),
                    "Recipient balance mismatch": round(float(eng["dest_balance_error"]), 2),
                    "Hour of day (from step)": int(eng["hour_of_day"]),
                })

    else:
        st.write("CSV must include columns: " + ", ".join(REQUIRED_COLS))
        uploaded = st.file_uploader("Upload transactions CSV", type="csv")
        if uploaded is not None:
            batch = pd.read_csv(uploaded)
            missing = [c for c in REQUIRED_COLS if c not in batch.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                pred, proba = predict(batch, model_key, threshold)
                out = batch.copy()
                out["fraud_probability"] = proba
                out["predicted_fraud"] = pred

                n_flagged = int(pred.sum())
                st.write(
                    f"**{n_flagged:,}** of **{len(out):,}** transactions flagged "
                    f"({n_flagged / len(out):.2%}) at threshold {threshold:.2f}."
                )
                st.dataframe(out, use_container_width=True)
                st.download_button(
                    "Download predictions as CSV",
                    out.to_csv(index=False).encode("utf-8"),
                    "fraud_predictions.csv",
                    "text/csv",
                )

                if "isFraud" in batch.columns:
                    st.subheader("Accuracy on this upload (ground truth provided)")
                    y_true = batch["isFraud"]
                    p = precision_score(y_true, pred, zero_division=0)
                    r = recall_score(y_true, pred, zero_division=0)
                    f1 = f1_score(y_true, pred, zero_division=0)
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Precision", f"{p:.3f}")
                    m2.metric("Recall", f"{r:.3f}")
                    m3.metric("F1-score", f"{f1:.3f}")
                    cm = confusion_matrix(y_true, pred)
                    st.write("Confusion matrix:")
                    st.write(pd.DataFrame(
                        cm,
                        index=["Actual: Not Fraud", "Actual: Fraud"],
                        columns=["Pred: Not Fraud", "Pred: Fraud"],
                    ))


# ---------------------------------------------------------------------------
# Tab 2: Dataset Insights
# ---------------------------------------------------------------------------
with tab_data:
    eda = bundle.get("eda", {})

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Class Distribution")
        dist = eda.get("class_distribution", {})
        if dist:
            fig, ax = plt.subplots()
            labels = ["Not Fraud" if k == 0 else "Fraud" for k in dist.keys()]
            ax.bar(labels, dist.values(), color=["steelblue", "crimson"])
            ax.set_ylabel("Number of transactions")
            for i, v in enumerate(dist.values()):
                ax.text(i, v, f"{v:,}", ha="center", va="bottom")
            st.pyplot(fig)

    with c2:
        st.subheader("Fraudulent Transactions by Percent Risk")
        rate = eda.get("fraud_rate_by_type", {})
        if rate:
            # Only CASH_OUT/TRANSFER ever carry fraud in PaySim -- restricting
            # to those two (rather than all 5 types) is what makes this chart
            # about *risk*, not just raw fraud counts diluted across types
            # that never see any fraud at all.
            fraud_prone = {t: rate[t] * 100 for t in FRAUD_PRONE_TYPES if t in rate}
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.barplot(x=list(fraud_prone.keys()), y=list(fraud_prone.values()),
                        hue=list(fraud_prone.keys()), palette="viridis", legend=False, ax=ax)
            ax.set_title("Fraudulent Transactions by Percent Risk")
            ax.set_xlabel("Transaction Type")
            ax.set_ylabel("Percent Risk")
            st.pyplot(fig)

    st.caption(
        "PaySim is a synthetic simulator, so these patterns — like fraud "
        "concentrated in TRANSFER/CASH_OUT — reflect how the simulator "
        "generates fraud, and may not generalize to real payment platforms."
    )


# ---------------------------------------------------------------------------
# Tab 3: Model Performance
# ---------------------------------------------------------------------------
with tab_perf:
    st.subheader("Model Comparison")
    st.caption(
        "Includes the original isFlaggedFraud rule as a baseline, plus each "
        "model at both the notebook default threshold (0.50) and its "
        "validation-selected threshold — exactly as produced by the "
        "notebook's model_comparison_df."
    )
    comparison = bundle.get("model_comparison", [])
    if comparison:
        st.dataframe(
            pd.DataFrame(comparison).set_index("Model").round(4),
            use_container_width=True,
        )

    # ============================================================
    # ADD-ON VISUAL: Baseline vs. Best Model Callout
    # To remove this visual: delete this entire block, from this banner
    # down to the matching "END ADD-ON" banner below. Nothing else in the
    # app depends on it -- it only reads bundle["model_comparison"], which
    # is used elsewhere too and is untouched by removing this.
    # ============================================================
    def render_baseline_vs_best_callout():
        comp = bundle.get("model_comparison")
        if not comp:
            return
        comp_df = pd.DataFrame(comp)
        if "Model" not in comp_df.columns or "Recall" not in comp_df.columns:
            return
        is_baseline = comp_df["Model"].str.contains("isFlaggedFraud", case=False, na=False)
        baseline_rows, model_rows = comp_df[is_baseline], comp_df[~is_baseline]
        if baseline_rows.empty or model_rows.empty:
            return
        baseline = baseline_rows.iloc[0]
        sort_cols = [c for c in ["Recall", "PR-AUC / Average Precision"] if c in model_rows.columns]
        best = model_rows.sort_values(by=sort_cols, ascending=False).iloc[0]

        recall_gain = (best["Recall"] - baseline["Recall"]) * 100
        st.success(
            f"**Baseline vs. best model:** **{best['Model']}** caught "
            f"**{best['Recall']:.0%}** of fraud cases, vs. **{baseline['Recall']:.0%}** "
            f"for the original `isFlaggedFraud` rule -- a **{recall_gain:+.0f} point** "
            f"difference in recall. False positive rate: "
            f"{best.get('False Positive Rate', float('nan')):.4f} (model) vs. "
            f"{baseline.get('False Positive Rate', float('nan')):.4f} (rule)."
        )

    render_baseline_vs_best_callout()
    # ============================================================
    # END ADD-ON: Baseline vs. Best Model Callout
    # ============================================================

    st.subheader(f"Confusion Matrices — {model_label}")
    cms = bundle.get("confusion_matrices", {}).get(model_key, {})
    col_default, col_selected = st.columns(2)

    def draw_cm(container, cm, title):
        if cm is None:
            return
        cm = np.array(cm)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(cm, cmap="Blues")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Not Fraud", "Fraud"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Not Fraud", "Fraud"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        thresh = cm.max() / 2 if cm.max() > 0 else 1
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")
        container.pyplot(fig)

    draw_cm(col_default, cms.get("default"), f"Threshold = {default_thr:.2f} (default)")
    draw_cm(col_selected, cms.get("selected"), f"Threshold = {selected_thr:.2f} (validation-selected)")

    st.subheader("Precision-Recall Curve Comparison")
    pr_data = bundle.get("pr_curve_data", {})
    pr_auc_lookup = bundle.get("pr_auc", {})
    if pr_data:
        fig, ax = plt.subplots()
        for k in available_models:
            if k in pr_data:
                precision, recall = pr_data[k]["precision"], pr_data[k]["recall"]
                pr_auc = pr_auc_lookup.get(k, float("nan"))
                ax.plot(recall, precision, label=f"{MODEL_LABELS[k]} (PR-AUC={pr_auc:.3f})")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.legend(); ax.grid(alpha=0.3)
        st.pyplot(fig)

    # ============================================================
    # ADD-ON VISUAL: Threshold Tradeoff Chart
    # To remove this visual: delete this entire block, from this banner
    # down to the matching "END ADD-ON" banner below.
    # Requires bundle["threshold_tables"][model_key] -- if that key is
    # missing (e.g. an older bundle), this silently renders nothing.
    # ============================================================
    def render_threshold_tradeoff_chart():
        table = bundle.get("threshold_tables", {}).get(model_key)
        if not table:
            return
        st.subheader(f"Threshold Tradeoff — {model_label}")
        st.caption(
            "Precision, recall, and F1 across the full threshold grid the "
            "notebook searched when picking the validation-selected "
            "threshold -- not just the single point that got chosen."
        )
        tdf = pd.DataFrame(table)
        fig, ax = plt.subplots()
        ax.plot(tdf["Threshold"], tdf["Precision"], label="Precision", marker="o", markersize=3)
        ax.plot(tdf["Threshold"], tdf["Recall"], label="Recall", marker="o", markersize=3)
        ax.plot(tdf["Threshold"], tdf["F1"], label="F1", marker="o", markersize=3)
        ax.axvline(selected_thr, color="gray", linestyle="--", alpha=0.6,
                   label=f"Validation-selected ({selected_thr:.2f})")
        ax.set_xlabel("Threshold"); ax.set_ylabel("Score")
        ax.legend(); ax.grid(alpha=0.3)
        st.pyplot(fig)

    render_threshold_tradeoff_chart()
    # ============================================================
    # END ADD-ON: Threshold Tradeoff Chart
    # ============================================================

    st.subheader(f"{model_label} - Top Feature Importances")
    importances = bundle.get("feature_importance", {}).get(model_key)
    if importances:
        imp_df = (
            pd.Series(importances)
            .sort_values(ascending=False)
            .head(15)
            .sort_values()
        )
        # xlabel depends on what kind of "importance" this actually is --
        # was previously hardcoded to describe Logistic Regression's
        # coefficients even when showing Random Forest or XGBoost.
        xlabel_by_model = {
            "logistic_regression": "Importance (|coefficient|)",
            "random_forest": "Importance (mean decrease in impurity)",
            "xgboost": "Importance (gain)",
        }
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(imp_df.index, imp_df.values)
        ax.set_xlabel(xlabel_by_model.get(model_key, "Importance"))
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.caption("Feature importance not available for this model.")

    # ============================================================
    # ADD-ON VISUAL: Engineered Feature Correlation Heatmap
    # To remove this visual: delete this entire block, from this banner
    # down to the matching "END ADD-ON" banner below.
    # Requires bundle["feature_correlation"] -- missing key = no render.
    # ============================================================
    def render_feature_correlation_heatmap():
        corr_info = bundle.get("feature_correlation")
        if not corr_info:
            return
        st.subheader("Feature Correlation (Why These Features Were Engineered)")
        st.caption(
            "Computed on training data only. High correlation between a raw "
            "column and an engineered feature built from it (e.g. amount vs. "
            "amount_log) is expected -- it's correlations BETWEEN different "
            "engineered features that are worth checking for redundancy."
        )
        columns = corr_info["columns"]
        matrix = np.array(corr_info["matrix"])
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax, label="Correlation")
        ax.set_xticks(range(len(columns))); ax.set_xticklabels(columns, rotation=90)
        ax.set_yticks(range(len(columns))); ax.set_yticklabels(columns)
        plt.tight_layout()
        st.pyplot(fig)

    render_feature_correlation_heatmap()
    # ============================================================
    # END ADD-ON: Engineered Feature Correlation Heatmap
    # ============================================================

    # ============================================================
    # ADD-ON VISUAL: Amount Distribution by Fraud vs. Non-Fraud
    # To remove this visual: delete this entire block, from this banner
    # down to the matching "END ADD-ON" banner below.
    # Requires bundle["amount_distribution"] -- missing key = no render.
    # ============================================================
    def render_amount_distribution():
        dist = bundle.get("amount_distribution")
        if not dist:
            return
        st.subheader("Transaction Amount Distribution by Class")
        st.caption(
            "Log-scale x-axis, each class shown as a density (not raw count) "
            "so the much smaller fraud class stays visible next to the much "
            "larger non-fraud class. This is the shape amount_log was "
            "engineered to help the models handle."
        ) 
        edges = np.array(dist["bin_edges"])
        centers = (edges[:-1] + edges[1:]) / 2
        non_fraud = np.array(dist["non_fraud_counts"], dtype=float)
        fraud = np.array(dist["fraud_counts"], dtype=float)
        non_fraud_density = non_fraud / non_fraud.sum() if non_fraud.sum() > 0 else non_fraud
        fraud_density = fraud / fraud.sum() if fraud.sum() > 0 else fraud

        fig, ax = plt.subplots()
        ax.plot(centers, non_fraud_density, label="Not Fraud", drawstyle="steps-mid")
        ax.plot(centers, fraud_density, label="Fraud", drawstyle="steps-mid", color="crimson")
        ax.set_xscale("log")
        ax.set_xlabel("Amount (log scale)"); ax.set_ylabel("Density")
        ax.legend()
        st.pyplot(fig)

    render_amount_distribution()
    # ============================================================
    # END ADD-ON: Amount Distribution by Fraud vs. Non-Fraud
    # ============================================================