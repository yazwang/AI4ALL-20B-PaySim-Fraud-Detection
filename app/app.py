"""
Mobile Payment Fraud Detection — Streamlit App

Deploys the Logistic Regression and Random Forest models trained in the
project notebook on the PaySim dataset. Gradient Boosting is intentionally
excluded here — the notebook marks it as a placeholder, not yet implemented.

Expects a file named `fraud_detection_bundle.pkl` in the same directory,
produced by the notebook's "Export for Deployment" cell (Section 18).
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

st.set_page_config(page_title="Mobile Payment Fraud Detection", layout="wide")

TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
REQUIRED_COLS = ["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
                  "oldbalanceDest", "newbalanceDest"]

# Order reflects the notebook's Key Findings: Random Forest was the
# strongest model on the development sample.
MODEL_LABELS = {
    "random_forest": "Random Forest",
    "logistic_regression": "Logistic Regression",
}


@st.cache_resource
def load_bundle():
    return joblib.load("fraud_detection_bundle.pkl")


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
    """Rebuilds the exact feature matrix each model was trained on. Missing
    columns (e.g. a 'type' category the training sample never saw) fall
    back to the training-set mean, matching transform_tree_features /
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

    X = pd.DataFrame(index=df.index)
    for col in feature_names:
        if col in df.columns:
            X[col] = df[col]
        else:
            X[col] = impute_means.get(col, 0)
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
    "not a validated production fraud system. Gradient Boosting is not "
    "yet implemented in the notebook and isn't available here."
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
        st.subheader("Fraud Rate by Transaction Type")
        rate = eda.get("fraud_rate_by_type", {})
        if rate:
            fig, ax = plt.subplots()
            ax.bar(rate.keys(), rate.values(), color="darkorange")
            ax.set_ylabel("Fraud rate")
            plt.xticks(rotation=30)
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

    st.subheader(f"Feature Importance — {model_label}")
    importances = bundle.get("feature_importance", {}).get(model_key)
    if importances:
        imp_df = (
            pd.Series(importances)
            .sort_values(ascending=False)
            .head(15)
            .sort_values()
        )
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.barh(imp_df.index, imp_df.values)
        ax.set_xlabel("Importance (|coefficient| for Logistic Regression)")
        st.pyplot(fig)
    else:
        st.caption("Feature importance not available for this model.")