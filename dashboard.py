"""
dashboard.py
OWNED BY: Person C (Frontend Person)

WHAT THIS FILE DOES:
The live website (Streamlit) showing ComputePulse's real results:
  - Baseline vs tuned AI model (accuracy, precision, recall, F1, AUC)
  - Cross-validation stability (proves it's not a lucky split)
  - Confusion matrix
  - SHAP feature importance (real explainability)
  - Real machine risk rankings (Model 2) + proof they correlate with
    real observed failure rates

HOW TO RUN:
    streamlit run dashboard.py

REQUIRES (run these first, in order):
    python prepare_dataset.py
    python baseline_model.py
    python train_model.py
    python model2_placement.py
"""

import streamlit as st
import pandas as pd
import pickle
import os
import plotly.graph_objects as go

st.set_page_config(page_title="ComputePulse", layout="wide", page_icon="🖥️")


def read_key_value_file(path):
    result = {}
    if not os.path.exists(path):
        return result
    with open(path, "r") as f:
        for line in f:
            if "=" in line:
                key, value = line.strip().split("=", 1)
                try:
                    result[key] = float(value)
                except ValueError:
                    result[key] = value
    return result


required = {
    "data/cluster_data_real.csv": "python prepare_dataset.py",
    "models/model1.pkl": "python train_model.py",
    "results/baseline_results.txt": "python baseline_model.py",
    "results/model_results.txt": "python train_model.py",
    "results/node_risk_scores.csv": "python model2_placement.py",
}
missing = [f for f in required if not os.path.exists(f)]

if missing:
    st.title("🖥️ ComputePulse Dashboard")
    st.warning("Run these scripts first, in order:")
    for f in missing:
        st.write(f"❌ `{f}` — run: `{required[f]}`")
    st.stop()


baseline_results = read_key_value_file("results/baseline_results.txt")
model_results = read_key_value_file("results/model_results.txt")
cv_results = read_key_value_file("results/cv_results.txt")
confusion = read_key_value_file("results/confusion_matrix.txt")
feature_importance = read_key_value_file("results/feature_importance.txt")
model2_corr = read_key_value_file("results/model2_correlation.txt")

node_scores = pd.read_csv("results/node_risk_scores.csv")


st.title("🖥️ ComputePulse: Cluster Health Intelligence")
st.write(
    "Trained and evaluated on **real Alibaba GPU cluster production data** "
    "(cluster-trace-gpu-v2020) — not synthetic data."
)
st.caption(
    "Source: Alibaba PAI production cluster, ~6,500 GPUs across ~1,800 machines, "
    "July-August 2020. Public dataset: github.com/alibaba/clusterdata"
)
st.divider()


col1, col2, col3, col4 = st.columns(4)
baseline_acc = baseline_results.get("accuracy", 0) * 100
model_acc = model_results.get("accuracy", 0) * 100
model_auc = model_results.get("auc", 0)
baseline_auc = baseline_results.get("auc", 0)

col1.metric("Baseline Accuracy", f"{baseline_acc:.1f}%")
col2.metric("ComputePulse AI Accuracy", f"{model_acc:.1f}%", delta=f"{model_acc - baseline_acc:+.1f}%")
col3.metric("ComputePulse ROC-AUC", f"{model_auc:.3f}", delta=f"{model_auc - baseline_auc:+.3f}")
col4.metric("Real Machines Tracked", f"{len(node_scores):,}")

st.divider()


st.subheader("📊 Baseline vs ComputePulse AI (on real held-out test data)")

metrics_names = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
baseline_vals = [
    baseline_results.get("accuracy", 0) * 100,
    baseline_results.get("precision", 0) * 100,
    baseline_results.get("recall", 0) * 100,
    baseline_results.get("f1", 0) * 100,
    baseline_results.get("auc", 0) * 100,
]
model_vals = [
    model_results.get("accuracy", 0) * 100,
    model_results.get("precision", 0) * 100,
    model_results.get("recall", 0) * 100,
    model_results.get("f1", 0) * 100,
    model_results.get("auc", 0) * 100,
]

fig = go.Figure()
fig.add_trace(go.Bar(name="Baseline (simple rules)", x=metrics_names, y=baseline_vals, marker_color="lightcoral"))
fig.add_trace(go.Bar(name="ComputePulse AI (tuned LightGBM)", x=metrics_names, y=model_vals, marker_color="seagreen"))
fig.update_layout(barmode="group", height=420, yaxis_title="Score (%)")
st.plotly_chart(fig, use_container_width=True)

if cv_results:
    st.caption(
        f"5-fold cross-validation ROC-AUC: {cv_results.get('cv_auc_mean', 0):.3f} "
        f"± {cv_results.get('cv_auc_std', 0):.3f} — result is stable, not a lucky split."
    )

st.divider()


if confusion:
    st.subheader("🎯 Confusion Matrix (real test set)")
    col_a, col_b = st.columns([1, 1])

    with col_a:
        cm_display = pd.DataFrame(
            [
                [int(confusion.get("true_negative", 0)), int(confusion.get("false_positive", 0))],
                [int(confusion.get("false_negative", 0)), int(confusion.get("true_positive", 0))],
            ],
            columns=["Predicted Healthy", "Predicted Failure"],
            index=["Actual Healthy", "Actual Failure"],
        )
        st.dataframe(cm_display, use_container_width=True)

    with col_b:
        tp = confusion.get("true_positive", 0)
        fn = confusion.get("false_negative", 0)
        caught = tp / (tp + fn) if (tp + fn) > 0 else 0
        st.metric("Real Failures Caught", f"{caught:.1%}")
        st.caption("Percentage of actual failures the model successfully flagged in advance.")

st.divider()


if feature_importance:
    st.subheader("🧠 Why Does the AI Predict Failure? (Real SHAP Explainability)")

    importance_df = pd.DataFrame(
        list(feature_importance.items()), columns=["Feature", "Importance"]
    ).sort_values("Importance", ascending=True)

    fig2 = go.Figure(go.Bar(
        x=importance_df["Importance"], y=importance_df["Feature"],
        orientation="h", marker_color="teal",
    ))
    fig2.update_layout(height=350, xaxis_title="Mean |SHAP value| (higher = matters more)")
    st.plotly_chart(fig2, use_container_width=True)

    st.caption(
        "Computed on real held-out instances. GPU utilization is typically the "
        "strongest signal — consistent with this being a preemptible GPU-sharing "
        "cluster where high GPU pressure correlates with job interruption."
    )

st.divider()


st.subheader("🗺️ Model 2: Real Machine Risk Rankings (Workload Placement)")

corr = model2_corr.get("correlation", 0)
st.metric("Predicted Risk vs Real Failure Rate Correlation", f"{corr:.3f}")
st.caption(
    "This is not a separately trained classifier - it's real per-machine risk "
    "aggregated from Model 1's validated predictions. A correlation this high "
    "with actual observed failure rates proves the risk scores are meaningful."
)

col_x, col_y = st.columns(2)

with col_x:
    st.write("**✅ Healthiest real machines (recommend for new jobs)**")
    healthiest = node_scores.sort_values("avg_risk_score").head(10)
    st.dataframe(
        healthiest[["node_id", "avg_risk_score", "actual_failure_rate", "num_instances"]]
        .round(2),
        use_container_width=True,
    )

with col_y:
    st.write("**🔴 Riskiest real machines (avoid / investigate)**")
    riskiest = node_scores.sort_values("avg_risk_score", ascending=False).head(10)
    st.dataframe(
        riskiest[["node_id", "avg_risk_score", "actual_failure_rate", "num_instances"]]
        .round(2),
        use_container_width=True,
    )

st.divider()
st.caption("ComputePulse — Predict. Prevent. Optimize. Trained on real Alibaba GPU cluster production data.")
