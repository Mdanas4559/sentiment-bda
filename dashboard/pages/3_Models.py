import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

METRICS = Path(__file__).resolve().parents[2] / "results" / "metrics"
MODEL_FILES = ["logreg", "naive_bayes", "linear_svc", "random_forest"]
st.set_page_config(page_title="Models", layout="wide")
st.title("Model comparison")

experiments = sorted(d.name for d in METRICS.iterdir() if d.is_dir()) if METRICS.exists() else []
if not experiments:
    st.info("results/metrics is empty. Run src.train first.")
    st.stop()
exp = st.selectbox("Experiment", experiments, index=experiments.index("main") if "main" in experiments else 0)
folder = METRICS / exp

rows, results = [], {}
for name in MODEL_FILES:
    f = folder / f"{name}.json"
    if f.exists():
        m = json.loads(f.read_text())
        results[name] = m
        rows.append({"model": name, "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                     "macro_precision": m["macro_precision"], "macro_recall": m["macro_recall"],
                     "roc_auc": m.get("roc_auc"), "fit_seconds": m.get("fit_seconds")})
if rows:
    st.dataframe(pd.DataFrame(rows).round(4), use_container_width=True)
    st.caption("Random Forest is trained on a small subsample with a small vocabulary (see the README), "
               "so it is not directly comparable to the linear models.")
    pick = st.selectbox("Confusion matrix for", list(results))
    cm = results[pick]["confusion_matrix"]
    names = ["negative" if l == 0.0 else "positive" for l in cm["labels"]]
    st.plotly_chart(px.imshow(cm["matrix"], x=names, y=names, text_auto=True,
                              labels=dict(x="predicted", y="actual", color="reviews"),
                              title=f"Confusion matrix: {pick}"), use_container_width=True)

top = folder / "top_features.csv"
if top.exists():
    tf = pd.read_csv(top)
    st.subheader("Most influential terms (Logistic Regression)")
    left, right = st.columns(2)
    left.dataframe(tf[tf["polarity"] == "positive"].head(20), use_container_width=True)
    right.dataframe(tf[tf["polarity"] == "negative"].head(20), use_container_width=True)

val = METRICS / "aspect_validation.json"
if val.exists():
    st.subheader("Aspect validation (hand-labeled sample)")
    v = json.loads(val.read_text())
    c1, c2, c3 = st.columns(3)
    c1.metric("Labeled clauses", v["n_rows"])
    c2.metric("Tagging precision", f"{v['tagging_precision']:.1%}")
    agree = v["polarity_agreement_non_neutral"]
    c3.metric("Polarity agreement (non-neutral)", "n/a" if agree is None else f"{agree:.1%}")
