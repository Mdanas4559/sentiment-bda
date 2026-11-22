import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "results" / "gold"
MAIN = ROOT / "results" / "metrics" / "main"

st.set_page_config(page_title="Product Review Sentiment (Big Data)", layout="wide")
st.title("Scalable Aspect-Based Sentiment Analysis of Product Reviews")
st.caption("Hadoop (HDFS, YARN, MapReduce) + Apache Spark pipeline. Use the pages in the sidebar for details.")


@st.cache_data
def load(name: str):
    path = GOLD / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else None


monthly = load("sentiment_by_month")
if monthly is None:
    st.info("results/gold/sentiment_by_month.csv not found. Run the pipeline (src.analytics) first.")
    st.stop()

n = monthly["n_reviews"].sum()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Reviews analysed", f"{int(n):,}")
c2.metric("Predicted positive", f"{(monthly['pct_positive_pred'] * monthly['n_reviews']).sum() / n:.1%}")
c3.metric("Average rating", f"{(monthly['avg_rating'] * monthly['n_reviews']).sum() / n:.2f}")
best = MAIN / "best_model.txt"
if best.exists() and (MAIN / f"{best.read_text().strip()}.json").exists():
    name = best.read_text().strip()
    c4.metric(f"Best model ({name})", f"macro-F1 {json.loads((MAIN / f'{name}.json').read_text())['macro_f1']:.3f}")

ratings = load("rating_vs_prediction")
if ratings is not None:
    dist = ratings.groupby("rating", as_index=False)["n"].sum()
    st.plotly_chart(px.bar(dist, x="rating", y="n", title="Rating distribution (reviews kept after cleaning)"),
                    use_container_width=True)

monthly["period"] = pd.to_datetime(dict(year=monthly["year"], month=monthly["month"], day=1))
st.plotly_chart(px.line(monthly.sort_values("period"), x="period", y="pct_positive_pred",
                        title="Share of positive reviews over time"), use_container_width=True)
