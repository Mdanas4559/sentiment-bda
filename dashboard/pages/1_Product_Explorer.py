from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

GOLD = Path(__file__).resolve().parents[2] / "results" / "gold"
st.set_page_config(page_title="Product Explorer", layout="wide")
st.title("Product Explorer")


@st.cache_data
def load(name: str):
    path = GOLD / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else None


products, aspects = load("product_summary"), load("aspect_by_product")
if products is None or aspects is None:
    st.info("results/gold/product_summary.csv or aspect_by_product.csv not found. Run the pipeline first.")
    st.stop()

products["label"] = (products["title"].fillna(products["parent_asin"]).astype(str).str.slice(0, 80)
                     + "  [" + products["parent_asin"].astype(str) + "]")
choice = st.selectbox("Product (most reviewed first)", products["label"].tolist())
p = products[products["label"] == choice].iloc[0]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Reviews", f"{int(p['n_reviews']):,}")
c2.metric("Average rating", f"{p['avg_rating']:.2f}")
c3.metric("Predicted positive", f"{p['pct_positive_pred']:.1%}")
c4.metric("Store", p["store"] if isinstance(p["store"], str) else "n/a")

a = aspects[aspects["parent_asin"] == p["parent_asin"]]
if a.empty:
    st.info("No aspect has enough mentions for this product.")
else:
    long = a.melt(id_vars=["aspect", "mentions"], value_vars=["pct_positive", "pct_negative"],
                  var_name="polarity", value_name="share")
    long["polarity"] = long["polarity"].str.replace("pct_", "", regex=False)
    fig = px.bar(long, x="aspect", y="share", color="polarity", barmode="group", hover_data=["mentions"],
                 color_discrete_map={"positive": "#2e7d32", "negative": "#c62828"},
                 title="Share of positive / negative clauses per aspect")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(a.sort_values("mentions", ascending=False), use_container_width=True)
st.caption("Aspects come from a keyword lexicon; sentiment is the Logistic Regression score of each clause. "
           "See the Models page for validation results.")
