from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

GOLD = Path(__file__).resolve().parents[2] / "results" / "gold"
st.set_page_config(page_title="Aspect Insights", layout="wide")
st.title("Aspect Insights")

path = GOLD / "aspect_trend.csv"
if not path.exists():
    st.info("results/gold/aspect_trend.csv not found. Run the pipeline first.")
    st.stop()
trend = pd.read_csv(path)

trend["w_neg"] = trend["pct_negative"] * trend["mentions"]
overall = trend.groupby("aspect", as_index=False).agg(mentions=("mentions", "sum"), w_neg=("w_neg", "sum"))
overall["pct_negative"] = overall["w_neg"] / overall["mentions"]

left, right = st.columns(2)
left.plotly_chart(px.bar(overall.sort_values("mentions"), x="mentions", y="aspect", orientation="h",
                         title="How often each aspect is mentioned"), use_container_width=True)
fig = px.bar(overall.sort_values("pct_negative"), x="pct_negative", y="aspect", orientation="h",
             title="Share of negative clauses per aspect", color_discrete_sequence=["#c62828"])
fig.update_xaxes(tickformat=".0%")
right.plotly_chart(fig, use_container_width=True)

trend["period"] = trend["year"].astype(int).astype(str) + "-Q" + trend["quarter"].astype(int).astype(str)
chosen = st.multiselect("Aspects to compare over time", sorted(trend["aspect"].unique()),
                        default=sorted(trend["aspect"].unique())[:3])
sub = trend[trend["aspect"].isin(chosen)].sort_values("period")
if not sub.empty:
    line = px.line(sub, x="period", y="pct_negative", color="aspect", markers=True,
                   title="Share of negative clauses by quarter")
    line.update_yaxes(tickformat=".0%")
    st.plotly_chart(line, use_container_width=True)
st.caption("Quarters with very few mentions are noisy; check the mentions column before drawing conclusions.")
