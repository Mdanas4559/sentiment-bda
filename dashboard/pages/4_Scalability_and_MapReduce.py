from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

BENCH = Path(__file__).resolve().parents[2] / "results" / "benchmarks"
st.set_page_config(page_title="Scalability and MapReduce", layout="wide")
st.title("Scalability and MapReduce")
st.caption("Single-node setup: 'cores' are threads on one machine (Spark local mode), not machines in a cluster.")


def read(name: str):
    p = BENCH / name
    return pd.read_csv(p) if p.exists() else None


summary = read("runtime_summary.csv")
if summary is None:
    st.info("runtime_summary.csv not found. Run src.benchmark and src.benchmark_summary.")
else:
    summary["cores"] = summary["cores"].astype(str)
    st.plotly_chart(px.line(summary, x="rows", y="total_s", color="cores", markers=True,
                            title="Spark runtime vs data size"), use_container_width=True)
    top = summary[summary["pct"] == summary["pct"].max()].copy()
    top["cores_n"] = top["cores"].astype(int)
    base = top["cores_n"].min()
    fig = px.line(top, x="cores_n", y="speedup", markers=True, title="Speedup at the largest data size")
    fig.add_scatter(x=top["cores_n"], y=top["cores_n"] / base, mode="lines", name="ideal", line=dict(dash="dash"))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(summary.round(2), use_container_width=True)

sk = read("sklearn_baseline.csv")
if sk is not None:
    st.subheader("Single-machine baseline (scikit-learn)")
    st.dataframe(sk, use_container_width=True)

mr = read("mr_vs_spark.csv")
if mr is not None:
    st.subheader("MapReduce (Hadoop Streaming on YARN) vs Spark, same input")
    st.dataframe(mr, use_container_width=True)
    st.caption("MapReduce time includes job submission and container start-up; Spark time excludes JVM start-up. "
               "Both produced identical results (column 'identical').")
