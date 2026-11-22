"""Turn results/benchmarks/runtime.csv into a summary table and figures."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.common.config import load_config, results_dir


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    g = (raw.groupby(["cores", "pct"])
            .agg(rows=("rows", "first"), reps=("rep", "count"), read_s=("read_s", "median"),
                 features_s=("features_s", "median"), train_s=("train_s", "median"),
                 total_s=("total_s", "median"))
            .reset_index())
    g["throughput_rows_per_s"] = g["rows"] / g["total_s"]
    base_cores = g["cores"].min()
    base = g[g["cores"] == base_cores].set_index("pct")["total_s"]
    g["speedup"] = g.apply(lambda r: base[r["pct"]] / r["total_s"], axis=1)
    g["efficiency"] = g["speedup"] / (g["cores"] / base_cores)
    return g.sort_values(["pct", "cores"]).reset_index(drop=True)


def make_figures(summary: pd.DataFrame, fig_dir, sklearn_csv=None) -> list[str]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    made = []

    fig, ax = plt.subplots(figsize=(6, 4))
    for cores, d in summary.groupby("cores"):
        ax.plot(d["rows"], d["total_s"], marker="o", label=f"{cores} core(s)")
    ax.set_xlabel("rows"); ax.set_ylabel("seconds (median)"); ax.set_title("Spark runtime vs data size")
    ax.legend(); fig.tight_layout(); fig.savefig(fig_dir / "runtime_vs_size.png", dpi=150); plt.close(fig)
    made.append("runtime_vs_size.png")

    top = summary[summary["pct"] == summary["pct"].max()]
    base_cores = summary["cores"].min()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(top["cores"], top["speedup"], marker="o", label="measured")
    ax.plot(top["cores"], top["cores"] / base_cores, linestyle="--", label="ideal")
    ax.set_xlabel("cores"); ax.set_ylabel("speedup"); ax.set_title("Speedup at largest data size")
    ax.legend(); fig.tight_layout(); fig.savefig(fig_dir / "speedup.png", dpi=150); plt.close(fig)
    made.append("speedup.png")

    if sklearn_csv is not None and sklearn_csv.exists():
        sk = pd.read_csv(sklearn_csv)
        sk = sk[sk["status"] == "ok"]
        best_cores = summary["cores"].max()
        sp = summary[summary["cores"] == best_cores]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(sp["rows"], sp["total_s"], marker="o", label=f"Spark ({best_cores} cores)")
        ax.plot(sk["rows"], sk["seconds"], marker="s", label="scikit-learn")
        ax.set_xlabel("rows"); ax.set_ylabel("seconds"); ax.set_title("Spark vs scikit-learn")
        ax.legend(); fig.tight_layout(); fig.savefig(fig_dir / "spark_vs_sklearn.png", dpi=150); plt.close(fig)
        made.append("spark_vs_sklearn.png")
    return made


def run(cfg: dict) -> None:
    res = results_dir(cfg)
    raw = pd.read_csv(res / "benchmarks" / "runtime.csv")
    summary = summarize(raw)
    summary.to_csv(res / "benchmarks" / "runtime_summary.csv", index=False)
    print(make_figures(summary, res / "figures", res / "benchmarks" / "sklearn_baseline.csv"))


if __name__ == "__main__":
    run(load_config())
