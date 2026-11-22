"""Single-machine baseline: TF-IDF + Logistic Regression in scikit-learn on the same subsets.

Spark writes the subsets to HDFS (src/make_bench_subsets.py). Copy them to the local disk first:
    mkdir -p data/bench_local
    for p in 10 25 50 100; do hdfs dfs -get /sentiment/bench/silver_$p data/bench_local/; done
Each size runs in its own process so peak memory (ru_maxrss) is measured per size.
"""
import argparse
import json
import resource
import subprocess
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

from src.common.config import ROOT, load_config, results_dir


def run_one(pct: int, source: Path) -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    warnings.filterwarnings("ignore")
    df = pd.read_parquet(source / f"silver_{pct}", columns=["text_clean", "label"])
    t0 = time.perf_counter()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=100000)
    X = vec.fit_transform(df["text_clean"])
    LogisticRegression(solver="lbfgs", max_iter=20).fit(X, df["label"])   # same iteration budget as the Spark benchmark
    seconds = time.perf_counter() - t0
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024    # Linux reports KB
    return {"pct": pct, "rows": int(len(df)), "seconds": round(seconds, 2), "peak_mb": round(peak_mb, 1), "status": "ok"}


def run_all(pcts: list[int], runner) -> pd.DataFrame:
    rows = []
    for pct in pcts:
        rows.append(runner(pct))
    return pd.DataFrame(rows)


def subprocess_runner(source: Path):
    def _run(pct: int) -> dict:
        p = subprocess.run([sys.executable, "-m", "src.baseline_sklearn", "--pct", str(pct), "--source", str(source)],
                           capture_output=True, text=True, cwd=ROOT)
        if p.returncode == 0:
            return json.loads(p.stdout.strip().splitlines()[-1])
        status = "killed (likely out of memory)" if p.returncode in (-9, 137) else "failed"
        return {"pct": pct, "rows": None, "seconds": None, "peak_mb": None, "status": status}
    return _run


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pct", type=int, default=None)
    ap.add_argument("--source", default="data/bench_local")
    a = ap.parse_args()
    src = Path(a.source) if Path(a.source).is_absolute() else ROOT / a.source
    if a.pct is not None:
        try:
            print(json.dumps(run_one(a.pct, src)))
        except MemoryError:
            print(json.dumps({"pct": a.pct, "rows": None, "seconds": None, "peak_mb": None, "status": "out of memory"}))
    else:
        cfg = load_config()
        pcts = [round(f * 100) for f in cfg["benchmark"]["fractions"]]
        out = run_all(pcts, subprocess_runner(src))
        dest = results_dir(cfg) / "benchmarks"
        dest.mkdir(parents=True, exist_ok=True)
        out.to_csv(dest / "sklearn_baseline.csv", index=False)
        print(out)
