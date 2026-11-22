import pandas as pd
import pytest

from src.baseline_sklearn import run_all, run_one
from src.benchmark import FIELDS, append_row, run_grid
from src.benchmark_summary import make_figures, summarize

CFG = {"benchmark": {"cores": [1, 2], "fractions": [0.1, 0.5, 1.0], "repetitions": 2}}


def fake_rows():
    rows = []
    for pct in (10, 50, 100):
        for cores in (1, 2):
            for rep in (0, 1):
                t = pct * 1.0 / (1 + 0.8 * (cores - 1)) + rep * 0.1
                rows.append(dict(cores=cores, pct=pct, rep=rep, rows=pct * 1000,
                                 read_s=t * 0.1, features_s=t * 0.5, train_s=t * 0.4, total_s=t))
    return pd.DataFrame(rows)


def test_grid_runs_warmup_then_every_combination(tmp_path):
    calls = []
    n = run_grid(CFG, lambda c, p, r: calls.append((c, p, r)), tmp_path / "runtime.csv")
    assert calls[0] == (2, 10, -1)                         # warm-up: most cores, smallest size, rep -1
    assert n == 2 * 3 * 2 and len(calls) == n + 1
    assert {(c, p) for c, p, r in calls[1:]} == {(c, p) for c in (1, 2) for p in (10, 50, 100)}


def test_summary_and_figures(tmp_path):
    s = summarize(fake_rows())
    one = s[(s.cores == 1) & (s.pct == 100)].iloc[0]
    two = s[(s.cores == 2) & (s.pct == 100)].iloc[0]
    assert one["speedup"] == pytest.approx(1.0)
    assert two["speedup"] > 1.0 and two["efficiency"] < 1.0
    pd.DataFrame({"pct": [10, 50], "rows": [10000, 50000], "seconds": [3.0, 20.0],
                  "peak_mb": [100, 500], "status": ["ok", "ok"]}).to_csv(tmp_path / "sk.csv", index=False)
    made = make_figures(s, tmp_path / "figs", tmp_path / "sk.csv")
    assert set(made) == {"runtime_vs_size.png", "speedup.png", "spark_vs_sklearn.png"}
    assert all((tmp_path / "figs" / m).stat().st_size > 1000 for m in made)


def test_baseline_run_one_and_run_all(tmp_path):
    texts = ["great battery love it " * 3 + str(i % 7) for i in range(300)] + \
            ["terrible broke not good " * 3 + str(i % 7) for i in range(300)]
    fake = pd.DataFrame({"text_clean": texts, "label": [1.0] * 300 + [0.0] * 300})
    original = pd.read_parquet
    pd.read_parquet = lambda path, columns=None: fake[columns]     # no pyarrow needed for this test
    try:
        r = run_one(10, tmp_path)
    finally:
        pd.read_parquet = original
    assert r["status"] == "ok" and r["rows"] == 600 and r["seconds"] >= 0 and r["peak_mb"] > 0
    out = run_all([10, 50], lambda p: {"pct": p, "rows": p, "seconds": 1.0, "peak_mb": 1.0, "status": "ok"})
    assert list(out["pct"]) == [10, 50]


def test_append_row_writes_header_once(tmp_path):
    f = tmp_path / "sub" / "runtime.csv"
    row = dict(cores=1, pct=10, rep=0, rows=1000, read_s=1.0, features_s=2.0, train_s=3.0, total_s=6.0)
    append_row(f, row); append_row(f, {**row, "rep": 1})
    df = pd.read_csv(f)
    assert list(df.columns) == FIELDS and len(df) == 2 and list(df["rep"]) == [0, 1]
