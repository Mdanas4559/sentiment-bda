"""Orchestrates the scalability benchmark: one fresh Spark process per (cores, size, repetition)."""
import csv
import subprocess
import sys
from pathlib import Path
from typing import Callable

from src.common.config import ROOT, load_config, results_dir

Runner = Callable[[int, int, int], None]
FIELDS = ["cores", "pct", "rep", "rows", "read_s", "features_s", "train_s", "total_s"]


def append_row(path: Path, row: dict) -> None:
    """Append one measurement to runtime.csv, writing the header the first time."""
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            w.writeheader()
        w.writerow(row)


def spark_runner(cores: int, pct: int, rep: int) -> None:
    subprocess.run(
        [sys.executable, "-m", "src.benchmark_run", "--cores", str(cores), "--pct", str(pct), "--rep", str(rep)],
        check=True, cwd=ROOT,
    )


def run_grid(cfg: dict, runner: Runner, out_csv: Path) -> int:
    """Run the full grid. rep = -1 is a warm-up run that benchmark_run must not record. Returns #measured runs."""
    b = cfg["benchmark"]
    pcts = [round(f * 100) for f in b["fractions"]]
    if out_csv.exists():
        out_csv.unlink()
    runner(max(b["cores"]), min(pcts), -1)                      # warm-up (JVM, file cache), not recorded
    n = 0
    for pct in pcts:
        for cores in b["cores"]:
            for rep in range(b["repetitions"]):
                runner(cores, pct, rep)
                n += 1
    return n


if __name__ == "__main__":
    cfg = load_config()
    out = results_dir(cfg) / "benchmarks" / "runtime.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    print("measured runs:", run_grid(cfg, spark_runner, out))
