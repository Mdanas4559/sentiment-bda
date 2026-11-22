"""Create a small committed sample from the raw files (pure Python, streams the files).

Usage:
    python scripts/make_sample.py --reviews data/raw/reviews.jsonl --meta data/raw/meta.jsonl \
        --out data/sample --n 20000
Takes a seeded random sample of about N lines (every line has the same chance, so the sample spans the
whole file and cannot be biased by any regular pattern in the file order), then keeps only the metadata
rows of the products that appear in the sample.
"""
import argparse
import json
import random
from pathlib import Path


def count_lines(path: Path) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def make_sample(reviews_in: Path, meta_in: Path | None, out_dir: Path, n: int = 20000, seed: int = 42) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    total = count_lines(reviews_in)
    p = min(1.0, n / max(total, 1))                  # expected sample size is n (give or take a few percent)
    rng = random.Random(seed)
    asins: set[str] = set()
    kept = 0
    with open(reviews_in, "r", encoding="utf-8") as fin, \
         open(out_dir / "reviews_sample.jsonl", "w", encoding="utf-8") as fout:
        for line in fin:
            if rng.random() >= p:
                continue
            fout.write(line if line.endswith("\n") else line + "\n")
            kept += 1
            try:
                asins.add(json.loads(line)["parent_asin"])
            except (ValueError, KeyError):
                pass
    meta_kept = 0
    if meta_in is not None and meta_in.exists():
        with open(meta_in, "r", encoding="utf-8") as fin, \
             open(out_dir / "meta_sample.jsonl", "w", encoding="utf-8") as fout:
            for line in fin:
                try:
                    pa = json.loads(line).get("parent_asin")
                except ValueError:
                    continue
                if pa in asins:
                    fout.write(line if line.endswith("\n") else line + "\n")
                    meta_kept += 1
    return {"reviews": kept, "products": len(asins), "meta_rows": meta_kept}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reviews", required=True)
    ap.add_argument("--meta", default=None)
    ap.add_argument("--out", default="data/sample")
    ap.add_argument("--n", type=int, default=20000)
    a = ap.parse_args()
    print(make_sample(Path(a.reviews), Path(a.meta) if a.meta else None, Path(a.out), a.n))
