"""Download one category of Amazon Reviews 2023 from Hugging Face into data/raw/.

Usage:
    python scripts/download_data.py --category Cell_Phones_and_Accessories --max-rows 2000000

[VERIFY] the repo id and file paths on the dataset page before running; they may change.
If the network is blocked, download the two .jsonl files in a browser and place them in data/raw/.
"""
import argparse
import os
import shutil
from pathlib import Path

REPO_ID = "McAuley-Lab/Amazon-Reviews-2023"


def truncate_lines(path: Path, max_rows: int) -> int:
    """Keep only the first max_rows lines of a text file (in place). Returns lines kept."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    kept = 0
    with open(path, "r", encoding="utf-8") as fin, open(tmp, "w", encoding="utf-8") as fout:
        for line in fin:
            if kept >= max_rows:
                break
            fout.write(line)
            kept += 1
    os.replace(tmp, path)
    return kept


def download(category: str, dest: Path, max_rows: int | None) -> None:
    from huggingface_hub import hf_hub_download  # imported here so tests do not need it

    dest.mkdir(parents=True, exist_ok=True)
    wanted = {
        f"raw/review_categories/{category}.jsonl": dest / f"{category}.jsonl",
        f"raw/meta_categories/meta_{category}.jsonl": dest / f"meta_{category}.jsonl",
    }
    for remote, target in wanted.items():
        got = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=remote, local_dir=str(dest / "_hf"))
        shutil.move(got, target)
        print("downloaded", target)
    shutil.rmtree(dest / "_hf", ignore_errors=True)
    if max_rows:
        n = truncate_lines(wanted[f"raw/review_categories/{category}.jsonl"], max_rows)
        print(f"reviews truncated to first {n} lines (file order may not be random)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="Cell_Phones_and_Accessories")
    ap.add_argument("--dest", default="data/raw")
    ap.add_argument("--max-rows", type=int, default=2_000_000)
    a = ap.parse_args()
    download(a.category, Path(a.dest), a.max_rows)
