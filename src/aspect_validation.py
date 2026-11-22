"""Compare the hand-labeled aspect sample with the model's output.

Inputs (written by scripts/export_annotation_sample.py, labeled by the author):
  data/validation/aspect_sample.csv      row_id, aspect, clause, aspect_correct, human_polarity
  data/validation/aspect_sample_key.csv  row_id, review_id, aspect, p_pos, polarity   (model output, hidden while labeling)
Output: results/metrics/aspect_validation.json
"""
import json
from pathlib import Path

import pandas as pd

from src.common.config import ROOT, load_config, results_dir

VALID_POLARITY = {"positive", "negative", "neutral"}


def validate(annotated_csv: Path, key_csv: Path) -> dict:
    a = pd.read_csv(annotated_csv)
    k = pd.read_csv(key_csv)
    if a["aspect_correct"].isna().any():
        raise ValueError("aspect_correct must be filled (0 or 1) for every row")
    a["aspect_correct"] = a["aspect_correct"].astype(int)
    if not set(a["aspect_correct"]) <= {0, 1}:
        raise ValueError("aspect_correct must be 0 or 1")
    a["human_polarity"] = a["human_polarity"].astype("string").str.strip().str.lower()
    tagged_ok = a["aspect_correct"] == 1
    pol = a.loc[tagged_ok, "human_polarity"]
    if pol.isna().any() or not set(pol.dropna()) <= VALID_POLARITY:
        raise ValueError("human_polarity must be positive/negative/neutral for every row with aspect_correct = 1")

    m = a.merge(k[["row_id", "p_pos", "polarity"]], on="row_id", how="left", validate="one_to_one")
    ok = m[m["aspect_correct"] == 1].copy()
    both_opinionated = ok[(ok["human_polarity"] != "neutral") & (ok["polarity"] != "neutral")]
    per_aspect = (m.groupby("aspect")["aspect_correct"].agg(["mean", "count"])
                    .rename(columns={"mean": "tagging_precision", "count": "n"}))
    confusion = pd.crosstab(ok["human_polarity"], ok["polarity"])
    return {
        "n_rows": int(len(m)),
        "tagging_precision": float(m["aspect_correct"].mean()),
        "tagging_precision_by_aspect": json.loads(per_aspect.to_json(orient="index")),
        "n_correctly_tagged": int(len(ok)),
        "polarity_agreement_all": float((ok["human_polarity"] == ok["polarity"]).mean()) if len(ok) else None,
        "polarity_agreement_non_neutral": (float((both_opinionated["human_polarity"] == both_opinionated["polarity"]).mean())
                                           if len(both_opinionated) else None),
        "confusion_human_vs_model": json.loads(confusion.to_json(orient="index")),
    }


def run(cfg: dict) -> None:
    val_dir = ROOT / "data" / "validation"
    out = validate(val_dir / "aspect_sample.csv", val_dir / "aspect_sample_key.csv")
    dest = results_dir(cfg) / "metrics"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "aspect_validation.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    run(load_config())
