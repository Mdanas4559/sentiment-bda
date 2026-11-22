"""Export ~150 random tagged clauses for hand labeling.

Run from the repo root (it imports the src package):  python -m scripts.export_annotation_sample

Writes two files into data/validation/:
  aspect_sample.csv      -> what the annotator sees (NO model output, so labels are not biased)
  aspect_sample_key.csv  -> the model's p_pos / polarity for the same row_id (used by src/aspect_validation.py)
"""
import math

from pyspark.sql import Window, functions as F

from src.common.config import ROOT, load_config, data_path
from src.common.spark_session import get_spark
from src.aspects import load_lexicon


def run(cfg: dict, total: int = 150) -> None:
    spark = get_spark("annotation_export", cfg)
    df = spark.read.parquet(data_path(cfg, "gold", "aspect_clauses"))
    per_aspect = math.ceil(total / len(load_lexicon(cfg)))
    frac = min(1.0, 100000 / max(df.count(), 1))                      # cheap pre-sample so the window sort stays small
    w = Window.partitionBy("aspect").orderBy(F.rand(cfg["project"]["seed"]))
    sample = (df.sample(False, frac, cfg["project"]["seed"])
                .withColumn("rn", F.row_number().over(w)).filter(F.col("rn") <= per_aspect).drop("rn"))
    pdf = (sample.select("review_id", "aspect", "clause", "p_pos", "polarity").toPandas()
                 .sample(frac=1.0, random_state=cfg["project"]["seed"]).reset_index(drop=True))
    pdf.insert(0, "row_id", range(1, len(pdf) + 1))
    out = ROOT / "data" / "validation"
    out.mkdir(parents=True, exist_ok=True)
    pdf[["row_id", "aspect", "clause"]].assign(aspect_correct="", human_polarity="").to_csv(out / "aspect_sample.csv", index=False)
    pdf[["row_id", "review_id", "aspect", "p_pos", "polarity"]].to_csv(out / "aspect_sample_key.csv", index=False)
    print(f"exported {len(pdf)} rows to {out}")
    spark.stop()


if __name__ == "__main__":
    run(load_config())
