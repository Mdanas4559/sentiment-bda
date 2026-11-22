import re

import yaml
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F, DataFrame

from src.common.config import load_config, data_path, ROOT
from src.common.spark_session import get_spark

CLAUSE_SPLIT = r"(?<=[.!?;])\s+|\s+(?:but|however|although|though|yet)\s+"
LEADING_CONNECTOR = r"^(?:but|however|although|though|yet)\s+"


def load_lexicon(cfg: dict) -> dict:
    with open(ROOT / cfg["aspects"]["lexicon_file"], "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def to_clauses(silver: DataFrame, cfg: dict) -> DataFrame:
    clauses = (silver.select("review_id", "parent_asin", "year", "month",
                             F.explode(F.split("text_clean", CLAUSE_SPLIT)).alias("clause"))
                     .withColumn("clause", F.regexp_replace("clause", LEADING_CONNECTOR, "")))
    return clauses.filter(F.length("clause") >= cfg["aspects"]["min_clause_chars"])


def tag_aspects(clauses: DataFrame, lexicon: dict) -> DataFrame:
    exprs = []
    for aspect, words in lexicon.items():
        pattern = r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\b"
        exprs.append(F.when(F.col("clause").rlike(pattern), F.lit(aspect)))
    tagged = clauses.withColumn("aspects", F.filter(F.array(*exprs), lambda x: x.isNotNull()))
    return (tagged.filter(F.size("aspects") > 0)
                  .withColumn("aspect", F.explode("aspects")).drop("aspects"))


def score(tagged: DataFrame, model: PipelineModel, cfg: dict) -> DataFrame:
    a = cfg["aspects"]
    scored = model.transform(tagged.withColumn("text_clean", F.col("clause")))
    scored = scored.withColumn("p_pos", vector_to_array("probability")[1])
    return (scored.select("review_id", "parent_asin", "year", "month", "aspect", "clause", "p_pos")
                  .withColumn("polarity",
                              F.when(F.col("p_pos") >= a["pos_threshold"], "positive")
                               .when(F.col("p_pos") <= a["neg_threshold"], "negative")
                               .otherwise("neutral")))


def run(cfg: dict) -> None:
    spark = get_spark("aspects", cfg)
    model = PipelineModel.load(data_path(cfg, "models", "scoring_pipeline"))
    silver = spark.read.parquet(data_path(cfg, "silver", "reviews"))
    out = score(tag_aspects(to_clauses(silver, cfg), load_lexicon(cfg)), model, cfg)
    out.write.mode("overwrite").partitionBy("year").parquet(data_path(cfg, "gold", "aspect_clauses"))
    spark.stop()


if __name__ == "__main__":
    run(load_config())
