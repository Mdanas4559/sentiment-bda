from pyspark.sql import functions as F, DataFrame
from src.common.config import load_config, data_path, results_dir
from src.common.spark_session import get_spark

HTML = r"<[^>]+>"
URL = r"(https?://\S+|www\.\S+)"
APOSTROPHES = "[\u2018\u2019\u00b4`]"          # typographic apostrophes -> '  (so "don’t" stays "don't")
ALLOWED = r"[^a-z0-9'.,!?;\s-]"                # anything else (emoji, symbols, quotes) is replaced by a space


def normalize_text(col):
    c = F.lower(col)
    c = F.regexp_replace(c, APOSTROPHES, "'")   # must run BEFORE the character filter
    c = F.regexp_replace(c, HTML, " ")
    c = F.regexp_replace(c, URL, " ")
    c = F.regexp_replace(c, ALLOWED, " ")
    c = F.regexp_replace(c, r"\s+", " ")
    return F.trim(c)


def add_label(df: DataFrame) -> DataFrame:
    """Binary label: rating >= 4 -> 1.0 (positive), rating <= 2 -> 0.0 (negative); 3-star reviews are dropped."""
    df = df.filter(F.col("rating") != 3.0)
    return df.withColumn("label", F.when(F.col("rating") >= 4.0, 1.0).otherwise(0.0))


def clean(df: DataFrame, cfg: dict) -> DataFrame:
    c = cfg["clean"]
    df = df.filter(F.col("text").isNotNull() & F.col("rating").between(1.0, 5.0) & F.col("review_ts").isNotNull())
    df = df.withColumn("full_text", F.concat_ws(". ", F.col("title"), F.col("text")))
    ascii_ratio = (F.length(F.regexp_replace("full_text", r"[^\x00-\x7F]", ""))
                   / F.greatest(F.length("full_text"), F.lit(1)))
    df = df.filter(ascii_ratio >= c["min_ascii_ratio"])
    df = df.withColumn("text_clean", normalize_text(F.col("full_text")))
    df = df.withColumn("word_count", F.size(F.split("text_clean", r"\s+")))
    df = df.filter(F.col("word_count").between(c["min_words"], c["max_words"]))
    df = df.withColumn("text_hash", F.sha2("text_clean", 256))
    df = df.dropDuplicates(["parent_asin", "text_hash"])
    df = add_label(df)
    return df.select("review_id", "parent_asin", "user_id", "rating", "label", "text_clean",
                     "word_count", "helpful_vote", "verified_purchase", "review_ts", "year", "month")


def run(cfg: dict) -> None:
    spark = get_spark("clean", cfg)
    bronze = spark.read.parquet(data_path(cfg, "bronze", "reviews"))
    clean(bronze, cfg).write.mode("overwrite").partitionBy("year").parquet(data_path(cfg, "silver", "reviews"))
    n_in = bronze.count()
    n_out = spark.read.parquet(data_path(cfg, "silver", "reviews")).count()
    out = results_dir(cfg) / "data_quality"
    out.mkdir(parents=True, exist_ok=True)
    (out / "cleaning_funnel.csv").write_text(f"stage,rows\nbronze,{n_in}\nsilver,{n_out}\n")
    spark.stop()


if __name__ == "__main__":
    run(load_config())
