from pyspark.sql import functions as F
from pyspark.sql.types import (StructType, StructField, StringType, DoubleType,
                               LongType, IntegerType, BooleanType)
from src.common.config import load_config, data_path
from src.common.spark_session import get_spark

RAW_SCHEMA = StructType([
    StructField("rating", DoubleType()),
    StructField("title", StringType()),
    StructField("text", StringType()),
    StructField("asin", StringType()),
    StructField("parent_asin", StringType()),
    StructField("user_id", StringType()),
    StructField("timestamp", LongType()),
    StructField("helpful_vote", IntegerType()),
    StructField("verified_purchase", BooleanType()),
])

META_SCHEMA = StructType([
    StructField("parent_asin", StringType()),
    StructField("title", StringType()),
    StructField("main_category", StringType()),
    StructField("store", StringType()),
    StructField("average_rating", DoubleType()),
    StructField("rating_number", IntegerType()),
])


def run(cfg: dict) -> None:
    spark = get_spark("ingest", cfg)
    df = spark.read.schema(RAW_SCHEMA).json(data_path(cfg, cfg["paths"]["raw_reviews"]))
    frac = cfg["data"]["sample_fraction"]
    if frac < 1.0:
        df = df.sample(False, frac, cfg["project"]["seed"])
    df = (
        df.withColumn("review_ts", F.to_timestamp(F.from_unixtime((F.col("timestamp") / 1000).cast("long"))))
          .withColumn("year", F.year("review_ts"))
          .withColumn("month", F.month("review_ts"))
          .withColumn("review_id", F.sha2(F.concat_ws("|", "user_id", "parent_asin",
                                                      F.col("timestamp").cast("string")), 256))
          .drop("timestamp")
          .dropDuplicates(["review_id"])
    )
    df.write.mode("overwrite").partitionBy("year").parquet(data_path(cfg, "bronze", "reviews"))

    meta = spark.read.schema(META_SCHEMA).json(data_path(cfg, cfg["paths"]["raw_meta"]))
    meta.dropDuplicates(["parent_asin"]).write.mode("overwrite").parquet(data_path(cfg, "bronze", "meta"))
    spark.stop()


if __name__ == "__main__":
    run(load_config())
