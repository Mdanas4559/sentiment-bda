"""Verify the MapReduce outputs against Spark on the same inputs, and compare runtimes.
Run after hadoop_streaming/run_mapreduce.sh (which writes results/benchmarks/mr_runtime.csv)."""
import time

import pandas as pd
from pyspark.sql import functions as F

from hadoop_streaming.mapper_wordcount import STOP      # same stopword set as the MapReduce mapper
from src.common.config import load_config, data_path, results_dir
from src.common.spark_session import get_spark


def run(cfg: dict) -> None:
    spark = get_spark("mr_compare", cfg)
    mr_times = pd.read_csv(results_dir(cfg) / "benchmarks" / "mr_runtime.csv").set_index("job")["seconds"]

    # ---- Job A: word frequency per sentiment class -------------------------------------------------
    text = spark.read.option("sep", "\t").schema("label string, text string").csv(data_path(cfg, "mr", "in_text"))
    rows = text.count()
    words = (text.select("label", F.explode(F.split("text", " ")).alias("w"))
                 .withColumn("w", F.regexp_replace("w", r"^[.,!?;'\-]+|[.,!?;'\-]+$", ""))
                 .filter((F.length("w") >= 3) & (~F.col("w").isin(list(STOP))))
                 .groupBy(F.concat_ws("|", "label", "w").alias("key")).count())
    t0 = time.perf_counter()
    words.write.mode("overwrite").parquet(data_path(cfg, "mr", "spark_out_wordcount"))
    spark_wc = time.perf_counter() - t0
    mr_wc = spark.read.option("sep", "\t").schema("key string, cnt long").csv(data_path(cfg, "mr", "out_wordcount"))
    sp_wc = spark.read.parquet(data_path(cfg, "mr", "spark_out_wordcount"))
    wc_bad = (mr_wc.alias("m").join(sp_wc.alias("s"), F.col("m.key") == F.col("s.key"), "full_outer")
                   .filter(F.col("m.cnt").isNull() | F.col("s.count").isNull() | (F.col("m.cnt") != F.col("s.count")))
                   .count())

    # ---- Job B: per-product review count and average rating ----------------------------------------
    rt = spark.read.option("sep", "\t").schema("asin string, rating double").csv(data_path(cfg, "mr", "in_rating"))
    prod = rt.groupBy("asin").agg(F.count("*").alias("n"), F.avg("rating").alias("avg"))
    t0 = time.perf_counter()
    prod.write.mode("overwrite").parquet(data_path(cfg, "mr", "spark_out_product"))
    spark_pr = time.perf_counter() - t0
    mr_pr = spark.read.option("sep", "\t").schema("asin string, n long, avg double").csv(data_path(cfg, "mr", "out_product"))
    sp_pr = spark.read.parquet(data_path(cfg, "mr", "spark_out_product"))
    pr_bad = (mr_pr.alias("m").join(sp_pr.alias("s"), F.col("m.asin") == F.col("s.asin"), "full_outer")
                   .filter(F.col("m.n").isNull() | F.col("s.n").isNull() | (F.col("m.n") != F.col("s.n"))
                           | (F.abs(F.col("m.avg") - F.col("s.avg")) > 1e-3))
                   .count())

    out = pd.DataFrame([
        {"job": "wordcount", "rows": rows, "mapreduce_seconds": float(mr_times["wordcount"]),
         "spark_seconds": round(spark_wc, 2), "identical": wc_bad == 0, "mismatching_keys": wc_bad},
        {"job": "product_stats", "rows": rows, "mapreduce_seconds": float(mr_times["product_stats"]),
         "spark_seconds": round(spark_pr, 2), "identical": pr_bad == 0, "mismatching_keys": pr_bad},
    ])
    out.to_csv(results_dir(cfg) / "benchmarks" / "mr_vs_spark.csv", index=False)
    print(out)
    spark.stop()
    if not out["identical"].all():
        raise SystemExit("MapReduce and Spark results differ: investigate before using these numbers")


if __name__ == "__main__":
    run(load_config())
