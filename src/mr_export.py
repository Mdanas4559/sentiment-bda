"""Export a deterministic sample of silver to tab-separated text in HDFS (the MapReduce inputs)."""
from pyspark import StorageLevel

from src.common.config import load_config, data_path
from src.common.spark_session import get_spark


def run(cfg: dict) -> None:
    spark = get_spark("mr_export", cfg)
    silver = spark.read.parquet(data_path(cfg, "silver", "reviews")).select("parent_asin", "rating", "label", "text_clean")
    n = silver.count()
    frac = min(1.0, cfg["mapreduce"]["n_rows"] / n)
    df = silver if frac >= 1.0 else silver.sample(False, frac, cfg["project"]["seed"])
    df = df.persist(StorageLevel.MEMORY_AND_DISK)          # both outputs must come from exactly the same rows
    print("rows exported:", df.count())
    (df.select("label", "text_clean").coalesce(4).write.mode("overwrite")
       .option("sep", "\t").csv(data_path(cfg, "mr", "in_text")))
    (df.select("parent_asin", "rating").coalesce(4).write.mode("overwrite")
       .option("sep", "\t").csv(data_path(cfg, "mr", "in_rating")))
    df.unpersist()
    spark.stop()


if __name__ == "__main__":
    run(load_config())
