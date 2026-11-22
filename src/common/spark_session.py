from pyspark.sql import SparkSession


def get_spark(app_name: str, cfg: dict, cores: int | None = None) -> SparkSession:
    master = f"local[{cores}]" if cores else cfg["spark"]["master"]
    spark = (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.driver.memory", cfg["spark"]["driver_memory"])
        .config("spark.sql.shuffle.partitions", str(cfg["spark"]["shuffle_partitions"]))
        .config("spark.sql.adaptive.enabled", str(cfg["spark"]["adaptive"]).lower())
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark
