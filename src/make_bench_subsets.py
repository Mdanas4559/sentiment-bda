"""Write fixed subsets of silver (10/25/50/100 %) so every benchmark run reads only its own data."""
from src.common.config import load_config, data_path
from src.common.spark_session import get_spark


def run(cfg: dict) -> None:
    spark = get_spark("bench_subsets", cfg)
    silver = spark.read.parquet(data_path(cfg, "silver", "reviews")).select("label", "text_clean")
    for frac in cfg["benchmark"]["fractions"]:
        pct = round(frac * 100)
        sub = silver if pct >= 100 else silver.sample(False, frac, cfg["project"]["seed"])
        sub.write.mode("overwrite").parquet(data_path(cfg, "bench", f"silver_{pct}"))
        print("written", pct)
    spark.stop()


if __name__ == "__main__":
    run(load_config())
