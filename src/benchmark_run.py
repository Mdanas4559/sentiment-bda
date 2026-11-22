"""One benchmark measurement in a fresh Spark session. Called by src/benchmark.py, not by hand."""
import argparse
import time

from pyspark import StorageLevel
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression

from src.benchmark import append_row
from src.common.config import load_config, data_path, results_dir
from src.common.spark_session import get_spark
from src.features import build_feature_stages


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cores", type=int, required=True)
    ap.add_argument("--pct", type=int, required=True)
    ap.add_argument("--rep", type=int, required=True)      # -1 = warm-up, not recorded
    a = ap.parse_args()
    cfg = load_config()
    spark = get_spark(f"bench-{a.cores}c-{a.pct}p", cfg, cores=a.cores)

    t0 = time.perf_counter()
    df = spark.read.parquet(data_path(cfg, "bench", f"silver_{a.pct}"))
    rows = df.count()
    read_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    fm = Pipeline(stages=build_feature_stages(cfg)).fit(df)
    feats = fm.transform(df).select("label", "features").persist(StorageLevel.MEMORY_AND_DISK)
    feats.count()
    features_s = time.perf_counter() - t1

    t2 = time.perf_counter()
    LogisticRegression(featuresCol="features", labelCol="label", maxIter=20, regParam=0.01).fit(feats)
    train_s = time.perf_counter() - t2

    if a.rep >= 0:
        append_row(results_dir(cfg) / "benchmarks" / "runtime.csv", dict(
            cores=a.cores, pct=a.pct, rep=a.rep, rows=rows, read_s=round(read_s, 2),
            features_s=round(features_s, 2), train_s=round(train_s, 2),
            total_s=round(read_s + features_s + train_s, 2)))
    spark.stop()


if __name__ == "__main__":
    main()
