from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from src.common.config import load_config, data_path
from src.common.spark_session import get_spark


def run(cfg: dict) -> None:
    spark = get_spark("predict", cfg)
    model = PipelineModel.load(data_path(cfg, "models", "scoring_pipeline"))
    silver = spark.read.parquet(data_path(cfg, "silver", "reviews"))
    scored = model.transform(silver)
    out = scored.select(
        "review_id", "parent_asin", "rating", "label", "prediction",
        vector_to_array("probability")[1].alias("p_pos"),
        "helpful_vote", "verified_purchase", "review_ts", "year", "month", "word_count",
    )
    out.write.mode("overwrite").parquet(data_path(cfg, "gold", "predictions"))
    spark.stop()


if __name__ == "__main__":
    run(load_config())
