from src.common.config import load_config, data_path, results_dir
from src.common.spark_session import get_spark


def run(cfg: dict) -> None:
    spark = get_spark("analytics", cfg)
    spark.read.parquet(data_path(cfg, "gold", "predictions")).createOrReplaceTempView("predictions")
    spark.read.parquet(data_path(cfg, "gold", "aspect_clauses")).createOrReplaceTempView("aspect_clauses")
    spark.read.parquet(data_path(cfg, "bronze", "meta")).createOrReplaceTempView("meta")
    min_reviews = cfg["data"]["min_product_reviews"]
    min_mentions = cfg["aspects"]["min_mentions_per_product"]

    queries = {   # order matters: later queries use the views created from earlier ones
        "sentiment_by_month": """
            SELECT year, month, COUNT(*) AS n_reviews, AVG(prediction) AS pct_positive_pred,
                   AVG(rating) AS avg_rating, AVG(p_pos) AS avg_p_pos
            FROM predictions GROUP BY year, month ORDER BY year, month""",
        "product_summary": f"""
            SELECT p.parent_asin, m.title, m.store, COUNT(*) AS n_reviews, AVG(p.rating) AS avg_rating,
                   AVG(p.prediction) AS pct_positive_pred, AVG(p.p_pos) AS avg_p_pos
            FROM predictions p LEFT JOIN meta m ON p.parent_asin = m.parent_asin
            GROUP BY p.parent_asin, m.title, m.store
            HAVING COUNT(*) >= {min_reviews} ORDER BY n_reviews DESC LIMIT 2000""",
        "aspect_by_product": f"""
            SELECT c.parent_asin, c.aspect, COUNT(*) AS mentions, AVG(c.p_pos) AS avg_p_pos,
                   AVG(CASE WHEN c.polarity = 'positive' THEN 1.0 ELSE 0.0 END) AS pct_positive,
                   AVG(CASE WHEN c.polarity = 'negative' THEN 1.0 ELSE 0.0 END) AS pct_negative
            FROM aspect_clauses c
            WHERE c.parent_asin IN (SELECT parent_asin FROM product_summary)
            GROUP BY c.parent_asin, c.aspect HAVING COUNT(*) >= {min_mentions}""",
        "aspect_trend": """
            SELECT aspect, year, CEIL(month / 3) AS quarter, COUNT(*) AS mentions,
                   AVG(p_pos) AS avg_p_pos,
                   AVG(CASE WHEN polarity = 'negative' THEN 1.0 ELSE 0.0 END) AS pct_negative
            FROM aspect_clauses GROUP BY aspect, year, CEIL(month / 3) ORDER BY aspect, year, quarter""",
        "rating_vs_prediction": """
            SELECT rating, prediction, COUNT(*) AS n FROM predictions GROUP BY rating, prediction""",
    }
    out = results_dir(cfg) / "gold"
    out.mkdir(parents=True, exist_ok=True)
    for name, sql in queries.items():
        df = spark.sql(sql)
        df.createOrReplaceTempView(name)
        df.toPandas().to_csv(out / f"{name}.csv", index=False)
    spark.stop()


if __name__ == "__main__":
    run(load_config())
