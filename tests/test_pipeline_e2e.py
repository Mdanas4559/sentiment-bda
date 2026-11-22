"""Runs clean -> features -> Logistic Regression -> save/load -> clause scoring -> evaluate on 60 synthetic reviews.
This is the main safety net for the Spark code: if it passes, the riskiest API calls work on your install."""
from datetime import datetime

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import functions as F

from src.aspects import load_lexicon, score, tag_aspects, to_clauses
from src.clean import clean
from src.common.config import load_config
from src.features import build_feature_stages
from src.train import add_class_weights, evaluate
from tests.test_clean import BRONZE

POS = ["great battery life and fast charging, i love it", "excellent quality and it fits my phone perfectly",
       "the cable is sturdy and the price is a bargain", "fast delivery and nice packaging, works great",
       "sound is crisp and the screen is bright"]
NEG = ["terrible battery, it died in a week, not good", "cheap plastic, broke quickly, do not buy",
       "the cable stopped working and support was useless", "arrived late and the box was damaged, very bad",
       "screen cracked and the sound is awful"]


def make_bronze(spark):
    ts = datetime(2022, 1, 1)
    rows = []
    for i in range(60):
        pos = i % 2 == 0
        text = f"{(POS if pos else NEG)[(i // 2) % 5]} number {i}"
        rows.append((f"r{i}", f"P{i % 3}", f"A{i}", f"u{i}", 5.0 if pos else 1.0, None, text, 0, True, ts, 2022, (i % 12) + 1))
    return spark.createDataFrame(rows, BRONZE)


def test_end_to_end(spark, tmp_path):
    cfg = load_config()
    cfg["features"].update(min_df=1, vocab_size=500)
    silver = clean(make_bronze(spark), cfg).cache()
    assert silver.count() == 60

    weighted = add_class_weights(silver)
    assert weighted.select("weight").distinct().count() == 1                       # balanced classes -> equal weights
    skewed = add_class_weights(silver.filter((F.col("label") == 1.0) | (F.col("review_id") < "r2")))
    w = {r["label"]: r["weight"] for r in skewed.select("label", "weight").distinct().collect()}
    assert w[0.0] > w[1.0]                                                         # minority class weighs more

    pipe = Pipeline(stages=[*build_feature_stages(cfg),
                            LogisticRegression(featuresCol="features", labelCol="label", maxIter=30)])
    model = pipe.fit(silver)
    path = str(tmp_path / "scoring_pipeline")
    model.write().overwrite().save(path)
    loaded = PipelineModel.load(path)

    metrics = evaluate(loaded.transform(silver))
    assert metrics["macro_f1"] > 0.9 and 0.0 <= metrics["roc_auc"] <= 1.0

    scored = score(tag_aspects(to_clauses(silver, cfg), load_lexicon(cfg)), loaded, cfg)
    assert {"aspect", "clause", "p_pos", "polarity"} <= set(scored.columns)
    assert scored.count() > 20
    neg_mean = scored.filter(F.col("clause").rlike("terrible|cracked|broke|awful")).agg(F.avg("p_pos")).first()[0]
    pos_mean = scored.filter(F.col("clause").rlike("great|excellent|sturdy|crisp")).agg(F.avg("p_pos")).first()[0]
    assert neg_mean < 0.5 < pos_mean
