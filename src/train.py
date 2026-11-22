import copy
import json
import time

import numpy as np
import pandas as pd
from pyspark import StorageLevel
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import (LogisticRegression, NaiveBayes, LinearSVC,
                                       RandomForestClassifier)
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array
from pyspark.ml.feature import CountVectorizerModel
from pyspark.sql import functions as F

from src.common.config import load_config, data_path, results_dir
from src.common.spark_session import get_spark
from src.features import build_feature_stages


def add_class_weights(df, enabled: bool = True):
    """weight = total / (n_classes * class_count): minority class gets a larger weight."""
    if not enabled:
        return df.withColumn("weight", F.lit(1.0))
    rows = df.groupBy("label").count().collect()
    total, k = sum(r["count"] for r in rows), len(rows)
    weights = {r["label"]: total / (k * r["count"]) for r in rows}
    mapping = F.create_map([F.lit(x) for kv in weights.items() for x in kv])
    return df.withColumn("weight", mapping[F.col("label")])


def get_linear_models(cfg: dict) -> dict:
    common = dict(featuresCol="features", labelCol="label", weightCol="weight")
    models = {
        "logreg": LogisticRegression(**common, maxIter=50, regParam=0.01),
        "naive_bayes": NaiveBayes(**common, modelType="multinomial", smoothing=1.0),
        "linear_svc": LinearSVC(**common, maxIter=50, regParam=0.01),
    }
    return {k: v for k, v in models.items() if k in cfg["train"]["models"]}


def evaluate(pred_df) -> dict:
    """Metrics from the confusion counts (small), plus ROC-AUC from rawPrediction."""
    cm = pred_df.groupBy("label", "prediction").count().toPandas()
    labels = sorted(set(cm["label"]) | set(cm["prediction"]))
    mat = (cm.pivot_table(index="label", columns="prediction", values="count", fill_value=0)
             .reindex(index=labels, columns=labels, fill_value=0))
    m = mat.values.astype(float)
    tp = np.diag(m)
    prec = tp / np.maximum(m.sum(axis=0), 1)
    rec = tp / np.maximum(m.sum(axis=1), 1)
    denom = prec + rec
    f1 = np.divide(2 * prec * rec, denom, out=np.zeros_like(denom), where=denom > 0)
    return {
        "accuracy": float(tp.sum() / m.sum()),
        "macro_precision": float(prec.mean()),
        "macro_recall": float(rec.mean()),
        "macro_f1": float(f1.mean()),
        "per_class": {str(l): {"precision": float(p), "recall": float(r), "f1": float(x)}
                      for l, p, r, x in zip(labels, prec, rec, f1)},
        "confusion_matrix": {"labels": [float(l) for l in labels], "matrix": m.tolist()},
        "roc_auc": BinaryClassificationEvaluator(metricName="areaUnderROC", rawPredictionCol="score").evaluate(
            pred_df.withColumn("score", vector_to_array("rawPrediction")[1] - vector_to_array("rawPrediction")[0])),
    }


def split(df, cfg: dict):
    t = cfg["train"]
    if t["split"] == "temporal":
        cut = t["temporal_cutoff_year"]
        return df.filter(F.col("year") <= cut), df.filter(F.col("year") > cut)
    return df.randomSplit([t["train_fraction"], 1 - t["train_fraction"]], seed=cfg["project"]["seed"])


def fit_random_forest(cfg: dict, train, test) -> dict:
    """Spark's tree learners hold one dense binned array per row (length = number of features), so a
    100k-feature, 1M-row RF cannot fit in memory. Train on a small subsample with a small vocabulary instead."""
    rf = cfg["train"]["rf"]
    n_train = train.count()
    sample = train.sample(False, min(1.0, rf["train_rows"] / n_train), cfg["project"]["seed"]) \
        if rf["train_rows"] < n_train else train
    small = copy.deepcopy(cfg)
    small["features"].update(vocab_size=rf["vocab_size"], use_bigrams=False)
    fm = Pipeline(stages=build_feature_stages(small)).fit(sample)
    s_f = fm.transform(sample).select("label", "weight", "features").persist(StorageLevel.MEMORY_AND_DISK)
    n_sample = s_f.count()
    est = RandomForestClassifier(featuresCol="features", labelCol="label", weightCol="weight",
                                 numTrees=rf["num_trees"], maxDepth=rf["max_depth"], seed=cfg["project"]["seed"])
    t0 = time.perf_counter()
    model = est.fit(s_f)
    fit_s = time.perf_counter() - t0
    metrics = evaluate(model.transform(fm.transform(test).select("label", "features")))
    metrics.update(fit_seconds=fit_s, trained_on_rows=n_sample, vocab_size=rf["vocab_size"])
    s_f.unpersist()
    return metrics


def run(cfg: dict) -> None:
    spark = get_spark("train", cfg)
    out_dir = results_dir(cfg) / "metrics" / cfg["train"]["experiment_name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    silver = spark.read.parquet(data_path(cfg, "silver", "reviews"))
    train, test = split(silver, cfg)
    train = add_class_weights(train, cfg["train"]["use_class_weights"])

    feat_model = Pipeline(stages=build_feature_stages(cfg)).fit(train)
    train_f = feat_model.transform(train).select("label", "weight", "features").persist(StorageLevel.MEMORY_AND_DISK)
    test_f = feat_model.transform(test).select("label", "features").persist(StorageLevel.MEMORY_AND_DISK)
    n_train, n_test = train_f.count(), test_f.count()
    label_counts = {str(r["label"]): r["count"] for r in train.groupBy("label").count().collect()}
    (out_dir / "dataset.json").write_text(json.dumps(
        {"train_rows": n_train, "test_rows": n_test, "train_label_counts": label_counts}, indent=2))

    results, fitted = {}, {}
    for name, est in get_linear_models(cfg).items():
        t0 = time.perf_counter()
        model = est.fit(train_f)
        fit_s = time.perf_counter() - t0
        metrics = evaluate(model.transform(test_f))
        metrics["fit_seconds"] = fit_s
        results[name], fitted[name] = metrics, model
        (out_dir / f"{name}.json").write_text(json.dumps(metrics, indent=2))

    if "random_forest" in cfg["train"]["models"]:
        results["random_forest"] = fit_random_forest(cfg, train, test)
        (out_dir / "random_forest.json").write_text(json.dumps(results["random_forest"], indent=2))

    (out_dir / "best_model.txt").write_text(max(results, key=lambda k: results[k]["macro_f1"]))

    if "logreg" in fitted:
        if cfg["train"]["save_scoring_model"]:
            PipelineModel(stages=feat_model.stages + [fitted["logreg"]]).write().overwrite().save(
                data_path(cfg, "models", "scoring_pipeline"))
        cv = next(s for s in feat_model.stages if isinstance(s, CountVectorizerModel))
        coef = fitted["logreg"].coefficients.toArray()
        order = np.argsort(coef)
        rows = ([(cv.vocabulary[i], float(coef[i]), "negative") for i in order[:50]] +
                [(cv.vocabulary[i], float(coef[i]), "positive") for i in order[::-1][:50]])
        pd.DataFrame(rows, columns=["term", "coefficient", "polarity"]).to_csv(out_dir / "top_features.csv", index=False)
    elif cfg["train"]["save_scoring_model"]:
        raise ValueError("train.models must include 'logreg' because it is the scoring model")

    train_f.unpersist()
    test_f.unpersist()
    spark.stop()


if __name__ == "__main__":
    run(load_config())
