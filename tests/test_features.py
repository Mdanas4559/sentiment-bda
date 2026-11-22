from pyspark.ml import Pipeline
from pyspark.ml.feature import CountVectorizerModel

from src.common.config import load_config
from src.features import build_feature_stages, custom_stopwords


def test_negations_survive_stopword_removal(spark):
    cfg = load_config()
    stop = set(custom_stopwords(cfg))
    assert not {"not", "no", "never", "don't", "isn't", "cannot"} & stop
    assert "the" in stop
    cfg["features"]["keep_negations"] = False        # experiment E3 behaviour
    assert "not" in set(custom_stopwords(cfg))


def test_bigram_not_good_is_in_the_vocabulary(spark):
    cfg = load_config()
    cfg["features"].update(min_df=1, vocab_size=1000)
    df = spark.createDataFrame([("this is not good at all",), ("this is very good",)], ["text_clean"])
    model = Pipeline(stages=build_feature_stages(cfg)).fit(df)
    vocab = set(next(s for s in model.stages if isinstance(s, CountVectorizerModel)).vocabulary)
    assert {"not", "good", "not good"} <= vocab
    assert "features" in model.transform(df).columns
