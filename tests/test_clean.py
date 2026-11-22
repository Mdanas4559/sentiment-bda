from datetime import datetime

from pyspark.sql import functions as F
from pyspark.sql.types import (StructType, StructField, StringType, DoubleType, IntegerType,
                               BooleanType, TimestampType)

from src.clean import add_label, clean, normalize_text
from src.common.config import load_config

BRONZE = StructType([
    StructField("review_id", StringType()), StructField("parent_asin", StringType()),
    StructField("asin", StringType()), StructField("user_id", StringType()),
    StructField("rating", DoubleType()), StructField("title", StringType()), StructField("text", StringType()),
    StructField("helpful_vote", IntegerType()), StructField("verified_purchase", BooleanType()),
    StructField("review_ts", TimestampType()), StructField("year", IntegerType()), StructField("month", IntegerType()),
])
TS = datetime(2022, 1, 1)


def row(i, parent, rating, title, text):
    return (f"r{i}", parent, f"A{i}", f"u{i}", rating, title, text, 0, True, TS, 2022, 1)


def test_normalize_text_handles_html_urls_emoji_and_curly_apostrophes(spark):
    df = spark.createDataFrame([("Great <b>PHONE</b>!! See http://x.com \U0001F600 DON\u2019T stop",)], ["t"])
    assert df.select(normalize_text(F.col("t")).alias("o")).first()["o"] == "great phone !! see don't stop"


def test_binary_labels_drop_three_stars(spark):
    df = spark.createDataFrame([(5.0,), (4.0,), (3.0,), (2.0,), (1.0,)], ["rating"])
    got = {r["rating"]: r["label"] for r in add_label(df).collect()}
    assert got == {5.0: 1.0, 4.0: 1.0, 2.0: 0.0, 1.0: 0.0}


def test_clean_filters_and_dedupes(spark):
    cfg = load_config()
    rows = [
        row(1, "P1", 5.0, "Great", "Works great and charges fast"),
        row(2, "P1", 5.0, "Great", "Works great and charges fast"),          # duplicate text, same product -> dropped
        row(3, "P1", 1.0, "Bad", "Stopped working after a week"),
        row(4, "P1", 5.0, None, None),                                       # no text -> dropped
        row(5, "P1", 4.0, None, "ok fine"),                                  # fewer than min_words -> dropped
        row(6, "P1", 3.0, "Meh", "It is an average phone case"),             # 3 stars -> dropped
        row(7, "P1", 5.0, None, "\u8fd9\u4e2a\u624b\u673a\u5f88\u597d\u7528 \u8fd9\u4e2a\u624b\u673a\u5f88\u597d\u7528"),  # non-English -> dropped
    ]
    out = clean(spark.createDataFrame(rows, BRONZE), cfg)
    assert out.count() == 2
    assert {r["label"] for r in out.collect()} == {0.0, 1.0}
    assert out.filter(F.col("text_clean").isNull()).count() == 0
