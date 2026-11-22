from src.aspects import load_lexicon, tag_aspects, to_clauses
from src.common.config import load_config


def test_clauses_split_on_contrast_words_and_get_tagged(spark):
    cfg = load_config()
    silver = spark.createDataFrame(
        [("r1", "P1", 2022, 1, "great phone but terrible battery life. the cable works fine; however the box was damaged")],
        ["review_id", "parent_asin", "year", "month", "text_clean"])
    clauses = [r["clause"] for r in to_clauses(silver, cfg).collect()]
    assert any(c.startswith("terrible battery life") for c in clauses)
    assert all(not c.startswith(("but ", "however ")) for c in clauses)            # leading connectors stripped
    tagged = {(r["clause"], r["aspect"]) for r in tag_aspects(to_clauses(silver, cfg), load_lexicon(cfg)).collect()}
    assert any(a == "battery_charging" and "battery" in c for c, a in tagged)
    assert any(a == "cable_connectivity" and "cable" in c for c, a in tagged)
    assert any(a == "delivery_packaging" and "box" in c for c, a in tagged)
    assert not any(c.startswith("great phone") for c, a in tagged)                  # no aspect word -> dropped
