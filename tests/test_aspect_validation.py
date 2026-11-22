import pandas as pd
import pytest

from src.aspect_validation import validate


def write(tmp_path, polarity_for_wrong=None):
    annotated = pd.DataFrame({
        "row_id": [1, 2, 3, 4],
        "aspect": ["battery_charging", "battery_charging", "price_value", "price_value"],
        "clause": ["a", "b", "c", "d"],
        "aspect_correct": [1, 1, 1, 0],
        "human_polarity": ["negative", "positive", "neutral", polarity_for_wrong],
    })
    key = pd.DataFrame({
        "row_id": [1, 2, 3, 4], "review_id": list("wxyz"),
        "aspect": annotated["aspect"], "p_pos": [0.1, 0.9, 0.5, 0.7],
        "polarity": ["negative", "negative", "neutral", "positive"],
    })
    annotated.to_csv(tmp_path / "a.csv", index=False)
    key.to_csv(tmp_path / "k.csv", index=False)
    return tmp_path / "a.csv", tmp_path / "k.csv"


def test_metrics(tmp_path):
    out = validate(*write(tmp_path))
    assert out["n_rows"] == 4
    assert out["tagging_precision"] == pytest.approx(0.75)
    assert out["n_correctly_tagged"] == 3
    assert out["polarity_agreement_all"] == pytest.approx(2 / 3)          # rows 1 and 3 agree, row 2 does not
    assert out["polarity_agreement_non_neutral"] == pytest.approx(0.5)    # rows 1 and 2 only
    assert out["tagging_precision_by_aspect"]["price_value"]["tagging_precision"] == pytest.approx(0.5)


def test_rejects_unlabeled_rows(tmp_path):
    a, k = write(tmp_path)
    df = pd.read_csv(a)
    df.loc[0, "aspect_correct"] = None
    df.to_csv(a, index=False)
    with pytest.raises(ValueError):
        validate(a, k)


def test_rejects_bad_polarity(tmp_path):
    a, k = write(tmp_path)
    df = pd.read_csv(a)
    df.loc[0, "human_polarity"] = "good"
    df.to_csv(a, index=False)
    with pytest.raises(ValueError):
        validate(a, k)
