import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from download_data import truncate_lines      # noqa: E402
from make_sample import make_sample           # noqa: E402


def test_truncate_lines(tmp_path):
    f = tmp_path / "x.jsonl"
    f.write_text("".join(f"{i}\n" for i in range(10)))
    assert truncate_lines(f, 4) == 4
    assert f.read_text().splitlines() == ["0", "1", "2", "3"]


def test_make_sample_is_random_spread_and_filters_meta(tmp_path):
    reviews = tmp_path / "r.jsonl"
    # product ids repeat every 20 lines: evenly spaced sampling would see only a few of them
    reviews.write_text("".join(json.dumps({"parent_asin": f"P{i % 20}", "rating": 5.0, "text": f"t{i}"}) + "\n"
                               for i in range(2000)))
    meta = tmp_path / "m.jsonl"
    meta.write_text("".join(json.dumps({"parent_asin": f"P{i}", "title": "x"}) + "\n" for i in range(200)))
    info = make_sample(reviews, meta, tmp_path / "out", n=200)
    assert 150 <= info["reviews"] <= 250                      # about n
    assert info["products"] >= 18                             # random sampling sees (almost) every product
    lines = (tmp_path / "out" / "reviews_sample.jsonl").read_text().splitlines()
    ids = [int(json.loads(l)["text"][1:]) for l in lines]
    assert max(ids) > 1800 and min(ids) < 200                 # spans the whole file, not just the beginning
    sample_products = {json.loads(l)["parent_asin"] for l in lines}
    kept_meta = {json.loads(l)["parent_asin"] for l in (tmp_path / "out" / "meta_sample.jsonl").read_text().splitlines()}
    assert kept_meta == sample_products and len(kept_meta) == info["meta_rows"] > 0
