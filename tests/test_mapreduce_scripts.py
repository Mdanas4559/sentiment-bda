"""Simulates Hadoop Streaming locally: map -> sort by key -> (combine) -> reduce."""
import subprocess
import sys
from collections import Counter
from pathlib import Path

HS = Path(__file__).resolve().parents[1] / "hadoop_streaming"


def run_script(name: str, lines: list[str]) -> list[str]:
    p = subprocess.run([sys.executable, str(HS / name)], input="".join(l + "\n" for l in lines),
                       capture_output=True, text=True, check=True)
    return p.stdout.splitlines()


def shuffle(lines: list[str]) -> list[str]:
    return sorted(lines, key=lambda l: l.split("\t", 1)[0])      # Hadoop sorts by key between map and reduce


def test_wordcount_matches_counter_and_keeps_negations():
    rows = ["1.0\tgreat battery, not good.", "0.0\tthe battery died; don't buy", "1.0\tgreat great phone"]
    mapped = run_script("mapper_wordcount.py", rows)
    reduced = run_script("reducer_sum.py", shuffle(mapped))
    got = {k: int(v) for k, v in (l.split("\t") for l in reduced)}
    assert got["1.0|great"] == 3
    assert got["1.0|not"] == 1 and got["0.0|don't"] == 1          # negations survive
    assert "1.0|the" not in got and "0.0|the" not in got           # stopword removed
    assert Counter(l.split("\t")[0] for l in mapped) == Counter({k: v for k, v in got.items()})


def test_combiner_does_not_change_result():
    rows = ["1.0\tgreat phone great price", "0.0\tbad phone bad cable"] * 20
    mapped = shuffle(run_script("mapper_wordcount.py", rows))
    direct = run_script("reducer_sum.py", mapped)
    combined = run_script("reducer_sum.py", shuffle(run_script("reducer_sum.py", mapped)))
    assert direct == combined


def test_product_stats():
    rows = ["A1\t5.0", "A1\t3.0", "B2\t1.0", "bad line", "A1\tnot-a-number"]
    reduced = run_script("reducer_product.py", shuffle(run_script("mapper_product.py", rows)))
    assert reduced == ["A1\t2\t4.0000", "B2\t1\t1.0000"]


def test_two_map_tasks_each_with_a_combiner():
    """Several partial counts for the same key reach the reducer, as on a real cluster with >1 map task."""
    chunk1 = ["1.0\tgreat phone great price"] * 7
    chunk2 = ["1.0\tgreat cable", "0.0\tbad phone"] * 5
    partial1 = run_script("reducer_sum.py", shuffle(run_script("mapper_wordcount.py", chunk1)))     # combiner of map task 1
    partial2 = run_script("reducer_sum.py", shuffle(run_script("mapper_wordcount.py", chunk2)))     # combiner of map task 2
    final = {k: int(v) for k, v in (l.split("\t") for l in run_script("reducer_sum.py", shuffle(partial1 + partial2)))}
    assert final["1.0|great"] == 7 * 2 + 5 and final["1.0|phone"] == 7 and final["0.0|bad"] == 5
