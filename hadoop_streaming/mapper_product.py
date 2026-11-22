#!/usr/bin/env python3
import sys

for line in sys.stdin:
    parts = line.rstrip("\n").split("\t")
    if len(parts) != 2:
        continue
    asin, rating = parts
    try:
        float(rating)
    except ValueError:
        continue
    print(f"{asin}\t{rating}")
