#!/usr/bin/env python3
import sys

cur, n, s = None, 0, 0.0
for line in sys.stdin:
    key, _, val = line.rstrip("\n").partition("\t")
    if key != cur:
        if cur is not None:
            print(f"{cur}\t{n}\t{s / n:.4f}")
        cur, n, s = key, 0, 0.0
    n += 1
    s += float(val)
if cur is not None:
    print(f"{cur}\t{n}\t{s / n:.4f}")
