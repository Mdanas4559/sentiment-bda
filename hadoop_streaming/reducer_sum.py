#!/usr/bin/env python3
import sys

cur, total = None, 0
for line in sys.stdin:
    key, _, val = line.rstrip("\n").partition("\t")
    if key == cur:
        total += int(val)
    else:
        if cur is not None:
            print(f"{cur}\t{total}")
        cur, total = key, int(val)
if cur is not None:
    print(f"{cur}\t{total}")
