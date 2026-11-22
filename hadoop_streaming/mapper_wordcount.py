#!/usr/bin/env python3
import sys

# small stopword list; negations are intentionally NOT included
STOP = {"the", "a", "an", "and", "or", "of", "to", "in", "is", "it", "this", "that", "for",
        "with", "on", "as", "was", "are", "be", "i", "my", "me", "you", "they", "have",
        "has", "had", "at", "by", "from", "so", "if", "its", "am", "were", "been", "do"}


def main():
    for line in sys.stdin:
        parts = line.rstrip("\n").split("\t", 1)
        if len(parts) != 2:
            continue
        label, text = parts
        for w in text.split():
            w = w.strip(".,!?;-'")
            if len(w) >= 3 and w not in STOP:
                print(f"{label}|{w}\t1")


if __name__ == "__main__":
    main()
