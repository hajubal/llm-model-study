#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter

from guardlab.eval import evaluate
from guardlab.io import read_jsonl, read_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()
    report = evaluate(read_jsonl(args.gold), read_predictions(args.pred))
    counts = Counter((item["gold"], item["pred"], item["source"], item["language"]) for item in report.errors)
    print("gold -> pred / source / language")
    for key, count in counts.most_common():
        print(f"{count:3}  {key}")
    print("\nexamples")
    for item in sorted(report.errors, key=lambda row: -row["score"])[:args.limit]:
        print(f"[{item['id']}] {item['gold']} -> {item['pred']} score={item['score']:.3f} | {item['text']}")


if __name__ == "__main__":
    main()

