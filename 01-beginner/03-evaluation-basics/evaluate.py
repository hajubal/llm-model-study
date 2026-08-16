#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from guardlab.eval import evaluate, slice_reports
from guardlab.io import read_jsonl, read_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--json-out")
    args = parser.parse_args()
    gold = read_jsonl(args.gold)
    pred = read_predictions(args.pred)
    report = evaluate(gold, pred)
    print(report.to_markdown())
    for field in ("language", "source"):
        print(f"\n### slice: {field}")
        for value, sliced in slice_reports(gold, pred, field).items():
            print(f"- {value}: n={sliced.n_samples}, macro_f1={sliced.macro_f1:.3f}, benign_fpr={sliced.benign_fpr:.3f}")
    if args.json_out:
        target = Path(args.json_out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

