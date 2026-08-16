#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from guardlab.eval import error_summary, evaluate, slice_reports
from guardlab.io import read_jsonl, read_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--out", required=True, help="report를 저장할 디렉터리")
    args = parser.parse_args()
    gold, pred = read_jsonl(args.gold), read_predictions(args.pred)
    report = evaluate(gold, pred)
    slices = {
        field: {value: sliced.to_dict() for value, sliced in slice_reports(gold, pred, field).items()}
        for field in ("source", "language")
    }
    payload = report.to_dict() | {"slices": slices, "error_summary": error_summary(report)}
    target = Path(args.out)
    target.mkdir(parents=True, exist_ok=True)
    (target / "report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (target / "report.md").write_text(report.to_markdown(), encoding="utf-8")
    (target / "errors.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in report.errors), encoding="utf-8"
    )
    print(report.to_markdown())
    print(f"\nreport -> {target}")


if __name__ == "__main__":
    main()

