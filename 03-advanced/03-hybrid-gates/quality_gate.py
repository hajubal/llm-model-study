#!/usr/bin/env python
from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("--min-attack-recall", type=float, default=0.90)
    parser.add_argument("--max-benign-fpr", type=float, default=0.05)
    args = parser.parse_args()
    report = json.load(open(args.report, encoding="utf-8"))
    failures = []
    if report["attack_recall"] < args.min_attack_recall:
        failures.append(f"attack_recall {report['attack_recall']:.3f} < {args.min_attack_recall:.3f}")
    if report["benign_fpr"] > args.max_benign_fpr:
        failures.append(f"benign_fpr {report['benign_fpr']:.3f} > {args.max_benign_fpr:.3f}")
    if failures:
        raise SystemExit("QUALITY GATE FAILED: " + "; ".join(failures))
    print("QUALITY GATE PASSED")


if __name__ == "__main__":
    main()

