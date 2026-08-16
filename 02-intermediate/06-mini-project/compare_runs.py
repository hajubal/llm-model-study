#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", help="report.json 경로들")
    args = parser.parse_args()
    print("| run | macro F1 | attack recall | benign FPR | n |")
    print("|---|---:|---:|---:|---:|")
    for path_string in args.reports:
        path = Path(path_string)
        report = json.loads(path.read_text(encoding="utf-8"))
        print(
            f"| {path.parent} | {report['macro_f1']:.3f} | {report['attack_recall']:.3f} | "
            f"{report['benign_fpr']:.3f} | {report['n_samples']} |"
        )


if __name__ == "__main__":
    main()

