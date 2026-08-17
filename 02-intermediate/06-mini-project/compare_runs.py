#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def cell(report: dict, metric: str) -> str:
    """지표 하나를 신뢰구간과 함께 표시한다.

    구간이 없으면 점 추정만 보여준다. 구간이 없는 숫자를 다른 run과 비교하는 것은
    표본 변동인지 개선인지 구분하지 않겠다는 뜻이므로, 비교표에서는 늘 붙인다.
    """
    value = f"{report[metric]:.3f}"
    bounds = (report.get("ci") or {}).get(metric)
    if not bounds:
        return value
    return f"{value} [{bounds[0]:.3f}–{bounds[1]:.3f}]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", help="report.json 경로들")
    args = parser.parse_args()
    print("| run | macro F1 | attack recall | benign FPR | n |")
    print("|---|---:|---:|---:|---:|")
    without_ci = []
    for path_string in args.reports:
        path = Path(path_string)
        report = json.loads(path.read_text(encoding="utf-8"))
        if not report.get("ci"):
            without_ci.append(str(path.parent))
        print(
            f"| {path.parent} | {cell(report, 'macro_f1')} | "
            f"{cell(report, 'attack_recall')} | {cell(report, 'benign_fpr')} | "
            f"{report['n_samples']} |"
        )
    if without_ci:
        print(
            f"\n주의: 신뢰구간이 없는 run({', '.join(without_ci)})은 표본 변동을 알 수 없다. "
            "evaluate.py를 --bootstrap 없이 돌렸다면 다시 돌린다."
        )
    print(
        "\n구간이 크게 겹치는 두 run의 차이는 개선이라고 부를 수 없다. "
        "seed를 바꿔 다시 학습해도 그만큼 움직인다면 더더욱 그렇다."
    )


if __name__ == "__main__":
    main()
