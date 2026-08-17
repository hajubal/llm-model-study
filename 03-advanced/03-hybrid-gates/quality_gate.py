#!/usr/bin/env python
"""report.json이 기준을 넘는지 검사하고, 못 넘으면 0이 아닌 코드로 종료한다.

게이트의 핵심은 사람이 잊어도 자동으로 막히는 것이다. CI에서 이 스크립트를 돌려
기준선 회귀를 배포 전에 잡는다(.github/workflows/ci.yml 참고).

기준값에는 두 종류가 있다. 헷갈리면 게이트가 무의미해진다.
- 회귀 방지선: "지금보다 나빠지면 실패". 현재 실측보다 살짝 낮게 잡는다. CI용.
- 운영 목표치: "이 값을 넘어야 배포". 아직 도달하지 못한 값이다. 릴리스 심사용.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# (인자 이름, report.json 키, 비교 방향) — 방향은 "min"이면 이상, "max"면 이하여야 통과다.
CHECKS = (
    ("min_macro_f1", "macro_f1", "min"),
    ("min_attack_recall", "attack_recall", "min"),
    ("max_benign_fpr", "benign_fpr", "max"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    # 기본값을 두지 않는다. 넘긴 기준만 검사한다.
    # 공격 샘플이 없는 하드 네거티브 파일에 attack recall 기준이 자동으로 걸리면
    # "미탐"이 아니라 "해당 없음"인 0.000 때문에 게이트가 늘 실패한다.
    parser.add_argument("--min-macro-f1", type=float, help="macro F1 하한")
    parser.add_argument("--min-attack-recall", type=float, help="attack recall 하한")
    parser.add_argument("--max-benign-fpr", type=float, help="benign FPR 상한")
    parser.add_argument(
        "--label", default="quality gate", help="여러 게이트를 돌릴 때 로그에서 구분할 이름",
    )
    args = parser.parse_args()
    if all(getattr(args, name) is None for name, _, _ in CHECKS):
        raise SystemExit("검사할 기준을 하나 이상 지정하세요 (--min-macro-f1 / --min-attack-recall / --max-benign-fpr)")
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))

    failures, passes = [], []
    for arg_name, key, direction in CHECKS:
        threshold = getattr(args, arg_name)
        if threshold is None:
            continue
        actual = report[key]
        ok = actual >= threshold if direction == "min" else actual <= threshold
        symbol = ">=" if direction == "min" else "<="
        line = f"{key} {actual:.3f} {symbol} {threshold:.3f}"
        (passes if ok else failures).append(line if ok else line.replace(symbol, "<" if direction == "min" else ">"))

    for line in passes:
        print(f"  ok   {line}")
    for line in failures:
        print(f"  FAIL {line}")

    # 신뢰구간이 있으면 함께 보여준다. 표본이 작으면 게이트 통과/실패 자체가 흔들린다.
    ci = report.get("ci") or {}
    if ci:
        bounds = ci.get("macro_f1")
        if bounds:
            print(
                f"  참고 n={report['n_samples']} · macro F1 95% CI "
                f"[{bounds[0]:.3f}–{bounds[1]:.3f}] — 구간이 넓으면 이 판정도 표본 운에 좌우된다"
            )
    else:
        print("  참고: 신뢰구간이 없다. evaluate.py를 --bootstrap과 함께 돌리면 판정의 불확실성을 볼 수 있다")

    if failures:
        raise SystemExit(f"{args.label} FAILED: " + "; ".join(failures))
    print(f"{args.label} PASSED")


if __name__ == "__main__":
    main()
