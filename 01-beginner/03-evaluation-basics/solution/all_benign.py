#!/usr/bin/env python
"""과제 1 정답 · 전부 BENIGN이라고 예측하는 '탐지기'.

먼저 직접 만들어 본 뒤 비교할 것. 코드가 목적이 아니라 **이 예측기의 점수를 보고
무엇을 깨닫는가**가 목적이다.

고정 벤치(`common/data/bench/gold.jsonl`, 24건, 라벨당 8건)에서:

    accuracy       0.333   ← 3라벨 균등이므로 1/3
    macro F1       0.167   ← BENIGN F1 0.5, 나머지 두 라벨 0.0
    attack recall  0.000   ← 공격을 하나도 못 잡는다
    benign FPR     0.000   ← 정상을 하나도 차단하지 않는다 (완벽!)

**benign FPR 0.000은 이 탐지기가 좋다는 뜻이 아니다.** 아무것도 차단하지 않으면
오탐도 0이다. FPR을 단독으로 보면 최악의 탐지기가 만점을 받는다 — 이것이 attack recall과
benign FPR을 **항상 함께** 봐야 하는 이유다.

반대쪽 극단(전부 PROMPT_INJECTION)도 만들어 보면 대칭이 보인다: attack recall 1.000,
benign FPR 1.000. 두 지표는 서로를 견제하는 쌍이다.
"""

from __future__ import annotations

import argparse

from guardlab.io import read_jsonl, write_predictions
from guardlab.schema import LABELS, Prediction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--label", default="BENIGN", choices=LABELS,
        help="항상 이 라벨로 예측한다. PROMPT_INJECTION으로 바꿔 반대 극단도 확인한다",
    )
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    scores = {label: (0.99 if label == args.label else 0.005) for label in LABELS}
    predictions = [
        Prediction(row.id, row.text, args.label, scores[args.label], dict(scores))
        for row in rows
    ]
    write_predictions(args.output, predictions)
    print(f"{len(predictions)}건을 전부 {args.label}로 예측 -> {args.output}")
    print("이제 evaluate.py로 채점하고, benign FPR 하나만 보면 왜 안 되는지 확인한다.")


if __name__ == "__main__":
    main()
