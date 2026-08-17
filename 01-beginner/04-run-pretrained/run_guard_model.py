#!/usr/bin/env python
"""공개된 '전용' 가드 모델을 우리 벤치에 돌려 기준선으로 삼는다.

`run_zero_shot.py`는 범용 NLI 모델을 학습 없이 쓴다. 이 스크립트는 프롬프트 인젝션
탐지를 위해 **학습된 전용 모델**을 쓴다. 우리가 만들 모델이 넘어야 할 진짜 기준선은
"전부 BENIGN"이나 제로샷이 아니라 이쪽이다.

## 핵심 함정: 라벨 체계가 우리와 다르다

대부분의 공개 가드 모델은 2-class(안전/공격)다. 우리는 3-class(BENIGN /
PROMPT_INJECTION / JAILBREAK)다. 2-class 모델은 JAILBREAK를 **구조적으로** 절대
맞힐 수 없으므로 macro F1이 낮게 나온다. 이것은 모델이 나빠서가 아니라 **비교할 수
없는 지표를 비교했기 때문**이다.

    비교 가능:   attack recall, benign FPR   (둘 다 공격/정상 2분류 지표)
    비교 불가:   macro F1, 라벨별 F1          (라벨 개수가 다르다)

이 스크립트는 3-class를 못 내는 매핑을 감지하면 경고를 출력한다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from guardlab.io import read_jsonl, write_predictions
from guardlab.schema import LABELS, Prediction

# 후보 모델. 라이선스와 접근 조건이 서로 다르므로 쓰기 전에 직접 확인한다.
#
#   protectai/deberta-v3-base-prompt-injection-v2  Apache-2.0, 게이트 없음, 2-class
#   meta-llama/Prompt-Guard-86M                    Llama 라이선스, 접근 승인 필요, 다중 클래스
#   meta-llama/Llama-Guard-3-8B                    Llama 라이선스, 승인 필요, 생성형(8B, 무겁다)
#
# 기본값은 승인 절차 없이 바로 받을 수 있는 것으로 둔다. 교육용 저장소에서 게이트가
# 걸린 모델을 기본값으로 두면 대부분의 학습자가 첫 줄에서 막힌다.
DEFAULT_MODEL = "protectai/deberta-v3-base-prompt-injection-v2"

# 모델이 내는 라벨 -> 우리 라벨. 모델마다 다르므로 --map으로 덮어쓸 수 있다.
DEFAULT_MAP = {
    "SAFE": "BENIGN",
    "INJECTION": "PROMPT_INJECTION",
    "LABEL_0": "BENIGN",
    "LABEL_1": "PROMPT_INJECTION",
    "BENIGN": "BENIGN",
    "JAILBREAK": "JAILBREAK",
    "INDIRECT_INJECTION": "PROMPT_INJECTION",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument(
        "--map", help='라벨 매핑 JSON. 예: \'{"INJECTION":"PROMPT_INJECTION","SAFE":"BENIGN"}\'',
    )
    args = parser.parse_args()

    mapping = dict(DEFAULT_MAP)
    if args.map:
        # 파일 경로로도, 인라인 JSON으로도 받는다.
        raw = Path(args.map).read_text(encoding="utf-8") if Path(args.map).exists() else args.map
        mapping.update(json.loads(raw))
    unknown = {value for value in mapping.values() if value not in LABELS}
    if unknown:
        raise SystemExit(f"매핑의 목적지 라벨이 우리 라벨이 아닙니다: {sorted(unknown)}")

    from transformers import pipeline

    rows = read_jsonl(args.input)
    classifier = pipeline(
        "text-classification", model=args.model, device=-1, top_k=None, truncation=True,
        max_length=args.max_len,
    )
    results = classifier([row.text for row in rows], batch_size=args.batch)

    reachable: set[str] = set()
    predictions = []
    for row, result in zip(rows, results):
        scores = {label: 0.0 for label in LABELS}
        for entry in result:
            target = mapping.get(entry["label"])
            if target is None:
                raise SystemExit(
                    f"모델 라벨 {entry['label']!r}을 매핑할 수 없습니다. --map으로 지정하세요.\n"
                    f"이 모델이 내는 라벨: {sorted({e['label'] for e in result})}"
                )
            # 여러 모델 라벨이 한 우리 라벨로 합쳐질 수 있으므로 확률을 더한다.
            scores[target] += float(entry["score"])
            reachable.add(target)
        label = max(scores, key=scores.get)
        predictions.append(Prediction(row.id, row.text, label, scores[label], scores))

    write_predictions(args.output, predictions)
    print(f"{len(predictions)} predictions -> {args.output}  (model={args.model})")

    missing = [label for label in LABELS if label not in reachable]
    if missing:
        print(
            f"\n주의: 이 모델+매핑은 {', '.join(missing)}를 절대 예측할 수 없다.\n"
            "  -> macro F1과 라벨별 F1은 우리 모델과 비교하면 안 된다 (구조적으로 손해다).\n"
            "  -> attack recall과 benign FPR만 비교한다. 둘은 공격/정상 2분류 지표라 라벨 개수와 무관하다."
        )


if __name__ == "__main__":
    main()
