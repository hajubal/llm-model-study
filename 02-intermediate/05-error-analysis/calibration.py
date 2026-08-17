#!/usr/bin/env python
"""모델이 말한 확률이 실제 확률인지 검사한다(reliability diagram + ECE).

threshold를 다루는 레슨의 빠진 절반이다. `policy.json`의 `block_threshold: 0.80`은
"공격일 확률 80% 이상이면 차단"처럼 읽히지만, 파인튜닝한 분류기의 softmax 값은 보정된
확률이 아니다. 보통 과신한다 — 0.9라고 말한 것들 중 실제 공격은 0.7뿐인 식이다.

그러면 threshold 0.80은 확률적 의미가 없고, 그냥 "이 모델에서 상위 몇 %"라는 뜻일
뿐이다. 그것 자체는 정책으로 쓸 수 있지만, **모델을 재학습하면 같은 0.80이 다른 지점을
가리킨다.** 그래서 모델을 바꾸면 threshold를 dev에서 다시 정해야 한다.

이 스크립트는 그 사실을 실측으로 보여준다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from guardlab.io import read_jsonl, read_predictions
from guardlab.stats import (
    attack_probability,
    expected_calibration_error,
    reliability_bins,
    reliability_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--out", help="결과를 저장할 json 경로")
    args = parser.parse_args()

    gold = read_jsonl(args.gold)
    pred = read_predictions(args.pred)
    if not any(p.scores for p in pred):
        print(
            "주의: 예측 파일에 scores가 없다. label과 score만으로 공격 확률을 되돌리므로 "
            "구간이 거칠게 나온다. predict.py는 scores를 저장한다."
        )

    bins = reliability_bins(gold, pred, args.bins)
    ece = expected_calibration_error(gold, pred, args.bins)
    print(reliability_markdown(bins, ece))

    # 확신이 높은 구간에서 실제로 얼마나 맞는지가 threshold 설계에 직접 쓰인다.
    high = [bucket for bucket in bins if bucket.low >= 0.8 and bucket.count]
    if high:
        said = sum(b.mean_confidence * b.count for b in high) / sum(b.count for b in high)
        real = sum(b.observed_rate * b.count for b in high) / sum(b.count for b in high)
        print(
            f"\n확률 0.8 이상이라고 말한 {sum(b.count for b in high)}건: "
            f"모델이 말한 평균 {said:.3f}, 실제 공격 비율 {real:.3f}"
        )
        if said - real > 0.05:
            print("  -> 과신이다. threshold 0.8은 '80% 확률'이 아니다.")
        elif real - said > 0.05:
            print("  -> 과소평가다. 모델이 실제보다 자신 없어 한다.")
        else:
            print("  -> 이 구간은 비교적 잘 보정되어 있다.")
    else:
        print("\n0.8 이상 구간에 샘플이 없다. threshold를 그 위에 두면 아무것도 차단하지 않는다.")

    probabilities = sorted(attack_probability(p) for p in pred)
    if probabilities:
        middle = probabilities[len(probabilities) // 2]
        edge = sum(1 for value in probabilities if value < 0.05 or value > 0.95)
        print(
            f"\n공격 확률 분포: 중앙값 {middle:.3f} · 0.05 미만이거나 0.95 초과인 예측 "
            f"{edge}/{len(probabilities)}건"
        )
        if edge / len(probabilities) > 0.8:
            print(
                "  -> 예측이 양 끝에 몰려 있다. 과신한 모델의 전형적인 모습이고, "
                "이 상태에서는 threshold를 조금 움직여도 결과가 거의 안 바뀐다."
            )

    print(
        "\n보정을 고치는 방법(이 커리큘럼 범위 밖): dev에서 temperature scaling을 맞추거나 "
        "isotonic regression을 쓴다. 다만 **보정을 고쳐도 분류 성능(macro F1)은 그대로다.** "
        "보정은 순위를 바꾸지 않고 확률의 눈금만 바꾼다."
    )

    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {"ece": round(ece, 5), "n_bins": args.bins, "bins": [b.to_dict() for b in bins]},
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\n-> {target}")


if __name__ == "__main__":
    main()
