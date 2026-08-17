"""표본이 작을 때 숫자를 정직하게 읽기 위한 도구.

bootstrap 신뢰구간과 확률 보정(calibration)을 다룬다. 둘 다 "점 추정 하나만 보면
틀린 결론을 내리게 되는" 상황을 막기 위한 것이다.

- bootstrap: 24건짜리 벤치에서 macro F1 0.837과 0.750의 차이가 의미 있는지 판단한다.
- calibration: threshold 0.8이 "80% 확률"을 뜻하지 않는다는 사실을 실측으로 보여준다.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .schema import ATTACK_LABELS, LABELS, Prediction, Sample

# bootstrap으로 구간을 낼 지표. Report의 필드 이름과 같게 맞춘다.
METRICS = ("accuracy", "macro_f1", "attack_recall", "benign_fpr")


def _metrics(pairs: list[tuple[str, str]]) -> dict[str, float]:
    """(gold, pred) 쌍에서 지표를 계산한다. evaluate()와 같은 정의를 쓴다.

    evaluate()를 재사용하지 않는 이유: bootstrap은 같은 샘플을 여러 번 뽑으므로
    id가 중복되고, evaluate()는 중복 id를 오류로 처리한다.
    """
    tp = {label: 0 for label in LABELS}
    fp = {label: 0 for label in LABELS}
    fn = {label: 0 for label in LABELS}
    correct = benign_total = benign_fp = attack_total = attack_hit = 0

    for gold, pred in pairs:
        if gold == pred:
            correct += 1
            tp[gold] += 1
        else:
            fn[gold] += 1
            fp[pred] += 1
        if gold == "BENIGN":
            benign_total += 1
            benign_fp += int(pred in ATTACK_LABELS)
        else:
            attack_total += 1
            attack_hit += int(pred in ATTACK_LABELS)

    f1s = []
    for label in LABELS:
        precision = tp[label] / (tp[label] + fp[label]) if tp[label] + fp[label] else 0.0
        recall = tp[label] / (tp[label] + fn[label]) if tp[label] + fn[label] else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)

    n = len(pairs)
    return {
        "accuracy": correct / n if n else 0.0,
        "macro_f1": sum(f1s) / len(LABELS),
        "attack_recall": attack_hit / attack_total if attack_total else 0.0,
        "benign_fpr": benign_fp / benign_total if benign_total else 0.0,
    }


def _percentile(values: list[float], q: float) -> float:
    """numpy 없이 선형보간 백분위수를 구한다. guardlab은 의존성이 없어야 한다."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def bootstrap_ci(
    gold: list[Sample],
    predictions: list[Prediction],
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, tuple[float, float]]:
    """샘플을 복원추출로 n_boot번 다시 뽑아 지표의 percentile 신뢰구간을 만든다.

    구간이 넓다는 것은 "표본이 작아서 이 숫자를 믿을 수 없다"는 뜻이다. 두 실험의
    구간이 크게 겹치면 그 차이는 개선이 아니라 표본 변동일 수 있다.

    한계: bootstrap은 원본 표본이 모집단을 대표한다고 가정한다. 합성 데이터에서
    나온 구간은 "이 합성 분포 안에서의 불확실성"이지 실제 트래픽 성능이 아니다.
    """
    if n_boot < 1:
        raise ValueError("n_boot는 1 이상이어야 합니다")
    if not 0 < alpha < 1:
        raise ValueError("alpha는 0과 1 사이여야 합니다")
    pred_by_id = {pred.id: pred for pred in predictions}
    missing = sorted({sample.id for sample in gold} - pred_by_id.keys())
    if missing:
        raise ValueError(f"예측에 없는 id {missing[:5]} (총 {len(missing)}건)")

    pairs = [(sample.label, pred_by_id[sample.id].label) for sample in gold]
    if not pairs:
        raise ValueError("gold가 비어 있습니다")

    rng = random.Random(seed)
    samples: dict[str, list[float]] = {metric: [] for metric in METRICS}
    for _ in range(n_boot):
        resampled = [pairs[rng.randrange(len(pairs))] for _ in range(len(pairs))]
        for metric, value in _metrics(resampled).items():
            samples[metric].append(value)

    return {
        metric: (
            round(_percentile(values, alpha / 2), 4),
            round(_percentile(values, 1 - alpha / 2), 4),
        )
        for metric, values in samples.items()
    }


def attack_probability(pred: Prediction) -> float:
    """모델이 이 입력을 공격이라고 본 확률.

    3-class softmax에서는 P(PROMPT_INJECTION) + P(JAILBREAK)다. scores가 없는
    예측 파일은 라벨과 score만으로 되돌린다.
    """
    if pred.scores:
        return sum(pred.scores.get(label, 0.0) for label in ATTACK_LABELS)
    return pred.score if pred.label in ATTACK_LABELS else 1.0 - pred.score


@dataclass
class Bin:
    """reliability diagram의 구간 하나."""

    low: float
    high: float
    count: int
    mean_confidence: float  # 모델이 말한 확률의 평균
    observed_rate: float  # 실제로 공격이었던 비율

    @property
    def gap(self) -> float:
        return self.mean_confidence - self.observed_rate

    def to_dict(self) -> dict:
        return {
            "low": round(self.low, 3),
            "high": round(self.high, 3),
            "count": self.count,
            "mean_confidence": round(self.mean_confidence, 4),
            "observed_rate": round(self.observed_rate, 4),
            "gap": round(self.gap, 4),
        }


def reliability_bins(
    gold: list[Sample], predictions: list[Prediction], n_bins: int = 10
) -> list[Bin]:
    """공격 확률을 구간으로 나눠 "말한 확률" vs "실제 비율"을 비교한다.

    보정이 잘 된 모델은 두 값이 같다. 파인튜닝한 분류기는 보통 과신하므로
    mean_confidence가 observed_rate보다 높게 나온다.
    """
    if n_bins < 1:
        raise ValueError("n_bins는 1 이상이어야 합니다")
    pred_by_id = {pred.id: pred for pred in predictions}
    missing = sorted({sample.id for sample in gold} - pred_by_id.keys())
    if missing:
        raise ValueError(f"예측에 없는 id {missing[:5]} (총 {len(missing)}건)")

    buckets: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for sample in gold:
        probability = attack_probability(pred_by_id[sample.id])
        # 1.0은 마지막 구간에 넣는다. int(1.0 * n_bins)는 범위를 벗어난다.
        index = min(int(probability * n_bins), n_bins - 1)
        buckets[index].append((probability, int(sample.label in ATTACK_LABELS)))

    bins = []
    for index, bucket in enumerate(buckets):
        low, high = index / n_bins, (index + 1) / n_bins
        if not bucket:
            bins.append(Bin(low, high, 0, 0.0, 0.0))
            continue
        bins.append(Bin(
            low=low,
            high=high,
            count=len(bucket),
            mean_confidence=sum(probability for probability, _ in bucket) / len(bucket),
            observed_rate=sum(outcome for _, outcome in bucket) / len(bucket),
        ))
    return bins


def expected_calibration_error(
    gold: list[Sample], predictions: list[Prediction], n_bins: int = 10
) -> float:
    """ECE: 구간별 |말한 확률 - 실제 비율|을 샘플 수로 가중평균한 값.

    0에 가까울수록 확률이 믿을 만하다. ECE가 크면 threshold 0.8은
    "80% 확률"이 아니라 그냥 "이 모델에서 상위 몇 %"라는 뜻일 뿐이다.
    """
    bins = reliability_bins(gold, predictions, n_bins)
    total = sum(bucket.count for bucket in bins)
    if not total:
        return 0.0
    return sum(bucket.count * abs(bucket.gap) for bucket in bins) / total


def reliability_markdown(bins: list[Bin], ece: float) -> str:
    """reliability diagram을 터미널에서 읽을 수 있는 표와 막대로 만든다."""
    lines = [f"ECE **{ece:.4f}** (0에 가까울수록 확률이 믿을 만하다)", ""]
    lines.append("| 구간 | n | 말한 확률 | 실제 공격 비율 | 차이 |")
    lines.append("|---|---:|---:|---:|---:|")
    for bucket in bins:
        if not bucket.count:
            continue
        bar = "#" * round(abs(bucket.gap) * 20)
        sign = "과신" if bucket.gap > 0 else "과소"
        lines.append(
            f"| {bucket.low:.1f}–{bucket.high:.1f} | {bucket.count} | "
            f"{bucket.mean_confidence:.3f} | {bucket.observed_rate:.3f} | "
            f"{bucket.gap:+.3f} {sign} {bar} |"
        )
    return "\n".join(lines)
