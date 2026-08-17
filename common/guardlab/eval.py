from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .schema import ATTACK_LABELS, LABELS, Prediction, Sample


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        total = self.precision + self.recall
        return 2 * self.precision * self.recall / total if total else 0.0

    @property
    def support(self) -> int:
        return self.tp + self.fn

    def to_dict(self) -> dict:
        return {
            "support": self.support,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


@dataclass
class Report:
    n_samples: int
    accuracy: float
    macro_f1: float
    attack_recall: float
    benign_fpr: float
    per_label: dict[str, Counts]
    confusion: dict[str, dict[str, int]]
    errors: list[dict]
    # guardlab.stats.bootstrap_ci의 결과. 지표 이름 -> (하한, 상한).
    # None이면 구간을 계산하지 않은 것이고, 그 경우 숫자를 다른 실험과 비교하면 안 된다.
    ci: dict[str, tuple[float, float]] | None = None

    def _interval(self, metric: str) -> str:
        if not self.ci or metric not in self.ci:
            return ""
        low, high = self.ci[metric]
        return f" [{low:.3f}–{high:.3f}]"

    def to_dict(self) -> dict:
        obj = {
            "n_samples": self.n_samples,
            "accuracy": round(self.accuracy, 4),
            "macro_f1": round(self.macro_f1, 4),
            "attack_recall": round(self.attack_recall, 4),
            "benign_fpr": round(self.benign_fpr, 4),
            "per_label": {label: self.per_label[label].to_dict() for label in LABELS},
            "confusion": self.confusion,
            "n_errors": len(self.errors),
        }
        if self.ci:
            obj["ci"] = {metric: list(bounds) for metric, bounds in self.ci.items()}
        return obj

    def to_markdown(self, title: str = "평가 결과") -> str:
        lines = [f"### {title}", ""]
        lines.append(
            f"- 샘플 **{self.n_samples}** · accuracy **{self.accuracy:.3f}**"
            f"{self._interval('accuracy')} · "
            f"macro F1 **{self.macro_f1:.3f}**{self._interval('macro_f1')}"
        )
        lines.append(
            f"- attack recall **{self.attack_recall:.3f}**{self._interval('attack_recall')} · "
            f"benign FPR **{self.benign_fpr:.3f}**{self._interval('benign_fpr')}"
        )
        if self.ci:
            lines.append(
                "- 대괄호는 95% bootstrap 신뢰구간이다. 두 실험의 구간이 크게 겹치면 "
                "그 차이는 개선이 아니라 표본 변동일 수 있다"
            )
        # support 0인 라벨은 F1이 0으로 macro 평균에 들어가고, 해당 지표의 분모도 0이 된다.
        # 단일 라벨 데이터(hard negative 모음 등)에서 숫자를 잘못 읽지 않도록 명시한다.
        empty = [label for label in LABELS if self.per_label[label].support == 0]
        if empty:
            lines.append(
                f"- 주의: support가 0인 라벨({', '.join(empty)})의 F1 0이 macro 평균에 포함된다. "
                "이 데이터에서는 macro F1을 다른 데이터와 비교하지 않는다"
            )
        if not any(self.per_label[label].support for label in ATTACK_LABELS):
            lines.append("- 주의: 공격 샘플이 없어 attack recall 0.000은 '미탐'이 아니라 '해당 없음'이다")
        lines.extend(["", "| 라벨 | support | P | R | F1 |", "|---|---:|---:|---:|---:|"])
        for label in LABELS:
            counts = self.per_label[label]
            lines.append(
                f"| {label} | {counts.support} | {counts.precision:.3f} | "
                f"{counts.recall:.3f} | {counts.f1:.3f} |"
            )
        lines.extend(["", "Confusion matrix (행=정답, 열=예측)", ""])
        lines.append("| gold \\ pred | " + " | ".join(LABELS) + " |")
        lines.append("|---|" + "---:|" * len(LABELS))
        for gold in LABELS:
            lines.append("| " + gold + " | " + " | ".join(str(self.confusion[gold][pred]) for pred in LABELS) + " |")
        return "\n".join(lines)


def evaluate(gold: list[Sample], predictions: list[Prediction], collect_errors: bool = True) -> Report:
    pred_by_id = {pred.id: pred for pred in predictions}
    if len(pred_by_id) != len(predictions):
        raise ValueError("prediction id가 중복되었습니다")
    gold_ids = {sample.id for sample in gold}
    missing = gold_ids - pred_by_id.keys()
    extra = pred_by_id.keys() - gold_ids
    if missing or extra:
        raise ValueError(f"gold/pred id 불일치 missing={sorted(missing)[:5]} extra={sorted(extra)[:5]}")

    confusion = {g: {p: 0 for p in LABELS} for g in LABELS}
    per_label = {label: Counts() for label in LABELS}
    errors: list[dict] = []
    correct = 0
    benign_total = benign_fp = attack_total = attack_hit = 0

    for sample in gold:
        pred = pred_by_id[sample.id]
        if pred.text != sample.text:
            raise ValueError(f"{sample.id}: gold/pred text 불일치")
        if pred.label not in LABELS:
            raise ValueError(f"{sample.id}: 알 수 없는 prediction label {pred.label!r}")
        confusion[sample.label][pred.label] += 1
        if pred.label == sample.label:
            correct += 1
        elif collect_errors:
            errors.append({
                "id": sample.id,
                "text": sample.text,
                "gold": sample.label,
                "pred": pred.label,
                "score": pred.score,
                "source": sample.source,
                "language": sample.language,
                "group_id": sample.group_id,
            })

        if sample.label == "BENIGN":
            benign_total += 1
            benign_fp += int(pred.label in ATTACK_LABELS)
        else:
            attack_total += 1
            attack_hit += int(pred.label in ATTACK_LABELS)

    for label in LABELS:
        per_label[label].tp = confusion[label][label]
        per_label[label].fn = sum(confusion[label][other] for other in LABELS if other != label)
        per_label[label].fp = sum(confusion[other][label] for other in LABELS if other != label)

    n = len(gold)
    return Report(
        n_samples=n,
        accuracy=correct / n if n else 0.0,
        macro_f1=sum(per_label[label].f1 for label in LABELS) / len(LABELS),
        attack_recall=attack_hit / attack_total if attack_total else 0.0,
        benign_fpr=benign_fp / benign_total if benign_total else 0.0,
        per_label=per_label,
        confusion=confusion,
        errors=errors,
    )


def slice_reports(gold: list[Sample], predictions: list[Prediction], field: str) -> dict[str, Report]:
    if field not in {"source", "language"}:
        raise ValueError("slice field는 source 또는 language만 지원합니다")
    pred_by_id = {pred.id: pred for pred in predictions}
    values = sorted({getattr(sample, field) for sample in gold})
    reports = {}
    for value in values:
        subset = [sample for sample in gold if getattr(sample, field) == value]
        reports[value] = evaluate(subset, [pred_by_id[sample.id] for sample in subset], collect_errors=False)
    return reports


def error_summary(report: Report) -> dict[str, int]:
    return dict(Counter(f"{item['gold']}->{item['pred']}" for item in report.errors))

