from __future__ import annotations

from dataclasses import dataclass, field

LABELS = ("BENIGN", "PROMPT_INJECTION", "JAILBREAK")
ATTACK_LABELS = frozenset({"PROMPT_INJECTION", "JAILBREAK"})
SOURCES = frozenset({"user", "retrieved", "tool", "system"})
LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}


@dataclass
class Sample:
    id: str
    text: str
    label: str
    source: str = "user"
    language: str = "ko"
    group_id: str = ""
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        obj = {
            "id": self.id,
            "text": self.text,
            "label": self.label,
            "source": self.source,
            "language": self.language,
            "group_id": self.group_id,
        }
        if self.meta:
            obj["meta"] = self.meta
        return obj


@dataclass
class Prediction:
    id: str
    text: str
    label: str
    score: float
    scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        obj = {
            "id": self.id,
            "text": self.text,
            "label": self.label,
            "score": round(float(self.score), 6),
        }
        if self.scores:
            obj["scores"] = {k: round(float(v), 6) for k, v in self.scores.items()}
        return obj


class DataValidationError(ValueError):
    pass


def validate_sample(sample: Sample) -> None:
    if not sample.id.strip():
        raise DataValidationError("id가 비어 있습니다")
    if not sample.text.strip():
        raise DataValidationError(f"{sample.id}: text가 비어 있습니다")
    if sample.label not in LABELS:
        raise DataValidationError(f"{sample.id}: 알 수 없는 label {sample.label!r}")
    if sample.source not in SOURCES:
        raise DataValidationError(f"{sample.id}: 알 수 없는 source {sample.source!r}")
    if not sample.group_id.strip():
        raise DataValidationError(f"{sample.id}: group_id가 비어 있습니다")

