from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .schema import DataValidationError, Prediction, Sample, validate_sample


def _sample_from_dict(obj: dict) -> Sample:
    return Sample(
        id=str(obj["id"]),
        text=str(obj["text"]),
        label=str(obj["label"]),
        source=str(obj.get("source", "user")),
        language=str(obj.get("language", "ko")),
        group_id=str(obj.get("group_id", "")),
        meta=obj.get("meta", {}) or {},
    )


def read_jsonl(path: str | Path, validate: bool = True) -> list[Sample]:
    samples: list[Sample] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                sample = _sample_from_dict(json.loads(line))
                if validate:
                    validate_sample(sample)
                if sample.id in seen:
                    raise DataValidationError(f"중복 id {sample.id!r}")
            except Exception as exc:
                raise DataValidationError(f"{path}:{line_no}: {exc}") from exc
            seen.add(sample.id)
            samples.append(sample)
    return samples


def write_jsonl(path: str | Path, samples: Iterable[Sample]) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(target, "w", encoding="utf-8") as stream:
        for sample in samples:
            validate_sample(sample)
            stream.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def read_predictions(path: str | Path) -> list[Prediction]:
    predictions: list[Prediction] = []
    with open(path, encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                predictions.append(Prediction(
                    id=str(obj["id"]), text=str(obj["text"]), label=str(obj["label"]),
                    score=float(obj.get("score", 0.0)),
                    scores={str(key): float(value) for key, value in (obj.get("scores", {}) or {}).items()},
                ))
            except Exception as exc:
                raise DataValidationError(f"{path}:{line_no}: {exc}") from exc
    return predictions


def write_predictions(path: str | Path, predictions: Iterable[Prediction]) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(target, "w", encoding="utf-8") as stream:
        for pred in predictions:
            stream.write(json.dumps(pred.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count
