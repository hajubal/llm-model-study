from __future__ import annotations

import re

from .schema import Prediction, Sample

# 교육용 고정 기준선이다. 문맥 이해가 없고 표현 변형에 취약하다는 사실을 측정하기 위한 코드다.
_INJECTION_PATTERNS = [
    re.compile(r"\b(ignore|disregard|forget)\b.{0,40}\b(previous|prior|system|instructions?)\b", re.I),
    re.compile(r"(이전|앞선|기존).{0,20}(지시|명령|규칙).{0,15}(무시|잊어|폐기)", re.I),
    re.compile(r"\b(system prompt|developer message)\b.{0,30}\b(show|reveal|print|repeat)\b", re.I),
    re.compile(r"(시스템|개발자)\s*(프롬프트|메시지).{0,20}(보여|공개|출력|반복)", re.I),
]

_JAILBREAK_PATTERNS = [
    re.compile(r"\b(jailbreak|DAN mode|developer mode|no restrictions?)\b", re.I),
    re.compile(r"(안전|보안|윤리).{0,20}(정책|제한|필터).{0,15}(해제|우회|무시)", re.I),
    re.compile(r"(제한|검열)\s*(없는|없이).{0,15}(모드|AI|답변)", re.I),
]

# 공격 용어를 설명·분석하는 정상 요청은 단순 키워드 규칙의 대표적 오탐이다.
_EDUCATIONAL_CONTEXT = re.compile(
    r"(정의|설명|탐지|분류|연구|논문|교육|예시를 분석|방어|what is|explain|detect|research|paper|defen[cs]e)",
    re.I,
)


def predict_one(sample: Sample) -> Prediction:
    injection_hits = sum(bool(pattern.search(sample.text)) for pattern in _INJECTION_PATTERNS)
    jailbreak_hits = sum(bool(pattern.search(sample.text)) for pattern in _JAILBREAK_PATTERNS)
    discount = 1 if _EDUCATIONAL_CONTEXT.search(sample.text) else 0
    injection_score = max(0.0, min(0.99, 0.25 + 0.35 * injection_hits - 0.25 * discount)) if injection_hits else 0.02
    jailbreak_score = max(0.0, min(0.99, 0.25 + 0.35 * jailbreak_hits - 0.25 * discount)) if jailbreak_hits else 0.02
    benign_score = max(0.01, 1.0 - max(injection_score, jailbreak_score))
    scores = {
        "BENIGN": benign_score,
        "PROMPT_INJECTION": injection_score,
        "JAILBREAK": jailbreak_score,
    }
    label = max(scores, key=scores.get)
    return Prediction(sample.id, sample.text, label, scores[label], scores)


def predict(samples: list[Sample]) -> list[Prediction]:
    return [predict_one(sample) for sample in samples]

