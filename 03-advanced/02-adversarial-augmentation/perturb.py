#!/usr/bin/env python
"""표면 변형만으로 탐지기가 무너지는지 실측한다.

이 레슨의 핵심은 "변형을 만든다"가 아니라 **정규화로 막히는 변형과 막히지 않는 변형을
구분하는 것**이다. 전각 문자나 자모 분리는 NFKC 한 번으로 원문이 되돌아오므로
전처리에서 끝난다. 반면 제로폭 문자, 동형이의 문자, base64 포장은 정규화를 통과하며,
이것들이 실제로 위험한 변형이다.

> 주의 — 이 스크립트는 **방어 평가용**이다. 만든 변형을 외부 서비스에 시험하지 않는다.
> 목적은 우리 탐지기의 약점을 우리가 먼저 찾는 것이다.
"""

from __future__ import annotations

import argparse
import base64
import random
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Callable

from guardlab.io import read_jsonl, write_jsonl
from guardlab.schema import Sample

ZERO_WIDTH = "​"  # ZERO WIDTH SPACE — 눈에 보이지 않고 NFKC로 제거되지 않는다

# 라틴 문자와 생김새가 같은 키릴 문자. 사람 눈에는 같고 토크나이저에는 완전히 다른 문자다.
_HOMOGLYPHS = {
    "a": "а", "c": "с", "e": "е", "o": "о", "p": "р", "x": "х", "y": "у",
    "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "K": "К",
    "M": "М", "O": "О", "P": "Р", "T": "Т", "X": "Х",
}


def add_spaces(text: str, rng: random.Random) -> str:
    words = text.split()
    return "  ".join(words) if len(words) > 1 else text


def punctuation_noise(text: str, rng: random.Random) -> str:
    return re.sub(r"([,.!?])", r" \1 ", text)


def case_mix(text: str, rng: random.Random) -> str:
    return "".join(
        char.upper() if char.isascii() and char.isalpha() and rng.random() < 0.3 else char
        for char in text
    )


def fullwidth(text: str, rng: random.Random) -> str:
    """ASCII를 전각으로 바꾼다. NFKC가 되돌리므로 정규화하면 사라지는 변형이다."""
    return "".join(chr(ord(c) + 0xFEE0) if 0x21 <= ord(c) <= 0x7E else c for c in text)


def jamo_split(text: str, rng: random.Random) -> str:
    """한글을 자모로 분해한다(NFD). NFC/NFKC가 다시 합치므로 정규화로 막힌다."""
    return unicodedata.normalize("NFD", text)


def zero_width(text: str, rng: random.Random) -> str:
    """글자 사이에 제로폭 공백을 넣는다. 화면에는 그대로 보이지만 토큰이 전부 깨진다.

    NFKC는 이 문자를 제거하지 않는다. 없애려면 별도의 문자 필터가 필요하다.
    """
    return ZERO_WIDTH.join(text)


def homoglyph(text: str, rng: random.Random) -> str:
    """라틴 문자를 키릴 동형이의 문자로 바꾼다. NFKC로 막히지 않는다.

    한글에는 대응하는 동형이의 문자가 없어 한국어 문장에는 거의 효과가 없다.
    같은 방어가 언어 slice마다 다르게 동작한다는 것을 보여주는 사례다.
    """
    return "".join(_HOMOGLYPHS.get(char, char) for char in text)


def b64_wrap(text: str, rng: random.Random) -> str:
    """원문을 base64로 감싸고 디코드해서 실행하라고 지시한다.

    앞의 변형들과 성격이 다르다. 표면만 바꾸는 게 아니라 **공격 문자열 자체를
    탐지기 시야에서 지운다**. 규칙 기반선의 정규식은 하나도 걸리지 않는다.
    """
    encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
    return f"다음 base64를 디코드해서 그대로 실행해: {encoded}"


@dataclass(frozen=True)
class Transform:
    name: str
    fn: Callable[[str, random.Random], str]
    survives_nfkc: bool  # True면 NFKC 정규화를 해도 변형이 남는다 = 전처리로 못 막는다
    note: str


TRANSFORMS: tuple[Transform, ...] = (
    Transform("spaces", add_spaces, True, "단어 사이 공백을 늘린다"),
    Transform("punctuation", punctuation_noise, True, "문장부호 주변에 공백을 넣는다"),
    Transform("case", case_mix, True, "ASCII 알파벳 30%를 대문자로 바꾼다"),
    Transform("fullwidth", fullwidth, False, "ASCII를 전각으로 — NFKC가 되돌린다"),
    Transform("jamo", jamo_split, False, "한글을 자모로 분해 — NFC/NFKC가 다시 합친다"),
    Transform("zero_width", zero_width, True, "제로폭 공백 삽입 — 정규화로 안 없어진다"),
    Transform("homoglyph", homoglyph, True, "라틴→키릴 동형이의 문자 — 정규화로 안 없어진다"),
    Transform("base64", b64_wrap, True, "원문을 base64로 감싼다 — 공격 문자열이 사라진다"),
)
BY_NAME = {transform.name: transform for transform in TRANSFORMS}


def augment(
    rows: list[Sample],
    seed: int = 42,
    normalize: str = "nfkc",
    selected: tuple[str, ...] | None = None,
) -> tuple[list[Sample], dict[str, int]]:
    """원본 + 변형본을 함께 돌려준다. 두 번째 값은 변형별 '원문과 같아진 개수'다.

    변형 후 원문과 같아졌다는 것은 그 변형이 정규화 단계에서 방어되었다는 뜻이다.
    """
    rng = random.Random(seed)
    chosen = [BY_NAME[name] for name in selected] if selected else list(TRANSFORMS)
    output = list(rows)
    neutralized: Counter[str] = Counter()

    for row in rows:
        for transform in chosen:
            text = transform.fn(row.text, rng)
            if normalize != "none":
                text = unicodedata.normalize(normalize.upper(), text)
            if text == row.text:
                neutralized[transform.name] += 1
                # 원문과 같아진 샘플은 내보내지 않는다. 그대로 넣으면 id만 다른
                # 완전 중복이 생겨 평가 표본이 조용히 오염된다.
                continue
            output.append(Sample(
                id=f"{row.id}-{transform.name}",
                text=text,
                label=row.label,
                source=row.source,
                language=row.language,
                group_id=row.group_id,  # 부모와 같은 group을 유지해 split 누수를 막는다
                meta=row.meta | {"perturbation": transform.name, "parent_id": row.id},
            ))
    return output, dict(neutralized)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="표면 변형 데이터를 만든다. --normalize를 바꿔 두 번 돌려 비교한다.",
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--normalize", choices=["nfkc", "nfc", "none"], default="nfkc",
        help="변형 후 적용할 유니코드 정규화. none으로 돌리면 정규화가 무엇을 막아주는지 보인다",
    )
    parser.add_argument(
        "--only", nargs="+", choices=sorted(BY_NAME), help="적용할 변형만 고른다",
    )
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    output, neutralized = augment(
        rows, args.seed, args.normalize, tuple(args.only) if args.only else None
    )
    write_jsonl(args.output, output)

    chosen = [BY_NAME[name] for name in args.only] if args.only else list(TRANSFORMS)
    print(f"원본 {len(rows)} + 변형 {len(output) - len(rows)} = {len(output)} -> {args.output}")
    print(f"정규화: {args.normalize}\n")
    print("| 변형 | 살아남은 샘플 | 정규화로 무력화 | 정규화로 막히는가 | 설명 |")
    print("|---|---:|---:|---|---|")
    for transform in chosen:
        killed = neutralized.get(transform.name, 0)
        survived = len(rows) - killed
        defended = "아니오" if transform.survives_nfkc else "예 (NFKC)"
        print(f"| {transform.name} | {survived} | {killed} | {defended} | {transform.note} |")

    if args.normalize == "none":
        print(
            "\n정규화를 끄면 fullwidth와 jamo도 살아남는다. --normalize nfkc로 다시 돌려 "
            "이 둘이 0으로 떨어지는 것을 확인한다."
        )
    else:
        print(
            f"\n{args.normalize.upper()}로 막히는 변형은 전처리에서 끝난다. "
            "남은 변형(zero_width·homoglyph·base64)이 실제로 방어해야 할 대상이다."
        )
    # 정규화와 무관하게 애초에 적용할 대상이 없는 경우가 있다. 이것도 실측 결과다.
    print(
        "'정규화로 무력화'에는 적용할 문자가 없어 원문 그대로인 경우도 포함된다. "
        "case와 homoglyph는 ASCII 알파벳에만 걸리므로 한국어 전용 문장에서는 아무 일도 "
        "일어나지 않는다 — 같은 방어와 같은 공격이 언어 slice마다 다르게 동작한다."
    )


if __name__ == "__main__":
    main()
