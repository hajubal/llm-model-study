#!/usr/bin/env python
from __future__ import annotations

import argparse
import random
import re
import unicodedata

from guardlab.io import read_jsonl, write_jsonl
from guardlab.schema import Sample


def add_spaces(text: str, rng: random.Random) -> str:
    words = text.split()
    return "  ".join(words) if len(words) > 1 else text


def punctuation_noise(text: str, rng: random.Random) -> str:
    return re.sub(r"([,.!?])", r" \1 ", text)


def case_mix(text: str, rng: random.Random) -> str:
    return "".join(char.upper() if char.isascii() and char.isalpha() and rng.random() < 0.3 else char for char in text)


TRANSFORMS = {"spaces": add_spaces, "punctuation": punctuation_noise, "case": case_mix}


def augment(rows: list[Sample], seed: int = 42) -> list[Sample]:
    rng = random.Random(seed)
    output = list(rows)
    for row in rows:
        for name, transform in TRANSFORMS.items():
            text = unicodedata.normalize("NFKC", transform(row.text, rng))
            output.append(Sample(
                id=f"{row.id}-{name}", text=text, label=row.label, source=row.source,
                language=row.language, group_id=row.group_id,
                meta=row.meta | {"perturbation": name, "parent_id": row.id},
            ))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = augment(read_jsonl(args.input), args.seed)
    write_jsonl(args.output, rows)
    print(f"{len(rows)} samples -> {args.output}")


if __name__ == "__main__":
    main()

