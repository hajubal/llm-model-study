#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from guardlab.io import read_jsonl
from guardlab.split import assert_no_group_leakage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="train/dev/test.jsonl이 있는 디렉터리")
    args = parser.parse_args()
    root = Path(args.data)
    splits = {name: read_jsonl(root / f"{name}.jsonl") for name in ("train", "dev", "test")}
    assert_no_group_leakage(splits)
    all_ids = [row.id for rows in splits.values() for row in rows]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("split 사이에 중복 id가 있습니다")
    for name, rows in splits.items():
        labels = Counter(row.label for row in rows)
        sources = Counter(row.source for row in rows)
        languages = Counter(row.language for row in rows)
        print(f"{name:5} n={len(rows):4} labels={dict(labels)} sources={dict(sources)} languages={dict(languages)}")
    print("OK: id 중복 없음, group_id leakage 없음")


if __name__ == "__main__":
    main()

