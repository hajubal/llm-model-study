#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from guardlab.io import write_jsonl
from guardlab.split import group_stratified_split
from guardlab.synth import TEMPLATES, generate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="runs/data/v1")
    parser.add_argument("--n-per-group", type=int, default=8, help="템플릿 하나에서 만들 문장 수(중복 없이)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    target = Path(args.out)
    splits = group_stratified_split(generate(args.n_per_group, args.seed), seed=args.seed)
    for name, rows in splits.items():
        write_jsonl(target / f"{name}.jsonl", rows)
        print(f"{name}: {len(rows)}")
    manifest = {
        "generator": "guardlab.synth.generate",
        "seed": args.seed,
        "n_per_group": args.n_per_group,
        "n_templates": len(TEMPLATES),
        "split_strategy": "label/language/source-stratified group_id split",
        "counts": {name: len(rows) for name, rows in splits.items()},
        "groups": {name: len({row.group_id for row in rows}) for name, rows in splits.items()},
        "limitations": ["synthetic templates", "small vocabulary", "not a product benchmark"],
    }
    target.mkdir(parents=True, exist_ok=True)
    (target / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

