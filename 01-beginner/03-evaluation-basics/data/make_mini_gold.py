#!/usr/bin/env python
from pathlib import Path

from guardlab.io import read_jsonl, write_jsonl

root = Path(__file__).resolve().parents[3]
rows = read_jsonl(root / "common/data/bench/gold.jsonl")
selected = rows[0:4] + rows[8:12] + rows[16:20]
target = Path(__file__).with_name("mini_gold.jsonl")
write_jsonl(target, selected)
print(f"{len(selected)} samples -> {target}")

