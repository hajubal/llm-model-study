#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from guardlab.io import read_jsonl, read_predictions
from guardlab.rules import predict_one
from guardlab.schema import ATTACK_LABELS


def decide(sample, prediction, config: dict) -> tuple[str, list[str]]:
    attack_score = sum(prediction.scores.get(label, 0.0) for label in ATTACK_LABELS)
    reasons = []
    if attack_score >= config["block_threshold"]:
        reasons.append("model_block_threshold")
        return "block", reasons
    rule = predict_one(sample)
    if rule.label in ATTACK_LABELS:
        reasons.append(f"rule:{rule.label}")
    if attack_score >= config["review_threshold"]:
        reasons.append("model_review_threshold")
    if reasons:
        return config.get("rule_match_action", "review"), reasons
    return "allow", ["below_thresholds"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="gold 형식 입력")
    parser.add_argument("--pred", required=True, help="모델 prediction JSONL")
    parser.add_argument("--config", default="03-advanced/03-hybrid-gates/policy.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    samples = read_jsonl(args.input)
    predictions = {row.id: row for row in read_predictions(args.pred)}
    config = json.load(open(args.config, encoding="utf-8"))
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as stream:
        for sample in samples:
            action, reasons = decide(sample, predictions[sample.id], config)
            stream.write(json.dumps({"id": sample.id, "action": action, "reasons": reasons}, ensure_ascii=False) + "\n")
    print(f"{len(samples)} decisions -> {target}")


if __name__ == "__main__":
    main()
