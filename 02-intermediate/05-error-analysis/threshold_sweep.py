#!/usr/bin/env python
from __future__ import annotations

import argparse

from guardlab.io import read_jsonl, read_predictions
from guardlab.schema import ATTACK_LABELS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--step", type=float, default=0.05)
    args = parser.parse_args()
    gold = read_jsonl(args.gold)
    predictions = {row.id: row for row in read_predictions(args.pred)}
    missing = sorted({sample.id for sample in gold} - predictions.keys())
    if missing:
        raise SystemExit(f"gold/pred id 불일치: 예측에 없는 id {missing[:5]} (총 {len(missing)}건)")
    print("threshold\tattack_recall\tbenign_fpr")
    threshold = 0.0
    while threshold <= 1.000001:
        attack_total = attack_hit = benign_total = benign_fp = 0
        for sample in gold:
            pred = predictions[sample.id]
            attack_score = sum(pred.scores.get(label, 0.0) for label in ATTACK_LABELS)
            predicted_attack = attack_score >= threshold
            if sample.label == "BENIGN":
                benign_total += 1
                benign_fp += predicted_attack
            else:
                attack_total += 1
                attack_hit += predicted_attack
        recall = attack_hit / attack_total if attack_total else 0.0
        fpr = benign_fp / benign_total if benign_total else 0.0
        print(f"{threshold:.2f}\t{recall:.4f}\t{fpr:.4f}")
        threshold += args.step


if __name__ == "__main__":
    main()

