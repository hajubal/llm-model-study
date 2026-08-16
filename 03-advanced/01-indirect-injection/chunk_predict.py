#!/usr/bin/env python
from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from guardlab.io import read_jsonl, write_predictions
from guardlab.schema import ATTACK_LABELS, LABELS, Prediction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-len", type=int, default=192)
    parser.add_argument("--stride", type=int, default=48)
    args = parser.parse_args()
    rows = read_jsonl(args.input)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    outputs = []
    with torch.inference_mode():
        for row in rows:
            windows = tokenizer(
                row.text, truncation=True, max_length=args.max_len, stride=args.stride,
                return_overflowing_tokens=True, padding=True, return_tensors="pt",
            )
            windows.pop("overflow_to_sample_mapping", None)
            probabilities = torch.softmax(model(**windows.to(device)).logits, dim=-1).cpu()
            attack_indices = [LABELS.index(label) for label in ATTACK_LABELS]
            winning_window = int(probabilities[:, attack_indices].sum(dim=1).argmax())
            values = probabilities[winning_window].tolist()
            scores = {label: values[idx] for idx, label in enumerate(LABELS)}
            label = max(scores, key=scores.get)
            outputs.append(Prediction(row.id, row.text, label, scores[label], scores))
    write_predictions(args.output, outputs)
    print(f"{len(outputs)} predictions -> {args.output} (device={device})")


if __name__ == "__main__":
    main()

