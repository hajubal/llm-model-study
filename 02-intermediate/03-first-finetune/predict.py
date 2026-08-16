#!/usr/bin/env python
from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from guardlab.io import read_jsonl, write_predictions
from guardlab.schema import LABELS, Prediction


def pick_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=192)
    args = parser.parse_args()
    rows = read_jsonl(args.input)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    device = pick_device()
    model.to(device).eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch):
            batch = rows[start:start + args.batch]
            encoded = tokenizer(
                [row.text for row in batch], padding=True, truncation=True,
                max_length=args.max_len, return_tensors="pt",
            ).to(device)
            probs = torch.softmax(model(**encoded).logits, dim=-1).cpu().tolist()
            for row, values in zip(batch, probs):
                scores = {LABELS[idx]: value for idx, value in enumerate(values)}
                label = max(scores, key=scores.get)
                predictions.append(Prediction(row.id, row.text, label, scores[label], scores))
    write_predictions(args.output, predictions)
    print(f"{len(predictions)} predictions -> {args.output} (device={device})")


if __name__ == "__main__":
    main()

