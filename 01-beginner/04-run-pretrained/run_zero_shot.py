#!/usr/bin/env python
from __future__ import annotations

import argparse

from guardlab.io import read_jsonl, write_predictions
from guardlab.schema import Prediction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    from transformers import pipeline

    rows = read_jsonl(args.input)
    classifier = pipeline("zero-shot-classification", model=args.model, device=-1)
    names = {
        "BENIGN": "a normal user request",
        "PROMPT_INJECTION": "an attempt to override the model instructions",
        "JAILBREAK": "an attempt to bypass AI safety restrictions",
    }
    results = classifier(
        [row.text for row in rows], candidate_labels=list(names.values()),
        hypothesis_template="This text is {}.", multi_label=False, batch_size=args.batch,
    )
    reverse = {value: key for key, value in names.items()}
    predictions = []
    for row, result in zip(rows, results):
        scores = {reverse[label]: score for label, score in zip(result["labels"], result["scores"])}
        label = max(scores, key=scores.get)
        predictions.append(Prediction(row.id, row.text, label, scores[label], scores))
    write_predictions(args.output, predictions)
    print(f"{len(predictions)} predictions -> {args.output}")


if __name__ == "__main__":
    main()

