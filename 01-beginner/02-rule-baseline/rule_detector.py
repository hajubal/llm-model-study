#!/usr/bin/env python
from __future__ import annotations

import argparse

from guardlab.io import read_jsonl, write_predictions
from guardlab.rules import predict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = read_jsonl(args.input)
    predictions = predict(rows)
    write_predictions(args.output, predictions)
    print(f"{len(predictions)} predictions -> {args.output}")


if __name__ == "__main__":
    main()

