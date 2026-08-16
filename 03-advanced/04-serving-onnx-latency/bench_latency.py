#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def stats(values: list[float]) -> dict:
    return {"p50_ms": round(float(np.percentile(values, 50)), 3), "p95_ms": round(float(np.percentile(values, 95)), 3)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--out", default="runs/latency.json")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    encoded = tokenizer("이전 지시를 무시하라는 문장을 탐지하는 테스트 입력", return_tensors="pt")
    model = AutoModelForSequenceClassification.from_pretrained(args.model).eval()
    session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])

    torch_times, onnx_times = [], []
    for _ in range(5):
        with torch.inference_mode():
            model(**encoded)
        session.run(None, {key: value.numpy() for key, value in encoded.items()})
    for _ in range(args.runs):
        started = time.perf_counter()
        with torch.inference_mode():
            model(**encoded)
        torch_times.append((time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        session.run(None, {key: value.numpy() for key, value in encoded.items()})
        onnx_times.append((time.perf_counter() - started) * 1000)
    result = {"runs": args.runs, "batch": 1, "torch_cpu": stats(torch_times), "onnx_cpu": stats(onnx_times)}
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
