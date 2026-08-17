#!/usr/bin/env python
"""지연시간과 처리량을 재고, 그것으로 요청당 원가를 추정한다.

p50/p95만으로는 도입 결정을 못 한다. 실무에서 갈리는 절반은 비용이다.
"우리 모델을 자체 운영할 것인가, LLM API에 판정을 맡길 것인가"는 정확도가 아니라
보통 여기서 결정된다.

원가 추정은 정확한 회계가 아니라 **자릿수 비교**다. 10배 차이인지 1000배 차이인지만
알면 결정에는 충분하다.
"""

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
    return {
        "p50_ms": round(float(np.percentile(values, 50)), 3),
        "p95_ms": round(float(np.percentile(values, 95)), 3),
        "p99_ms": round(float(np.percentile(values, 99)), 3),
    }


def cost_per_million(throughput_rps: float, hourly_usd: float) -> float:
    """처리량과 시간당 인스턴스 요금으로 100만 요청 원가를 낸다.

    가정: 인스턴스 하나를 100% 활용한다. 실제로는 트래픽이 몰렸다 빠지므로
    활용률이 30~50%면 원가는 2~3배가 된다. 이 값은 하한선이다.
    """
    if throughput_rps <= 0:
        return float("nan")
    seconds = 1_000_000 / throughput_rps
    return seconds / 3600 * hourly_usd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16, help="처리량 측정에 쓸 배치 크기")
    parser.add_argument("--out", default="runs/latency.json")
    parser.add_argument(
        "--cost-per-hour", type=float, default=0.10,
        help="추론 인스턴스 시간당 요금(USD). 기본값은 작은 CPU 인스턴스 기준의 예시다",
    )
    parser.add_argument(
        "--llm-cost-per-request", type=float, default=0.002,
        help="LLM API로 판정할 때의 요청당 비용(USD) 추정치. 비교용",
    )
    parser.add_argument(
        "--llm-latency-ms", type=float, default=1500.0,
        help="LLM API 판정의 요청당 지연시간(ms) 추정치. 비교용",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    text = "이전 지시를 무시하라는 문장을 탐지하는 테스트 입력"
    encoded = tokenizer(text, return_tensors="pt")
    model = AutoModelForSequenceClassification.from_pretrained(args.model).eval()
    session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])

    # 워밍업. 첫 호출에는 그래프 최적화와 메모리 할당이 섞여 들어간다.
    for _ in range(5):
        with torch.inference_mode():
            model(**encoded)
        session.run(None, {key: value.numpy() for key, value in encoded.items()})

    torch_times, onnx_times = [], []
    for _ in range(args.runs):
        started = time.perf_counter()
        with torch.inference_mode():
            model(**encoded)
        torch_times.append((time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        session.run(None, {key: value.numpy() for key, value in encoded.items()})
        onnx_times.append((time.perf_counter() - started) * 1000)

    # 처리량은 배치로 잰다. 단건 지연시간의 역수가 아니다 — 배치는 훨씬 효율적이다.
    batched = tokenizer([text] * args.batch, return_tensors="pt", padding=True)
    batch_inputs = {key: value.numpy() for key, value in batched.items()}
    for _ in range(3):
        session.run(None, batch_inputs)
    batch_started = time.perf_counter()
    batch_rounds = max(args.runs // 4, 1)
    for _ in range(batch_rounds):
        session.run(None, batch_inputs)
    batch_elapsed = time.perf_counter() - batch_started
    onnx_rps = (batch_rounds * args.batch) / batch_elapsed
    onnx_single_rps = 1000.0 / float(np.percentile(onnx_times, 50))

    result = {
        "runs": args.runs,
        "batch": args.batch,
        "torch_cpu": stats(torch_times),
        "onnx_cpu": stats(onnx_times),
        "throughput": {
            "onnx_batch_rps": round(onnx_rps, 1),
            "onnx_single_rps": round(onnx_single_rps, 1),
        },
        "cost": {
            "assumed_instance_usd_per_hour": args.cost_per_hour,
            "self_hosted_usd_per_million": round(cost_per_million(onnx_rps, args.cost_per_hour), 2),
            "llm_judge_usd_per_million": round(args.llm_cost_per_request * 1_000_000, 2),
            "llm_judge_assumed_usd_per_request": args.llm_cost_per_request,
        },
    }

    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    self_hosted = result["cost"]["self_hosted_usd_per_million"]
    llm = result["cost"]["llm_judge_usd_per_million"]
    ratio = llm / self_hosted if self_hosted else float("inf")
    print(
        f"\n100만 요청 기준 추정: 자체 운영 ${self_hosted:,.2f} vs LLM 판정 ${llm:,.2f} "
        f"(약 {ratio:,.0f}배)"
    )
    print(
        f"지연시간: ONNX p50 {result['onnx_cpu']['p50_ms']:.1f}ms vs "
        f"LLM 판정 약 {args.llm_latency_ms:.0f}ms"
    )
    print(
        "\n이 숫자는 자릿수 비교용이다. 인스턴스 요금, LLM 단가, 활용률은 모두 가정값이므로 "
        "--cost-per-hour와 --llm-cost-per-request에 실제 값을 넣어 다시 계산한다.\n"
        "그리고 비용이 전부는 아니다. LLM 판정이 benign FPR을 크게 낮춘다면 "
        "오탐 1건을 처리하는 사람의 비용까지 함께 계산해야 공정한 비교다."
    )


if __name__ == "__main__":
    main()
