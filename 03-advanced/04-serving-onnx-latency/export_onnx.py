#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class LogitsOnly(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).eval()
    example = tokenizer("환경 확인용 정상 문장", return_tensors="pt")
    torch.onnx.export(
        LogitsOnly(model), (example["input_ids"], example["attention_mask"]), str(target),
        input_names=["input_ids", "attention_mask"], output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "sequence"}, "attention_mask": {0: "batch", 1: "sequence"}, "logits": {0: "batch"}},
        opset_version=args.opset,
    )
    # 학습된 모델 폴더를 그대로 덮어쓰지 않도록 서빙용 토크나이저는 ONNX 옆 폴더에 따로 둔다.
    tokenizer_dir = target.parent / f"{target.stem}-tokenizer"
    tokenizer.save_pretrained(tokenizer_dir)
    print(f"ONNX -> {target}")
    print(f"tokenizer -> {tokenizer_dir}")


if __name__ == "__main__":
    main()

