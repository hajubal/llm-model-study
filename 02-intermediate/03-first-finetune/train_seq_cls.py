#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments

from guardlab.io import read_jsonl
from guardlab.schema import ID2LABEL, LABEL2ID, LABELS


def pick_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("MPS를 요청했지만 사용할 수 없습니다")
        return "mps"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA를 요청했지만 사용할 수 없습니다")
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def make_dataset(rows, tokenizer, max_len: int) -> Dataset:
    dataset = Dataset.from_dict({
        "text": [row.text for row in rows],
        "labels": [LABEL2ID[row.label] for row in rows],
    })
    return dataset.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_len),
        batched=True,
        remove_columns=["text"],
    )


def compute_metrics(eval_prediction) -> dict:
    logits, labels = eval_prediction
    preds = np.argmax(logits, axis=-1)
    f1s = []
    for label_id in range(len(LABELS)):
        tp = int(np.sum((preds == label_id) & (labels == label_id)))
        fp = int(np.sum((preds == label_id) & (labels != label_id)))
        fn = int(np.sum((preds != label_id) & (labels == label_id)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {"accuracy": float(np.mean(preds == labels)), "macro_f1": float(np.mean(f1s))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="distilbert-base-multilingual-cased")
    parser.add_argument("--epochs", type=float, default=4)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max-len", type=int, default=192)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_rows = read_jsonl(data_dir / "train.jsonl")
    dev_rows = read_jsonl(data_dir / "dev.jsonl")
    train_data = make_dataset(train_rows, tokenizer, args.max_len)
    dev_data = make_dataset(dev_rows, tokenizer, args.max_len)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID,
    )
    training_args = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch * 2,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.06,
        eval_strategy="epoch" if args.max_steps < 0 else "no",
        save_strategy="epoch" if args.max_steps < 0 else "no",
        load_best_model_at_end=args.max_steps < 0,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=10,
        report_to="none",
        seed=args.seed,
        use_cpu=device == "cpu",
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=dev_data,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )
    started = time.time()
    trainer.train()
    elapsed = time.time() - started
    metrics = trainer.evaluate() if args.max_steps < 0 else {}
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    trainer.state.save_to_json(str(out / "trainer_state.json"))
    shutil.rmtree(out / "checkpoints", ignore_errors=True)
    summary = {
        "base_model": args.model,
        "data_dir": str(data_dir),
        "device": device,
        "seed": args.seed,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "batch": args.batch,
        "learning_rate": args.lr,
        "max_len": args.max_len,
        "n_train": len(train_rows),
        "n_dev": len(dev_rows),
        "train_seconds": round(elapsed, 1),
        "dev_metrics": {key: round(float(value), 5) for key, value in metrics.items() if isinstance(value, (int, float))},
    }
    (out / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved model -> {out} ({elapsed:.1f}s, device={device})")


if __name__ == "__main__":
    main()
