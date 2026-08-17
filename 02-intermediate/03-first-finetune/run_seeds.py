#!/usr/bin/env python
"""같은 설정을 seed만 바꿔 여러 번 학습하고, 지표가 얼마나 흔들리는지 잰다.

중급 수료 기준 2번("성능 변화가 개선인지 학습 노이즈인지 판단")을 위한 도구다.
seed 편차를 모르면 v2가 v1보다 0.02 높다는 사실에 아무 의미가 없다.

기존 스크립트를 subprocess로 호출한다. 학습자가 표 한 줄을 직접 재현하고 싶을 때
같은 명령을 그대로 복사해 쓸 수 있어야 하기 때문이다.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
METRICS = ("macro_f1", "attack_recall", "benign_fpr")


def run(command: list[str]) -> None:
    print(f"\n$ {' '.join(command)}", flush=True)
    subprocess.run(command, check=True)


def train_one(seed: int, args, model_dir: Path) -> dict:
    """seed 하나로 학습 -> 예측 -> 평가하고 report.json을 읽어 돌려준다."""
    run([
        sys.executable, str(HERE / "train_seq_cls.py"),
        "--data", args.data, "--out", str(model_dir),
        "--seed", str(seed), "--epochs", str(args.epochs), "--lr", str(args.lr),
        "--device", args.device,
    ])
    pred_path = model_dir / "seed-pred.jsonl"
    run([
        sys.executable, str(HERE / "predict.py"),
        "--model", str(model_dir), "--input", args.gold, "--output", str(pred_path),
    ])
    report_dir = model_dir / "report"
    run([
        sys.executable, str(REPO / "02-intermediate/04-evaluation-harness/evaluate.py"),
        "--gold", args.gold, "--pred", str(pred_path), "--out", str(report_dir),
        "--bootstrap", str(args.bootstrap),
    ])
    return json.loads((report_dir / "report.json").read_text(encoding="utf-8"))


def summarize(values: list[float]) -> dict:
    """seed별 값에서 평균·표준편차·범위를 낸다.

    표본표준편차(n-1)를 쓴다. seed 3개는 모집단이 아니라 표본이다.
    seed가 1개뿐이면 편차를 알 수 없으므로 None으로 남긴다.
    """
    return {
        "mean": round(statistics.fmean(values), 4),
        "stdev": round(statistics.stdev(values), 4) if len(values) > 1 else None,
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "spread": round(max(values) - min(values), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="train/dev/test가 있는 데이터 폴더")
    parser.add_argument("--gold", required=True, help="평가에 쓸 gold jsonl (보통 test.jsonl)")
    parser.add_argument("--out", required=True, help="결과를 모을 폴더")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=float, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument(
        "--keep-models", action="store_true",
        help="모델 가중치를 남긴다. 하나에 약 520MB이므로 기본값은 평가 후 삭제다",
    )
    parser.add_argument(
        "--compare", help="비교할 다른 run의 report.json. 개선폭이 seed 편차보다 큰지 판단한다",
    )
    args = parser.parse_args()

    if len(args.seeds) != len(set(args.seeds)):
        raise SystemExit("--seeds에 중복이 있습니다. 같은 seed는 같은 결과를 냅니다")
    if len(args.seeds) < 2:
        print("주의: seed가 1개면 편차를 잴 수 없다. 최소 3개를 권장한다.", file=sys.stderr)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    per_seed = {}
    for seed in args.seeds:
        model_dir = out / f"seed-{seed}"
        report = train_one(seed, args, model_dir)
        per_seed[seed] = {metric: report[metric] for metric in METRICS}
        per_seed[seed]["n_samples"] = report["n_samples"]
        if not args.keep_models:
            # 가중치만 지우고 report/예측은 남긴다. 표를 다시 만들 수 있어야 한다.
            for item in model_dir.iterdir():
                if item.is_file() and item.suffix in {".safetensors", ".bin"}:
                    item.unlink()

    stats = {metric: summarize([per_seed[seed][metric] for seed in args.seeds]) for metric in METRICS}

    lines = ["## seed 편차", "", f"데이터 `{args.data}` · gold `{args.gold}` · seed {args.seeds}", ""]
    lines.append("| seed | " + " | ".join(METRICS) + " |")
    lines.append("|---|" + "---:|" * len(METRICS))
    for seed in args.seeds:
        lines.append(f"| {seed} | " + " | ".join(f"{per_seed[seed][m]:.3f}" for m in METRICS) + " |")
    lines.append("| **평균** | " + " | ".join(f"**{stats[m]['mean']:.3f}**" for m in METRICS) + " |")
    stdev_cells = [
        "-" if stats[m]["stdev"] is None else f"{stats[m]['stdev']:.3f}" for m in METRICS
    ]
    lines.append("| 표준편차 | " + " | ".join(stdev_cells) + " |")
    lines.append("| 최대-최소 | " + " | ".join(f"{stats[m]['spread']:.3f}" for m in METRICS) + " |")
    lines.extend([
        "",
        "**읽는 법**: 최대-최소가 seed만 바꿔서 생긴 변동 폭이다. 어떤 변경의 개선폭이 "
        "이 값보다 작으면 그것은 개선이 아니라 노이즈일 수 있다.",
    ])

    if args.compare:
        other = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        lines.extend(["", f"### `{args.compare}`와 비교", "", "| 지표 | 비교 대상 | seed 평균 | 차이 | 판정 |", "|---|---:|---:|---:|---|"])
        for metric in METRICS:
            delta = other[metric] - stats[metric]["mean"]
            spread = stats[metric]["spread"]
            verdict = "노이즈 범위 안" if abs(delta) <= spread else "seed 편차보다 큼"
            lines.append(
                f"| {metric} | {other[metric]:.3f} | {stats[metric]['mean']:.3f} | "
                f"{delta:+.3f} | {verdict} |"
            )
        lines.append("")
        lines.append(
            "'seed 편차보다 큼'도 증명은 아니다. seed 3개의 범위는 그 자체로 표본이 작다. "
            "확신이 필요하면 seed를 늘리고 신뢰구간까지 함께 본다."
        )

    markdown = "\n".join(lines)
    (out / "seed-variance.md").write_text(markdown + "\n", encoding="utf-8")
    (out / "seed-variance.json").write_text(
        json.dumps({"per_seed": per_seed, "stats": stats, "seeds": args.seeds}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\n" + markdown)
    print(f"\n-> {out}/seed-variance.md")
    if not args.keep_models:
        print("모델 가중치는 삭제했다. 남기려면 --keep-models를 쓴다.")
    total_mb = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"남은 결과 용량: {total_mb:.1f}MB")
    shutil.rmtree(out / "__pycache__", ignore_errors=True)


if __name__ == "__main__":
    main()
