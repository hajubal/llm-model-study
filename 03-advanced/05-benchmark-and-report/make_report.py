#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--latency")
    parser.add_argument("--out", default="runs/report/model-report.md")
    args = parser.parse_args()
    lines = ["# Jailbreak / Prompt Injection 탐지 모델 보고서", "", f"작성일: {date.today().isoformat()}", ""]
    lines.extend(["## 성능 요약", "", "| run | n | macro F1 | attack recall | benign FPR |", "|---|---:|---:|---:|---:|"])
    for report_path in args.reports:
        path = Path(report_path)
        report = json.loads(path.read_text(encoding="utf-8"))
        lines.append(
            f"| {path.parent} | {report['n_samples']} | {report['macro_f1']:.3f} | "
            f"{report['attack_recall']:.3f} | {report['benign_fpr']:.3f} |"
        )
    if args.latency:
        latency = json.loads(Path(args.latency).read_text(encoding="utf-8"))
        lines.extend(["", "## 지연시간", "", "```json", json.dumps(latency, ensure_ascii=False, indent=2), "```"])
    lines.extend([
        "", "## 평가 방법", "",
        "- 데이터 버전, 생성/수집 경로, group split, seed, threshold를 여기에 기록한다.",
        "- 전체 점수 외에 language/source/길이/변형 slice와 bootstrap 신뢰구간을 추가한다.",
        "- test는 threshold 결정에 사용하지 않고 최종 확인에 한 번 사용한다.",
        "", "## 알려진 한계", "",
        "- 탐지 통과는 안전 보증이 아니며 새 공격 표현과 분포 변화에 실패할 수 있다.",
        "- 합성 데이터와 작은 공개 벤치마크 점수는 실제 운영 성능을 대표하지 않는다.",
        "- 권한 분리, 도구 allowlist, 컨텍스트 경계, 출력 검증과 함께 사용한다.",
        "", "## 배포 결정", "",
        "담당자, 승인된 threshold, rollback 조건, 모니터링 지표를 기록한다.",
    ])
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report -> {target}")


if __name__ == "__main__":
    main()

