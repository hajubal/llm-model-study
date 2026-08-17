# 실험 결과

이 파일은 결과의 인덱스다. 원본 JSON, 예측, 설정은 `runs/<run-name>/`에 둔다.

| 날짜 | run | data | model/policy | macro F1 | attack recall | benign FPR | 비고 |
|---|---|---|---|---:|---:|---:|---|
| 2026-08-16 | rule-smoke | `common/data/bench/gold.jsonl` | `guardlab.rules` | 0.672 | 0.562 | 0.125 | 합성 24건, 파이프라인 확인용 |
| 2026-08-16 | zeroshot-smoke | `common/data/bench/gold.jsonl` | mDeBERTa-v3 XNLI 제로샷 | 0.465 | 0.500 | 0.500 | 학습 없이 라벨 설명만 사용. 규칙 기반선보다 낮다 |
| 2026-08-16 | mbert-v1 / test | `runs/data/v1` (seed 42, n_per_group 8) | distilbert-multilingual, 8ep/5e-5 | 0.750 | 0.914 | 0.328 | 합성 test 192건. source slice: user 0.887 / retrieved 0.607 / tool 0.578 |
| 2026-08-16 | mbert-v1 / bench | `common/data/bench/gold.jsonl` | 같은 모델 | 0.837 | 0.938 | 0.250 | 독립 벤치 24건. 표본이 작아 신뢰구간이 넓다 |

숫자에는 반드시 seed, 데이터 버전, 모델 체크포인트, threshold를 함께 남긴다.
위 smoke 결과는 제품 성능 주장이 아니며, `01-beginner/02-rule-baseline`과 공용 채점기의 연결을 확인한 실측이다.
