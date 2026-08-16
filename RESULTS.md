# 실험 결과

이 파일은 결과의 인덱스다. 원본 JSON, 예측, 설정은 `runs/<run-name>/`에 둔다.

| 날짜 | run | data | model/policy | macro F1 | attack recall | benign FPR | 비고 |
|---|---|---|---|---:|---:|---:|---|
| 2026-08-16 | rule-smoke | `common/data/bench/gold.jsonl` | `guardlab.rules` | 0.672 | 0.562 | 0.125 | 합성 24건, 파이프라인 확인용 |

숫자에는 반드시 seed, 데이터 버전, 모델 체크포인트, threshold를 함께 남긴다.
위 smoke 결과는 제품 성능 주장이 아니며, `01-beginner/02-rule-baseline`과 공용 채점기의 연결을 확인한 실측이다.
