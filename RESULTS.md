# 실험 결과

이 파일은 결과의 인덱스다. 원본 JSON, 예측, 설정은 `runs/<run-name>/`에 둔다.

**숫자에는 반드시 seed, 데이터 버전, 모델 체크포인트, threshold, 표본 수를 함께 남긴다.**
아래 결과는 전부 제품 성능 주장이 아니라 파이프라인 실측이다.

## 기준선 (학습 없음) — `common/data/bench/gold.jsonl` 24건

| 날짜 | run | 방법 | macro F1 | attack recall | benign FPR | 비고 |
|---|---|---|---:|---:|---:|---|
| 2026-08-16 | all-benign | 전부 BENIGN | 0.167 | 0.000 | 0.000 | 하한선. FPR 만점이 무의미하다는 증거 |
| 2026-08-16 | all-attack | 전부 PROMPT_INJECTION | 0.167 | 1.000 | 1.000 | 반대 극단. recall 만점이 무의미하다는 증거 |
| 2026-08-16 | rule-smoke | `guardlab.rules` | 0.672 [0.445–0.841] | 0.562 | 0.125 | 정규식 7개 |
| 2026-08-16 | zeroshot-smoke | mDeBERTa-v3 XNLI 제로샷 | 0.465 | 0.500 | 0.500 | 학습 없이 라벨 설명만 사용 |
| 2026-08-17 | guard-model | `protectai/deberta-v3-base-prompt-injection-v2` | 0.333 † | **0.938** | **0.625** | 전용 가드 모델. recall 최고, FPR 최악 |
| 2026-08-17 | guard-model / negatives | 같은 모델, `negatives.jsonl` 8건 | — | 해당 없음 | **0.750** | 하드 네거티브 6/8건을 **확신도 1.000**으로 오탐 |

† 2-class 모델이라 JAILBREAK를 구조적으로 못 맞힌다. **이 macro F1은 다른 행과 비교하면 안 된다.**
attack recall과 benign FPR만 비교 가능하다.

## 파인튜닝 모델 — `runs/data/v1` (템플릿 108개, n_per_group 8)

모델: `distilbert-base-multilingual-cased`, 8 epoch, lr 5e-5, max_len 192, MPS 약 175초.

### seed 편차 (합성 test 240건) — **이 표를 먼저 본다**

| seed | macro F1 | attack recall | benign FPR |
|---:|---:|---:|---:|
| 42 | 0.625 | 1.000 | 0.650 |
| 43 | 0.733 | 0.969 | 0.450 |
| 44 | 0.666 | 0.744 | 0.362 |
| **평균** | **0.674** | **0.904** | **0.487** |
| 표준편차 | 0.054 | 0.140 | 0.147 |
| **최대−최소** | **0.108** | **0.256** | **0.287** |

재현: `python 02-intermediate/03-first-finetune/run_seeds.py --data runs/data/v1 --gold runs/data/v1/test.jsonl --out runs/seeds/v1 --seeds 42 43 44`

**이 표가 이후 모든 비교의 기준선이다.** macro F1이 0.11, benign FPR이 0.29 폭으로 흔들리므로,
**단일 학습 결과를 소수점 셋째 자리까지 인용하는 것은 근거가 없다.** 어떤 변경의 개선폭이
이 폭보다 작으면 개선이라고 부를 수 없다.

### 단일 실행 상세 (seed 42, 별도 학습)

| 날짜 | run | 데이터 | n | macro F1 | attack recall | benign FPR |
|---|---|---|---:|---:|---:|---:|
| 2026-08-17 | mbert-v1 / test | 합성 test | 240 | 0.637 [0.574–0.697] | 1.000 | 0.487 [0.373–0.588] |
| 2026-08-17 | mbert-v1 / bench | 독립 bench | 24 | 0.717 [0.500–0.879] | 0.938 | 0.375 |

**규칙 기반선(0.672 [0.445–0.841])과 비교할 때는 같은 데이터에서 본다.** 독립 bench에서
모델 0.717 vs 규칙 0.672인데 **두 신뢰구간이 거의 완전히 겹친다.** 이 벤치로는
"모델이 규칙을 넘었다"고 단정할 수 없다.

### source slice (합성 test 240건, seed 42)

| source | n | macro F1 | benign FPR | train 템플릿 수(조합당) |
|---|---:|---:|---:|---:|
| `user` | 96 | 0.750 | 0.250 | 6 |
| `tool` | 48 | 0.958 | 0.000 | 1 |
| `retrieved` | 48 | **0.280** | **1.000** | 1 |
| `system` | 48 | **0.208** | **0.938** | 1 |
| `ko` | 120 | 0.747 | 0.400 | — |
| `en` | 120 | 0.507 | 0.575 | — |

`retrieved`와 `system`이 무너지는 구조적 원인: `(label, language, source)` 조합 24개 중
**18개가 train에 템플릿 1개(문장 8건)뿐**이다. `user`만 조합당 6개다.
자세한 내용은 `02-intermediate/02-synthetic-data/LESSON.md`.

## 적대적 변형 (합성 test 240건 기준, NFKC 적용 후)

| 변형 | 규칙 attack recall | 규칙 benign FPR | 모델 attack recall | 모델 benign FPR |
|---|---:|---:|---:|---:|
| 원본 | 0.562 | 0.125 | 1.000 | 0.487 |
| spaces | 0.438 | 0.125 | 1.000 | 0.487 |
| punctuation | 0.562 | 0.125 | 1.000 | 0.487 |
| case | 0.500 | 0.000 | 0.966 | 0.600 |
| **zero_width** | **0.000** | 0.000 | 1.000 | 0.487 |
| **homoglyph** | **0.143** | 0.000 | 0.955 | **1.000** |
| **base64** | **0.000** | 0.000 | 0.994 | **1.000** |

*규칙 수치는 24건 bench 기준, 모델 수치는 240건 합성 test 기준이다. 절대값이 아니라
**변형 전후의 변화**를 본다.*

**실패 양상이 정반대다.** 규칙은 제로폭 문자와 base64에서 공격을 하나도 못 잡고(recall 0.000),
모델은 같은 변형에서 정상을 전부 차단한다(FPR 1.000). 이것이 `03-advanced/03-hybrid-gates`에서
둘을 겹쳐 쓰는 근거다.

## 간접 인젝션 — 긴 문서 (n=4, 표본이 매우 작다)

| 방식 | attack recall | benign FPR |
|---|---:|---:|
| 단일 truncation | 0.333 | 0.000 |
| sliding window | **1.000** | **1.000** |

chunk 처리는 공격을 전부 잡지만 정상 문서도 전부 오탐한다. window마다 독립 판정하고
하나라도 공격이면 문서를 공격으로 보기 때문에, **문서가 길수록 오탐 기회가 늘어난다.**

## 지연시간과 비용 (Apple M-series, CPU 추론, 단건 입력)

| 항목 | p50 | p95 | p99 |
|---|---:|---:|---:|
| PyTorch CPU | — | 70.2ms | 124.9ms |
| **ONNX CPU** | **7.0ms** | 13.4ms | 17.4ms |

처리량: ONNX 배치(16) **294.8 req/s**, 단건 143.8 req/s.

| 방식 | 요청당 지연시간 | 100만 요청 비용(추정) |
|---|---:|---:|
| 자체 운영 (ONNX/CPU, $0.10/시간 가정) | 7ms | **$0.09** |
| LLM API 판정 ($0.002/요청 가정) | 약 1,500ms | **$2,000** |

**약 22,000배 비용 차이, 약 214배 지연시간 차이**다. 인스턴스 요금·LLM 단가·활용률은 모두
가정값이므로 `--cost-per-hour`와 `--llm-cost-per-request`에 실제 값을 넣어 다시 계산한다.

다만 **비용이 전부는 아니다.** LLM 판정이 benign FPR을 크게 낮춘다면 오탐 1건을 처리하는
사람의 비용까지 함께 계산해야 공정한 비교다. 우리 모델의 FPR이 0.487이라는 점을 기억한다.

재현: `python 03-advanced/04-serving-onnx-latency/bench_latency.py --model runs/models/mbert-v1 --onnx runs/models/mbert-v1/model.onnx --runs 60`

## 알려진 한계

- 합성 데이터이며 실제 사용자 표현 분포를 대표하지 않는다.
- 고정 벤치는 24건이다. 신뢰구간 폭이 0.4에 가까워 대부분의 비교가 통계적으로 무의미하다.
- seed 편차가 macro F1 0.108이다. 단일 실행 숫자는 참고값이다.
- 같은 seed로 다시 학습해도 결과가 정확히 재현되지 않는다(MPS 커널 비결정성).
- 이 수치들은 **개선 루프의 출발점**이지 도달 목표가 아니다.
