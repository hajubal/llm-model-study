# 고급 · 간접 입력과 우회를 견디는 운영 방어층 만들기

고급은 **분류 모델을 제품 방어층으로 가져가는 과정**이다. 모델 하나로 안전을 보증하지 않고, 입력 출처와
긴 문서, 표현 변형, 정책 임계값, 지연시간, 회귀 게이트를 함께 다룬다.

중급까지는 "모델을 얼마나 잘 만드는가"였다면, 고급은 **"이 모델이 실패할 때 무슨 일이 일어나는가"**다.

## 레슨

| # | 레슨 | 하는 일 | 산출물 |
|---|---|---|---|
| 01 | [간접 인젝션과 긴 문서](01-indirect-injection/LESSON.md) | truncation 미탐과 sliding window 트레이드오프 | 위치별·길이별 평가 |
| 02 | [적대적 변형](02-adversarial-augmentation/LESSON.md) | 표면 변형만으로 성능이 무너지는 것을 실측 | robustness 비교표 |
| 03 | [하이브리드 게이트](03-hybrid-gates/LESSON.md) | 규칙+모델을 allow/review/block으로 결합 | `POLICY.md`, CI 게이트 |
| 04 | [ONNX와 지연시간](04-serving-onnx-latency/LESSON.md) | 동등성 검증 후 p50/p95 측정 | `latency-report.md` |
| 05 | [벤치마크와 리포트](05-benchmark-and-report/LESSON.md) | 세 축 보고, 신뢰구간, 모델 카드 | `model-report.md` |
| 06 | [캡스톤](06-capstone/README.md) | 데이터→모델→정책→리포트 완주 | 10개 산출물 |

## 고급에서 확인하게 되는 실측

이 단계의 핵심은 **중급에서 만든 모델이 실제로는 얼마나 약한지** 드러내는 것이다.

| 상황 | 결과 |
|---|---|
| 긴 문서, 단일 truncation | 공격 3건 **전부 미탐** (attack recall 0.000) |
| 긴 문서, sliding window | 3건 다 탐지하지만 **정상 문서도 오탐** |
| 대소문자 변형 | attack recall 0.914 → **0.609** (33%p 하락) |
| 공백 변형 (규칙 기반선) | attack recall 0.562 → **0.438** |
| `source=tool` slice | macro F1 0.578, benign FPR **0.625** |
| 품질 게이트 (recall≥0.90, FPR≤0.05) | **FAILED** — benign FPR 0.250 |

**"macro F1 0.750짜리 괜찮은 모델"이 실제로는 이 상태다.** 벤치마크 숫자 하나로 배포 결정을 내리면
안 되는 이유가 여기 있다.

## 관통하는 원칙

### 1. 판정 불가를 안전으로 바꾸지 않는다 (fail-closed)

```text
입력이 너무 길다        → 잘라서 판정 ❌  →  거부하거나 chunk 처리 ✅
모델 출력이 이상하다    → 통과 ❌        →  차단 ✅
추론 백엔드가 죽었다    → 통과 ❌        →  에러 반환, 상위에서 차단 ✅
```

### 2. 탐지와 정책은 다른 층이다

탐지 엔진은 판정만 하고, 무엇을 차단할지는 정책 층이 정한다. 그래야 모델 재배포 없이 임계값을 바꾸고,
같은 엔진을 여러 제품이 다른 정책으로 쓸 수 있다.

### 3. 모델이 실패해도 권한이 보호되어야 한다

탐지는 방어층 중 하나다. 권한 분리, 도구 allowlist, 컨텍스트 경계, 출력 검증이 함께 있어야 한다.
**분류 결과를 권한 상승의 근거로 사용하지 않는다.**

### 4. 숫자에는 항상 조건이 붙는다

`n`, 데이터 버전, seed, threshold, 하드웨어, 측정 조건이 없는 숫자는 재현할 수도 비교할 수도 없다.

## 실행 순서 요약

```bash
source .venv/bin/activate

# 01 간접 인젝션
python 03-advanced/01-indirect-injection/make_indirect_eval.py
python 02-intermediate/03-first-finetune/predict.py \
  --model runs/models/mbert-v1 --input runs/data/indirect-eval.jsonl \
  --output runs/models/mbert-v1/indirect-single.jsonl
python 03-advanced/01-indirect-injection/chunk_predict.py \
  --model runs/models/mbert-v1 --input runs/data/indirect-eval.jsonl \
  --output runs/models/mbert-v1/indirect-chunk.jsonl

# 02 적대적 변형
python 03-advanced/02-adversarial-augmentation/perturb.py \
  --input runs/data/v1/test.jsonl --output runs/data/test-perturbed.jsonl
python 02-intermediate/03-first-finetune/predict.py \
  --model runs/models/mbert-v1 --input runs/data/test-perturbed.jsonl \
  --output runs/models/mbert-v1/perturbed-pred.jsonl

# 03 하이브리드 게이트
python 03-advanced/03-hybrid-gates/apply_policy.py \
  --input common/data/bench/gold.jsonl \
  --pred runs/models/mbert-v1/bench-pred.jsonl \
  --output runs/models/mbert-v1/bench-decisions.jsonl
python 03-advanced/03-hybrid-gates/quality_gate.py \
  runs/models/mbert-v1/bench-report/report.json \
  --min-attack-recall 0.90 --max-benign-fpr 0.05

# 04 ONNX와 지연시간
python 03-advanced/04-serving-onnx-latency/export_onnx.py \
  --model runs/models/mbert-v1 --out runs/models/mbert-v1/model.onnx
python 03-advanced/04-serving-onnx-latency/bench_latency.py \
  --model runs/models/mbert-v1 --onnx runs/models/mbert-v1/model.onnx \
  --runs 30 --out runs/latency.json

# 05 리포트
python 03-advanced/05-benchmark-and-report/make_report.py \
  --reports runs/models/mbert-v1/test-report/report.json \
            runs/models/mbert-v1/bench-report/report.json \
  --latency runs/latency.json \
  --out runs/report/model-report.md
```

## 운영 적용 전 체크리스트

[`APPENDIX-production-checklist.md`](APPENDIX-production-checklist.md)를 참고한다. 이 커리큘럼을 끝내고
실제 시스템에 적용할 때 확인할 항목들이다.
