# 중급 · 데이터를 만들고, 첫 모델을 학습하고, 오류로 개선한다

중급의 중심은 **학습 명령이 아니라 데이터 계약과 개선 루프**다. `train.py`를 실행하는 것은 몇 초면
배우지만, 왜 그 데이터로 학습하면 안 되는지 판단하는 데는 이 여섯 레슨이 필요하다.

## 레슨

| # | 레슨 | 하는 일 | 산출물 |
|---|---|---|---|
| 01 | [데이터 스키마와 group split](01-dataset-schema/LESSON.md) | 누수를 만들어 보고 검사기로 잡는다 | 누수 검사 결과, 분포표 |
| 02 | [합성 데이터](02-synthetic-data/LESSON.md) | 템플릿 생성기를 읽고 표현을 늘린다 | `runs/data/v1/{train,dev,test}.jsonl` |
| 03 | [첫 파인튜닝](03-first-finetune/LESSON.md) | 모델을 학습하고 과적합 지점을 찾는다 | `runs/models/mbert-v1/`, 모델 카드 |
| 04 | [평가 하네스](04-evaluation-harness/LESSON.md) | 결과를 파일로 남기고 slice로 약점을 찾는다 | `report.json`, `report.md`, `errors.jsonl` |
| 05 | [에러 분석과 threshold](05-error-analysis/LESSON.md) | dev에서 임계값을 정하고 test에 한 번 쓴다 | 오류 유형표, threshold 근거 |
| 06 | [미니 프로젝트](06-mini-project/README.md) | 한 변수만 바꿔 v2를 만들고 비교한다 | 비교표, `RETRO.md` |

## 기본 설정으로 나오는 실측값

이 값들을 기준으로 삼는다. 여러분의 결과가 이와 다르면 무엇이 달랐는지 먼저 확인한다.

| 항목 | 값 |
|---|---|
| 데이터 | 템플릿 90개 · train 384 / dev 144 / test 192 · 문장 중복 0% |
| 학습 | distilbert-base-multilingual-cased · 8 epoch · lr 5e-5 · MPS 약 95초 |
| 합성 test | macro F1 **0.750** · attack recall **0.914** · benign FPR **0.328** |
| 독립 bench(24건) | macro F1 **0.837** · attack recall **0.938** · benign FPR **0.250** |
| source slice | user **0.887** / retrieved **0.607** / tool **0.578** |

### 이 숫자를 어떻게 읽어야 하는가

- **초급 기준선(규칙 0.672)을 넘었다.** 모델이 규칙보다 낫다는 것은 확인됐다
- **benign FPR 0.328은 운영에 못 쓴다.** 정상의 3분의 1을 차단한다
- **`tool` slice가 0.578로 최악이다.** 도구 출력 형태의 정상 문장을 공격으로 오해한다
- 즉 이 모델은 **완성품이 아니라 개선 루프의 출발점**이다

`macro F1 ≥ 0.80`, `benign FPR ≤ 5%`는 **개선 루프의 목표치**이지 기본 설정의 결과가 아니다.
거기에 도달하려면 하드 네거티브 템플릿을 늘리고(02 과제), threshold를 조정해야 한다(05 과제).

## 중급 수료 기준

숫자가 아니라 다음을 할 수 있으면 통과다.

1. **누수를 두 종류 이상 설명**할 수 있다 (group 누수 / 텍스트 복제 / near-duplicate)
2. 성능 변화가 **개선인지 학습 노이즈인지** 판단하는 절차를 안다 (seed 편차 측정)
3. dev에서 threshold를 정하고 **test를 한 번만** 여는 규율을 지켰다
4. 오류를 유형별로 나누고 **처방을 하나만** 골랐다
5. 개선 결과를 **악화된 slice까지 포함해** 기록했다

## 실행 순서 요약

```bash
source .venv/bin/activate

# 02 → 01: 데이터 생성 후 검사
python 02-intermediate/02-synthetic-data/gen_synth.py --out runs/data/v1
python 02-intermediate/01-dataset-schema/inspect_data.py runs/data/v1

# 03: 학습과 예측 (MPS 기준 약 95초)
python 02-intermediate/03-first-finetune/train_seq_cls.py \
  --data runs/data/v1 --out runs/models/mbert-v1
python 02-intermediate/03-first-finetune/predict.py \
  --model runs/models/mbert-v1 --input runs/data/v1/test.jsonl \
  --output runs/models/mbert-v1/test-pred.jsonl

# 04: 평가 (합성 test + 독립 bench)
python 02-intermediate/04-evaluation-harness/evaluate.py \
  --gold runs/data/v1/test.jsonl --pred runs/models/mbert-v1/test-pred.jsonl \
  --out runs/models/mbert-v1/test-report
python 02-intermediate/03-first-finetune/predict.py \
  --model runs/models/mbert-v1 --input common/data/bench/gold.jsonl \
  --output runs/models/mbert-v1/bench-pred.jsonl
python 02-intermediate/04-evaluation-harness/evaluate.py \
  --gold common/data/bench/gold.jsonl --pred runs/models/mbert-v1/bench-pred.jsonl \
  --out runs/models/mbert-v1/bench-report

# 05: dev에서 오류 분석과 threshold 결정
python 02-intermediate/03-first-finetune/predict.py \
  --model runs/models/mbert-v1 --input runs/data/v1/dev.jsonl \
  --output runs/models/mbert-v1/dev-pred.jsonl
python 02-intermediate/05-error-analysis/error_dump.py \
  --gold runs/data/v1/dev.jsonl --pred runs/models/mbert-v1/dev-pred.jsonl
python 02-intermediate/05-error-analysis/threshold_sweep.py \
  --gold runs/data/v1/dev.jsonl --pred runs/models/mbert-v1/dev-pred.jsonl
```

## 디스크 관리

모델 하나가 약 520MB다. 실험을 여러 번 하면 금방 수 GB가 된다.

```bash
du -sh runs/models/*          # 확인
rm -rf runs/models/실험이름    # 정리
```

`train_summary.json`과 `report.json`만 남겨 두면 나중에 같은 명령으로 재현할 수 있다.
