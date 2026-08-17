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
| 데이터 | 템플릿 108개 · train 432 / dev 192 / test 240 · 문장 중복 0% |
| 학습 | distilbert-base-multilingual-cased · 8 epoch · lr 5e-5 · MPS 약 175초 |

**단일 실행 숫자를 먼저 보지 않는다.** seed 3개(42/43/44)로 잰 편차가 먼저다.

| 합성 test 지표 | seed 42 | seed 43 | seed 44 | 평균 | **최대−최소** |
|---|---:|---:|---:|---:|---:|
| macro F1 | 0.625 | 0.733 | 0.666 | 0.674 | **0.108** |
| attack recall | 1.000 | 0.969 | 0.744 | 0.904 | **0.256** |
| benign FPR | 0.650 | 0.450 | 0.362 | 0.487 | **0.287** |

독립 bench(24건, seed 42): macro F1 **0.717** [0.500–0.879] · attack recall 0.938 · benign FPR 0.375

source slice (합성 test, seed 42): user **0.750** / tool **0.958** / retrieved **0.280** / system **0.208**

### 이 숫자를 어떻게 읽어야 하는가

- **가장 중요한 열은 "최대−최소"다.** seed만 바꿔도 macro F1이 0.11, benign FPR이 0.29 폭으로
  흔들린다. **단일 실행 결과를 소수점 셋째 자리까지 인용하는 것은 근거가 없다.**
- **"규칙 기반선을 넘었다"고 단정할 수 없다.** 같은 데이터(bench 24건)에서 모델 0.717
  [0.500–0.879] vs 규칙 0.672 [0.445–0.841] — **두 구간이 거의 완전히 겹친다.**
  다른 데이터의 숫자끼리 비교하는 것은 더 나쁘다.
- **benign FPR 0.487은 운영에 못 쓴다.** 정상의 절반을 차단한다.
- **`retrieved`(0.280)와 `system`(0.208)이 최악이다.** 원인은 데이터에 있다 —
  `(label, language, source)` 조합 24개 중 **18개가 train에 템플릿 1개(문장 8건)뿐**이고,
  `user`만 조합당 6개다. 자세한 내용은 [02 레슨](02-synthetic-data/LESSON.md).
- 즉 이 모델은 **완성품이 아니라 개선 루프의 출발점**이다.

`macro F1 ≥ 0.80`, `benign FPR ≤ 5%`는 **개선 루프의 목표치**이지 기본 설정의 결과가 아니다.
거기에 도달하려면 약한 slice의 템플릿을 늘리고(02 과제 2-B), threshold를 조정해야 한다(05 과제).

예시 답안: [SOLUTIONS.md](SOLUTIONS.md) — 먼저 스스로 답한 뒤에 열 것.

## 중급 수료 기준

숫자가 아니라 다음을 할 수 있으면 통과다.

1. **누수를 두 종류 이상 설명**할 수 있다 (group 누수 / 텍스트 복제 / near-duplicate)
2. 성능 변화가 **개선인지 학습 노이즈인지** 판단하는 절차를 안다.
   **두 가지 불확실성을 구분**할 수 있다 — seed 편차(학습 불안정)와 신뢰구간(표본 부족)
3. dev에서 threshold를 정하고 **test를 한 번만** 여는 규율을 지켰다
4. 오류를 유형별로 나누고 **처방을 하나만** 골랐다
5. 개선 결과를 **악화된 slice까지 포함해** 기록했다

## 실행 순서 요약

```bash
source .venv/bin/activate

# 02 → 01: 데이터 생성 후 검사
python 02-intermediate/02-synthetic-data/gen_synth.py --out runs/data/v1
python 02-intermediate/01-dataset-schema/inspect_data.py runs/data/v1

# 03: 학습과 예측 (MPS 기준 약 175초)
python 02-intermediate/03-first-finetune/train_seq_cls.py \
  --data runs/data/v1 --out runs/models/mbert-v1

# 03: seed 편차 먼저 재기 — 이후 모든 비교의 기준선 (학습 3회, 약 10분)
python 02-intermediate/03-first-finetune/run_seeds.py \
  --data runs/data/v1 --gold runs/data/v1/test.jsonl \
  --out runs/seeds/v1 --seeds 42 43 44
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

# 05: 확률 보정 확인 — threshold 0.8이 "80% 확률"이 아님을 실측한다
python 02-intermediate/05-error-analysis/calibration.py \
  --gold runs/data/v1/dev.jsonl --pred runs/models/mbert-v1/dev-pred.jsonl
```

## 디스크 관리

모델 하나가 약 520MB다. 실험을 여러 번 하면 금방 수 GB가 된다.

```bash
du -sh runs/models/*          # 확인
rm -rf runs/models/실험이름    # 정리
```

`train_summary.json`과 `report.json`만 남겨 두면 나중에 같은 명령으로 재현할 수 있다.
