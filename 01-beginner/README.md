# 초급 · 먼저 정의하고, 규칙으로 잡고, 제대로 잰다

초급에서는 **모델을 학습하지 않는다.** 대신 이후 모든 단계가 딛고 설 바닥을 만든다. 무엇을 탐지할지
정의하고, 가장 단순한 탐지기로 기준선을 만들고, 그 결과를 정확하게 재는 법을 익힌다.

ML 경험이 없어도 따라올 수 있다. 여기서 나오는 코드는 정규식과 산수뿐이다.

## 레슨

| # | 레슨 | 하는 일 | 산출물 |
|---|---|---|---|
| 01 | [위협 모델과 라벨 경계](01-threat-model/LESSON.md) | 무엇을 탐지할지 정의하고 어노테이션 가이드를 쓴다 | `GUIDE.md`, 직접 만든 샘플 10건 |
| 02 | [규칙 기반선](02-rule-baseline/LESSON.md) | 정규식 탐지기를 돌리고 어디서 틀리는지 본다 | `runs/rule-pred.jsonl` |
| 03 | [평가 기초](03-evaluation-basics/LESSON.md) | 지표를 손으로 계산하고 slice로 약점을 찾는다 | `runs/rule-report.json` |
| 04 | [제로샷 모델](04-run-pretrained/LESSON.md) | 학습 없이 범용 모델을 써보고 규칙과 비교한다 | 규칙 vs 모델 비교표 |

## 초급을 마치면 손에 남는 것

세 가지 기준선의 숫자다. 중급에서 만들 모델은 이 숫자들을 넘어야 한다.

| 방법 | macro F1 | attack recall | benign FPR |
|---|---:|---:|---:|
| 전부 BENIGN 예측 | 0.167 | 0.000 | 0.000 |
| 규칙 기반선 | 0.672 | 0.562 | 0.125 |
| 제로샷 mDeBERTa | 0.465 | 0.500 | 0.500 |

*`common/data/bench/gold.jsonl` 24건 기준. 표본이 작아 신뢰구간이 넓다.*

## 수료 기준

높은 점수가 아니다. 다음 네 가지를 **설명할 수 있으면** 통과다.

1. 어휘가 거의 같은데 라벨이 반대인 문장(하드 네거티브)이 왜 따로 필요한가
2. `benign FPR`이 0인데도 그 탐지기를 배포하면 안 되는 경우
3. 전체 macro F1이 0.672인데 특정 slice가 0.000일 수 있는 이유
4. 1GB짜리 다국어 모델이 정규식 몇 줄에 진 이유

## 실행 순서 요약

```bash
source .venv/bin/activate

# 02 규칙 기반선
python 01-beginner/02-rule-baseline/rule_detector.py \
  --input common/data/bench/gold.jsonl --output runs/rule-pred.jsonl

# 03 평가
python 01-beginner/03-evaluation-basics/evaluate.py \
  --gold common/data/bench/gold.jsonl --pred runs/rule-pred.jsonl \
  --json-out runs/rule-report.json

# 02 하드 네거티브 전용 확인 (여기서 읽을 값은 benign FPR 하나뿐)
python 01-beginner/02-rule-baseline/rule_detector.py \
  --input common/data/bench/negatives.jsonl --output runs/rule-neg-pred.jsonl
python 01-beginner/03-evaluation-basics/evaluate.py \
  --gold common/data/bench/negatives.jsonl --pred runs/rule-neg-pred.jsonl

# 04 제로샷 (첫 실행 시 모델 약 1GB 다운로드)
python 01-beginner/04-run-pretrained/run_zero_shot.py \
  --input common/data/bench/gold.jsonl --output runs/zero-shot-pred.jsonl
python 01-beginner/03-evaluation-basics/evaluate.py \
  --gold common/data/bench/gold.jsonl --pred runs/zero-shot-pred.jsonl
```
