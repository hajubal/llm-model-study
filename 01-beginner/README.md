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
| 04 | [제로샷·기성 가드 모델](04-run-pretrained/LESSON.md) | 학습 없는 모델들과 비교한다 | 기준선 비교표 |

예시 답안: [SOLUTIONS.md](SOLUTIONS.md) — 먼저 스스로 답한 뒤에 열 것.

## 초급을 마치면 손에 남는 것

세 가지 기준선의 숫자다. 중급에서 만들 모델은 이 숫자들을 넘어야 한다.

| 방법 | macro F1 | attack recall | benign FPR |
|---|---:|---:|---:|
| 전부 BENIGN 예측 | 0.167 | 0.000 | 0.000 |
| 전부 PROMPT_INJECTION 예측 | 0.167 | 1.000 | 1.000 |
| 규칙 기반선 | 0.672 [0.445–0.841] | 0.562 | 0.125 |
| 제로샷 mDeBERTa | 0.465 | 0.500 | 0.500 |
| 전용 가드 모델 (`protectai/deberta-v3-...`) | 0.333 † | **0.938** | **0.625** |

*`common/data/bench/gold.jsonl` 24건 기준. 대괄호는 95% bootstrap 신뢰구간 — **폭이 0.4에
가깝다.** 이 벤치로는 0.67과 0.80을 구분할 수 없다.*

† 2-class 모델이라 JAILBREAK를 구조적으로 못 맞힌다. **이 macro F1은 다른 행과 비교하면 안 된다.**

두 극단(전부 BENIGN / 전부 공격)의 macro F1이 같고 각각 한 지표에서 만점이라는 점을 본다.
**지표 하나를 목표로 삼으면 최악의 탐지기가 그 목표를 달성한다.**

## 수료 기준

높은 점수가 아니다. 다음 네 가지를 **설명할 수 있으면** 통과다.

1. 어휘가 거의 같은데 라벨이 반대인 문장(하드 네거티브)이 왜 따로 필요한가
2. `benign FPR`이 0인데도 그 탐지기를 배포하면 안 되는 경우
3. 전체 macro F1이 0.672인데 특정 slice가 0.000일 수 있는 이유
4. 1GB짜리 다국어 모델이 정규식 몇 줄에 진 이유
5. **전용 가드 모델의 macro F1(0.333)을 우리 모델과 비교하면 안 되는 이유**
6. **규칙 기반선 macro F1의 신뢰구간이 [0.445–0.841]일 때, "0.70짜리 모델이 더 낫다"고
   말할 수 있는가**

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

# 04 전용 가드 모델 — 진짜 기준선 (모델 약 700MB 다운로드)
python 01-beginner/04-run-pretrained/run_guard_model.py \
  --input common/data/bench/gold.jsonl --output runs/guard-pred.jsonl
python 01-beginner/03-evaluation-basics/evaluate.py \
  --gold common/data/bench/gold.jsonl --pred runs/guard-pred.jsonl

# 04 하드 네거티브만 따로 — 이 모델의 benign FPR은 0.750까지 오른다
python 01-beginner/04-run-pretrained/run_guard_model.py \
  --input common/data/bench/negatives.jsonl --output runs/guard-neg-pred.jsonl
python 01-beginner/03-evaluation-basics/evaluate.py \
  --gold common/data/bench/negatives.jsonl --pred runs/guard-neg-pred.jsonl
```

선택 — LLM에게 직접 판정시키는 기준선. **벤치 문장을 외부 API로 전송한다.**

사전 준비: `pip install anthropic` 후 `ANTHROPIC_API_KEY` 환경변수를 설정하거나
`ant auth login`으로 로그인한다. 준비가 끝나면:

```bash
python 01-beginner/04-run-pretrained/run_llm_judge.py \
  --input common/data/bench/gold.jsonl --output runs/judge-pred.jsonl
```

실행하면 외부 전송 확인을 묻는다. 스크립트나 CI에서 돌릴 때는 `--yes`를 붙인다.
