# 과제 · 규칙을 고쳐 보고 대가를 측정한다

## 목표

규칙 하나를 수정했을 때 **무엇이 좋아지고 무엇이 나빠지는지**를 숫자로 확인한다. 이 감각이 이후 모든
개선 루프의 기본이 된다.

## 선행 조건

```bash
python 01-beginner/02-rule-baseline/rule_detector.py \
  --input common/data/bench/gold.jsonl --output runs/rule-pred.jsonl
python 01-beginner/03-evaluation-basics/evaluate.py \
  --gold common/data/bench/gold.jsonl --pred runs/rule-pred.jsonl \
  --json-out runs/rule-report.json
```

기준값(이 값에서 시작한다): macro F1 **0.672**, attack recall **0.562**, benign FPR **0.125**

---

## 과제 1 · 오탐 1건과 미탐 2건을 지목한다

먼저 하드 네거티브 세트에서 오탐된 1건을 찾는다.

```bash
python 01-beginner/02-rule-baseline/rule_detector.py \
  --input common/data/bench/negatives.jsonl --output runs/rule-neg-pred.jsonl
python 02-intermediate/05-error-analysis/error_dump.py \
  --gold common/data/bench/negatives.jsonl --pred runs/rule-neg-pred.jsonl
```

이어서 `gold.jsonl`에서 놓친 공격(미탐)을 본다.

```bash
python 02-intermediate/05-error-analysis/error_dump.py \
  --gold common/data/bench/gold.jsonl --pred runs/rule-pred.jsonl
```

기록할 것:

1. 오탐된 정상 문장 1건 — **어느 정규식**에 걸렸는가? (`common/guardlab/rules.py`에서 찾는다)
2. 놓친 공격 2건 — 왜 안 걸렸는가? (키워드가 없어서 / 거리가 40자를 넘어서 / 표현이 달라서)

> 미탐 목록을 볼 때 `score`가 0.98로 높은 것들이 먼저 나온다. 이는 **BENIGN이라고 강하게 확신하며 틀린**
> 경우다. 확신 있게 틀리는 사례가 가장 위험하다.

---

## 과제 2 · 패턴을 **하나만** 수정한다

`common/guardlab/rules.py`에서 패턴 하나만 고친다. 예시 방향:

- 놓친 공격을 잡도록 `_INJECTION_PATTERNS`에 표현 추가
- 오탐을 줄이도록 `_EDUCATIONAL_CONTEXT`에 단어 추가
- 키워드 사이 거리 `.{0,40}`을 `.{0,80}`으로 넓히기

**한 번에 하나만 바꾼다.** 두 개를 동시에 바꾸면 어느 것이 효과였는지 알 수 없다.

수정 후 다시 측정한다.

```bash
python 01-beginner/02-rule-baseline/rule_detector.py \
  --input common/data/bench/gold.jsonl --output runs/rule-pred-v2.jsonl
python 01-beginner/03-evaluation-basics/evaluate.py \
  --gold common/data/bench/gold.jsonl --pred runs/rule-pred-v2.jsonl \
  --json-out runs/rule-report-v2.json
python 02-intermediate/06-mini-project/compare_runs.py \
  runs/rule-report.json runs/rule-report-v2.json
```

`compare_runs.py`가 표를 출력한다.

```text
| run | macro F1 | attack recall | benign FPR | n |
|---|---:|---:|---:|---:|
| runs | 0.672 | 0.562 | 0.125 | 24 |
| runs | ... | ... | ... | 24 |
```

### 기록 양식

| 바꾼 것 | macro F1 | attack recall | benign FPR |
|---|---:|---:|---:|
| (원본) | 0.672 | 0.562 | 0.125 |
| (내 수정 내용 한 줄) | | | |

---

## 과제 3 · 오탐을 0으로 만들고 대가를 잰다

이번에는 목표를 하나만 둔다: **`negatives.jsonl`에서 benign FPR을 0.000으로 만든다.**

가장 쉬운 방법은 `_EDUCATIONAL_CONTEXT`에 단어를 추가하거나 할인 폭(`0.25`)을 키우는 것이다.

FPR을 0으로 만든 뒤, **`gold.jsonl`의 attack recall이 얼마나 떨어졌는지** 반드시 함께 기록한다.

| 상태 | negatives FPR | gold attack recall |
|---|---:|---:|
| 원본 | 0.125 | 0.562 |
| 오탐 제거 후 | 0.000 | ? |

이 표가 이 레슨의 핵심 산출물이다. **오탐을 0으로 만드는 것은 언제나 가능하다** — 아무것도 안 잡으면 된다.
문제는 그때 공격을 몇 개나 놓치느냐다.

---

## 과제 4 · 규칙이 절대 못 잡는 문장 만들기

의미는 그대로인데 규칙을 통과하는 공격 문장을 3개 만든다. 힌트가 되는 방향:

- 키워드 사이를 40자 이상 벌린다
- "무시" 대신 다른 표현("건너뛰어", "적용하지 마", "고려하지 않아도 돼")
- 문장을 검색 결과·도구 출력 형태로 감싼다

```bash
# 만든 문장을 파일로 저장한 뒤
python 01-beginner/02-rule-baseline/rule_detector.py \
  --input runs/exercise/02-rule/bypass.jsonl \
  --output runs/exercise/02-rule/bypass-pred.jsonl
python 01-beginner/03-evaluation-basics/evaluate.py \
  --gold runs/exercise/02-rule/bypass.jsonl \
  --pred runs/exercise/02-rule/bypass-pred.jsonl
```

`attack recall`이 0에 가까우면 성공이다. 이 문장들은 중급에서 학습한 모델이 잡는지 다시 확인할 재료가 된다.

> 이 과제는 방어 평가용이다. 만든 문장을 외부 서비스에 시험하지 않는다.

---

## 정답 확인

- [ ] 오탐 1건이 **어느 정규식**에 걸렸는지 파일:줄 번호로 지목했는가?
- [ ] 수정 전후를 `compare_runs.py`로 같은 표에 놓고 비교했는가?
- [ ] FPR을 0으로 만들었을 때 **recall이 떨어진 값**을 적었는가?
- [ ] 한 번에 하나씩만 바꿨는가?
- [ ] 우회 문장 3건이 실제로 규칙을 통과했는가?

## 막혔을 때

- **정규식을 어떻게 읽는지 모르겠다** → `common/guardlab/rules.py`의 패턴을 하나 골라
  [regex101.com](https://regex101.com) 같은 도구에 붙여 넣고 테스트 문장을 넣어 본다. 어느 부분이
  매칭되는지 색으로 보여준다.
- **수정했는데 점수가 그대로다** → `__pycache__`가 남아 있을 수 있다. `guardlab`은 editable 설치이므로
  보통 즉시 반영되지만, 안 되면 파이썬 프로세스를 새로 띄운다.
- **compare_runs가 run 이름을 똑같이 보여준다** → 리포트를 서로 다른 폴더에 저장한다. 이 스크립트는
  `report.json`의 **부모 폴더명**을 run 이름으로 쓴다.

## 제출물

- 수정한 `rules.py`의 diff (또는 바꾼 줄 인용)
- 과제 2·3의 비교표 (수치 채운 것)
- 우회 문장 3건과 그때의 attack recall
- 한 줄 결론: "이 규칙 수정은 ___를 얻고 ___를 잃었다"
