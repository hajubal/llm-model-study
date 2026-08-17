# 과제 · 지표를 직접 만들어 보고 속아 본다

## 목표

지표를 손으로 계산해 채점기 출력과 대조하고, **같은 macro F1인데 운영 판단이 정반대인 경우**를 직접 만든다.

## 선행 조건

```bash
python 01-beginner/02-rule-baseline/rule_detector.py \
  --input common/data/bench/gold.jsonl --output runs/rule-pred.jsonl
```

---

## 과제 1 · 전부 BENIGN 예측기 만들기

모든 샘플을 `BENIGN`으로 예측하는 파일을 만들고 채점한다.

```bash
mkdir -p runs/exercise/03-eval
python - <<'PY'
import json
out = open('runs/exercise/03-eval/all-benign.jsonl', 'w')
for line in open('common/data/bench/gold.jsonl'):
    if not line.strip():
        continue
    row = json.loads(line)
    out.write(json.dumps({
        "id": row["id"], "text": row["text"], "label": "BENIGN", "score": 1.0,
        "scores": {"BENIGN": 1.0, "PROMPT_INJECTION": 0.0, "JAILBREAK": 0.0},
    }, ensure_ascii=False) + "\n")
out.close()
PY
python 01-beginner/03-evaluation-basics/evaluate.py \
  --gold common/data/bench/gold.jsonl --pred runs/exercise/03-eval/all-benign.jsonl
```

기대 출력: accuracy **0.333**, macro F1 **0.167**, attack recall **0.000**, benign FPR **0.000**

기록할 것:

1. 이 탐지기의 benign FPR은 0이다. 이 값만 보고 배포해도 되는가? 왜 안 되는가?
2. 라벨 비율이 8/8/8이라 accuracy가 0.333이다. 실제 트래픽처럼 정상이 99%라면 accuracy는 얼마가 되는가?
3. 이 탐지기의 macro F1이 0.167인 이유를 계산으로 보인다. (힌트: BENIGN F1 = 0.5, 나머지 두 라벨 F1 = 0)

---

## 과제 2 · confusion matrix를 손으로 검산

`runs/rule-pred.jsonl`의 채점 결과에서 confusion matrix를 보고, **`JAILBREAK` 라벨의 P/R/F1을 손으로 계산**한
뒤 채점기 출력과 맞는지 확인한다.

```text
| gold \ pred | BENIGN | PROMPT_INJECTION | JAILBREAK |
| BENIGN      |      7 |                1 |         0 |
| PROMPT_INJ  |      3 |                5 |         0 |
| JAILBREAK   |      4 |                0 |         4 |
```

빈칸을 채운다.

```text
TP = ___    (JAILBREAK 행 ∩ JAILBREAK 열)
FP = ___    (JAILBREAK 열의 나머지 행 합)
FN = ___    (JAILBREAK 행의 나머지 열 합)

precision = TP / (TP + FP) = ___
recall    = TP / (TP + FN) = ___
F1        = 2PR / (P + R)  = ___
```

채점기가 출력한 값(P 1.000 / R 0.500 / F1 0.667)과 같은지 확인한다.

---

## 과제 3 · macro F1이 같은데 운영 판단이 반대인 두 경우

아래 두 confusion matrix를 만들어 채점기에 넣는다. 라벨당 8건, 총 24건으로 맞춘다.

**A 시나리오 — 미탐형** (공격을 놓치지만 정상은 안 막는다)

```text
| gold \ pred | BENIGN | PI | JB |
| BENIGN      |      8 |  0 |  0 |
| PI          |      4 |  4 |  0 |
| JB          |      4 |  0 |  4 |
```

**B 시나리오 — 오탐형** (공격은 다 잡지만 정상을 막는다)

```text
| gold \ pred | BENIGN | PI | JB |
| BENIGN      |      4 |  4 |  0 |
| PI          |      0 |  8 |  0 |
| JB          |      0 |  4 |  4 |
```

두 예측 파일을 만들어 채점하고 표를 채운다.

| 시나리오 | macro F1 | attack recall | benign FPR | 이 탐지기를 배포하겠는가? |
|---|---:|---:|---:|---|
| A (미탐형) | | | | |
| B (오탐형) | | | | |

그리고 답한다: **어느 쪽을 고르겠는가? 그 판단은 무엇에 달려 있는가?**
(힌트: 사내 도구인가 공개 서비스인가, 차단당한 사용자가 재시도할 수 있는가, review 경로가 있는가)

---

## 과제 4 · slice에서 약점 찾기

```bash
python 01-beginner/03-evaluation-basics/evaluate.py \
  --gold common/data/bench/gold.jsonl --pred runs/rule-pred.jsonl
```

출력의 slice 부분에서 다음을 찾는다.

1. 전체 macro F1(0.672)보다 **크게 낮은 slice** 두 개와 그 표본 수
2. 그 slice의 점수가 낮은 이유에 대한 가설 한 줄
3. 표본 수가 3건 이하인 slice — 이 숫자를 보고서에 쓸 때 어떤 단서를 붙여야 하는가?

---

## 정답 확인

- [ ] 전부 BENIGN 예측기의 네 지표를 모두 적었는가?
- [ ] `JAILBREAK`의 TP/FP/FN을 손으로 세어 채점기 값과 일치시켰는가?
- [ ] A/B 두 시나리오의 macro F1이 **거의 같은데** 운영 판단이 갈리는 것을 표로 보였는가?
- [ ] slice에서 전체 점수에 가려진 약점 두 개를 지목했는가?
- [ ] 표본 수가 적은 slice에 대해 "신뢰구간이 넓다"는 취지의 단서를 달았는가?

## 막혔을 때

- **예측 파일을 만들다 채점기가 id 불일치로 실패한다** → 정상이다. 채점기는 gold와 pred의 id 집합이
  정확히 같아야 통과시킨다. 빠진 id를 에러 메시지에서 확인한다.
- **`text 불일치` 에러가 난다** → 예측 파일의 `text`가 gold와 한 글자라도 달라서다. gold에서 그대로
  복사한다.
- **F1 계산이 안 맞는다** → FP는 **열** 방향(다른 라벨을 이 라벨로 예측), FN은 **행** 방향(이 라벨을 다른
  라벨로 예측)이다. 헷갈리면 표에 화살표를 그려 본다.

## 제출물

- `runs/exercise/03-eval/` 아래 예측 파일들
- 과제 2의 손계산 (TP/FP/FN과 P/R/F1)
- 과제 3의 비교표와 "어느 쪽을 배포할지 + 그 이유"
- 과제 4에서 찾은 약점 slice 2개와 가설
