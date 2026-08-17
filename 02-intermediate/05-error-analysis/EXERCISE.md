# 과제 · dev에서 threshold를 정하고 test에 한 번만 쓴다

## 목표

**test 규율**을 몸으로 익힌다. dev에서 결정을 내리고, test는 마지막에 한 번만 열어서 그 결정이 옳았는지
확인한다.

## 선행 조건

```bash
python 02-intermediate/03-first-finetune/predict.py \
  --model runs/models/mbert-v1 --input runs/data/v1/dev.jsonl \
  --output runs/models/mbert-v1/dev-pred.jsonl
```

> 이 과제를 하는 동안 **test 파일을 열지 않는다.** 과제 4에서 딱 한 번 연다.

---

## 과제 1 · 오류 지도를 그린다

```bash
python 02-intermediate/05-error-analysis/error_dump.py \
  --gold runs/data/v1/dev.jsonl --pred runs/models/mbert-v1/dev-pred.jsonl --limit 15
```

오류를 세 유형으로 묶어 건수를 센다.

| 유형 | 건수 | 비율 | 대표 사례 (id) |
|---|---:|---:|---|
| 정상 오탐 (`BENIGN → 공격`) | | | |
| 공격 미탐 (`공격 → BENIGN`) | | | |
| 유형 혼동 (`PI ↔ JB`) | | | |

**답할 것**:

1. 가장 큰 덩어리는 무엇인가?
2. 그 덩어리는 어느 `source`에 몰려 있는가?
3. 운영 관점에서 우선순위가 가장 높은 유형은? (가장 큰 덩어리와 같은가?)

---

## 과제 2 · 상위 오류 5건의 원문을 읽는다

집계만 보지 말고 **원문을 읽는다.** 가설은 여기서 나온다.

```bash
python -c "
import json
rows = [json.loads(l) for l in open('runs/models/mbert-v1/test-report/../dev-pred.jsonl')]
" 2>/dev/null
python 02-intermediate/05-error-analysis/error_dump.py \
  --gold runs/data/v1/dev.jsonl --pred runs/models/mbert-v1/dev-pred.jsonl --limit 5 | tail -7
```

각 사례에 대해 적는다.

| id | gold → pred | score | 이 문장에서 모델이 공격 신호로 오해했을 만한 부분 |
|---|---|---:|---|
| | | | |

**답할 것**: `score`가 0.9 이상인 오류가 몇 건인가? 이런 오류를 threshold로 고칠 수 있는가? 왜?

---

## 과제 3 · dev에서 threshold를 고른다

```bash
python 02-intermediate/05-error-analysis/threshold_sweep.py \
  --gold runs/data/v1/dev.jsonl --pred runs/models/mbert-v1/dev-pred.jsonl
```

**운영 제약을 먼저 정한다.** 아래 중 하나를 골라 적는다.

- (a) `benign FPR ≤ 0.20` — 정상 차단을 20% 이하로
- (b) `attack recall ≥ 0.95` — 공격을 95% 이상 잡기
- (c) 직접 정한 제약: ___

제약을 만족하는 threshold 중 **다른 지표가 가장 좋은 값**을 고른다.

| 항목 | 값 |
|---|---|
| 내가 고른 제약 | |
| 제약을 만족하는 threshold 범위 | |
| 최종 선택 | |
| 그때 dev attack recall | |
| 그때 dev benign FPR | |
| **선택 이유 한 줄** | |

> 여기서 고른 값은 **이제 바꾸지 않는다.** 과제 4에서 test에 그대로 적용한다.

---

## 과제 4 · test에 한 번 적용한다

**여기서 test를 처음이자 마지막으로 연다.**

```bash
# 1) test 예측 생성
python 02-intermediate/03-first-finetune/predict.py \
  --model runs/models/mbert-v1 --input runs/data/v1/test.jsonl \
  --output runs/models/mbert-v1/test-pred.jsonl

# 2) 과제 3에서 고른 threshold를 적용 (아래 THRESHOLD 값만 바꾼다)
THRESHOLD=0.9
python - <<PY
import json
ATTACK = {"PROMPT_INJECTION", "JAILBREAK"}
threshold = $THRESHOLD
gold = {json.loads(l)["id"]: json.loads(l) for l in open("runs/data/v1/test.jsonl")}
at = ah = bt = bf = 0
for line in open("runs/models/mbert-v1/test-pred.jsonl"):
    p = json.loads(line)
    flagged = sum(p["scores"][k] for k in ATTACK) >= threshold
    if gold[p["id"]]["label"] == "BENIGN":
        bt += 1; bf += flagged
    else:
        at += 1; ah += flagged
print(f"threshold={threshold}")
print(f"test attack_recall = {ah/at:.4f}")
print(f"test benign_fpr    = {bf/bt:.4f}")
PY
```

| | dev (과제 3) | test (지금) | 차이 |
|---|---:|---:|---:|
| attack recall | | | |
| benign FPR | | | |

**답할 것**:

1. dev에서 고른 threshold가 test에서도 제약을 만족했는가?
2. 만족하지 않았다면, 지금 threshold를 바꾸고 싶은 유혹이 드는가? **왜 바꾸면 안 되는가?**
3. 정말 바꿔야 한다면 올바른 절차는 무엇인가? (힌트: 새 dev가 필요하다)

---

## 과제 5 · 처방 하나를 제안한다

과제 1~2에서 찾은 가장 큰 오류 덩어리에 대해 **처방 하나**를 고르고 근거를 적는다.

```markdown
## 개선 제안 v2

### 관찰
- 가장 큰 오류 덩어리: ___ (___건, 전체 오류의 __%)
- 몰려 있는 slice: ___
- 대표 사례: [id] "원문 일부"

### 가설
모델이 ___를 ___로 오해하고 있다. 근거는 ___이다.

### 처방 (하나만)
- [ ] 데이터: ___ 템플릿을 ___개 추가
- [ ] 라벨: ___ 정의를 ___로 수정
- [ ] threshold: ___ → ___

### 검증 방법
- 같은 dev/test, 같은 채점기로 재측정
- 기대: ___ 지표가 ___만큼 개선
- 부작용 감시: ___ 지표가 ___보다 나빠지면 롤백

### 이 변경이 효과인지 노이즈인지 판단하는 법
- `02-intermediate/03` 과제 2에서 잰 seed 편차: ±___
- 이보다 작은 개선이면 결론 보류
```

이 문서가 다음 레슨(미니 프로젝트)의 입력이 된다.

---

## 정답 확인

- [ ] 오류를 세 유형으로 나누고 건수를 셌는가?
- [ ] 상위 오류 5건의 **원문을 읽고** 가설을 세웠는가?
- [ ] threshold를 **dev에서** 고르고, 고르기 전에 제약을 먼저 정했는가?
- [ ] test를 **한 번만** 열었는가?
- [ ] test 결과가 나빠도 threshold를 다시 고르지 않았는가?
- [ ] 처방을 **하나만** 골랐는가?

## 막혔을 때

- **threshold를 아무리 올려도 FPR이 안 떨어진다** → 모델이 오탐 문장에도 높은 확률을 준다는 뜻이다.
  이 경우 threshold로는 못 고친다. 데이터 처방으로 간다.
- **스윕 결과가 계단처럼 뚝뚝 끊긴다** → dev가 144건이라 한 건이 0.7%다. 표본이 작으면 곡선이 거칠다.
  실무에서는 수천 건으로 잰다.
- **`error_dump.py`가 KeyError를 낸다** → gold와 pred의 데이터가 다르다. 같은 split인지 확인한다.

## 제출물

- 과제 1의 오류 유형표
- 과제 2의 상위 5건 분석표
- 과제 3의 threshold 선택 근거
- 과제 4의 dev/test 비교표 + "왜 다시 고르면 안 되는가"에 대한 답
- 과제 5의 개선 제안서
