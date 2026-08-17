# 03 · 평가 기초

## 이 레슨을 마치면

- confusion matrix를 보고 precision / recall / F1을 **손으로 계산**할 수 있다.
- accuracy 하나로 판단하면 안 되는 이유를 예시로 설명할 수 있다.
- `attack recall`과 `benign FPR`이 운영에서 각각 무엇을 뜻하는지 말할 수 있다.
- 전체 점수에 가려진 약점을 **slice**로 찾아낼 수 있다.

---

## 1. accuracy는 왜 못 믿는가

`accuracy`는 맞힌 비율이다. 간단하지만 클래스 비율에 쉽게 속는다.

실제 서비스 트래픽은 대부분 정상이다. 정상 99%, 공격 1%인 환경에서 **"전부 정상"이라고만 답하는 탐지기**는
accuracy 99%를 받는다. 공격은 하나도 못 잡는데도 그렇다.

이 커리큘럼의 벤치는 라벨이 균등(8/8/8)이라 덜 극단적이지만, 그래도 확인할 수 있다.

```bash
# 모든 샘플을 BENIGN으로 예측하는 파일을 만들어 채점해 보면
- 샘플 24 · accuracy 0.333 · macro F1 0.167
- attack recall 0.000 · benign FPR 0.000
```

`benign FPR 0.000`만 보면 완벽해 보인다. 하지만 `attack recall 0.000` — 공격을 하나도 못 잡았다.
**두 지표를 항상 같이 봐야 하는 이유**가 이것이다.

---

## 2. confusion matrix 읽는 법

confusion matrix는 "정답이 A인데 B로 예측한 건수"를 전부 적은 표다. 행이 정답(gold), 열이 예측(pred)이다.

규칙 기반선의 실제 결과다.

```text
| gold \ pred | BENIGN | PROMPT_INJECTION | JAILBREAK |
|-------------|-------:|-----------------:|----------:|
| BENIGN      |      7 |                1 |         0 |
| PROMPT_INJ  |      3 |                5 |         0 |
| JAILBREAK   |      4 |                0 |         4 |
```

읽는 법:

- **대각선**(7, 5, 4)이 맞힌 것. 합 16. 전체 24건이므로 accuracy = 16/24 = **0.667**
- **첫 번째 열의 대각선 아래**(3 + 4 = 7)가 공격을 정상으로 놓친 것 = **미탐(false negative)**
- **첫 번째 행의 대각선 오른쪽**(1 + 0 = 1)이 정상을 공격으로 잡은 것 = **오탐(false positive)**
- `PROMPT_INJECTION` 열의 세 번째 행(0)과 `JAILBREAK` 열의 두 번째 행(0)은 **공격 유형 간 혼동**

이 표에서 가장 큰 문제가 바로 보인다. `JAILBREAK` 8건 중 4건을 `BENIGN`으로 놓쳤다. 규칙이 "제한 없는",
"DAN mode" 같은 특정 키워드에만 반응하기 때문이다.

---

## 3. precision / recall / F1 손으로 계산하기

한 라벨을 기준으로 세 가지 수를 센다. `PROMPT_INJECTION`을 예로 든다.

```text
TP (true positive)  = 정답도 PI, 예측도 PI          = 5   (표의 2행 2열)
FP (false positive) = 정답은 다른 것, 예측은 PI      = 1   (1행 2열 + 3행 2열 = 1 + 0)
FN (false negative) = 정답은 PI, 예측은 다른 것      = 3   (2행 1열 + 2행 3열 = 3 + 0)
```

여기서:

```text
precision = TP / (TP + FP) = 5 / 6 = 0.833   "PI라고 한 것 중 진짜 PI의 비율"
recall    = TP / (TP + FN) = 5 / 8 = 0.625   "진짜 PI 중 잡아낸 비율"
F1        = 2PR / (P + R)  = 2 × 0.833 × 0.625 / 1.458 = 0.714
```

- **precision이 낮다** = 헛짚는다 = 정상 사용자가 자주 차단당한다
- **recall이 낮다** = 놓친다 = 공격이 통과한다
- **F1**은 둘의 조화평균이다. 한쪽이 0이면 F1도 0이 된다(산술평균과 다른 점)

세 라벨의 F1을 단순 평균한 것이 **macro F1**이다.

```text
macro F1 = (0.636 + 0.714 + 0.667) / 3 = 0.672
```

`macro`는 라벨을 **똑같은 비중**으로 본다는 뜻이다. 샘플이 많은 라벨에 끌려가지 않으므로, 소수 클래스를
무시하는 모델을 걸러낸다.

> **용어 정리**
> - **TP/FP/FN/TN**: 한 라벨을 "양성"으로 놓고 센 4분면
> - **precision(정밀도)**: 예측 기준 정확도 · **recall(재현율)**: 정답 기준 포착률
> - **macro 평균**: 라벨별로 계산 후 단순 평균 · **micro 평균**: 전체를 한 덩어리로 계산(샘플 많은 라벨에 좌우됨)

---

## 4. 운영이 실제로 보는 두 지표

3-class F1은 모델 품질 지표다. 하지만 운영에서 내려야 하는 결정은 대개 **차단할까 말까** 하나다. 그래서
따로 두 지표를 본다.

```python
# common/guardlab/eval.py
if sample.label == "BENIGN":
    benign_total += 1
    benign_fp += int(pred.label in ATTACK_LABELS)      # 정상인데 공격으로 예측
else:
    attack_total += 1
    attack_hit += int(pred.label in ATTACK_LABELS)     # 공격을 공격으로 예측(유형 틀려도 인정)
```

| 지표 | 정의 | 운영 의미 | 나빠지면 |
|---|---|---|---|
| **attack recall** | 공격 중 공격으로 잡은 비율 | 방어 성공률 | 공격이 통과 |
| **benign FPR** | 정상 중 공격으로 잡은 비율 | 정상 사용자 차단률 | 사용자 이탈, CS 폭증 |

**중요한 차이**: `PROMPT_INJECTION`을 `JAILBREAK`로 잘못 예측하면 3-class F1에서는 오답이지만, attack
recall에서는 **성공**으로 센다. 둘 다 차단 대상이기 때문이다. 그래서 F1이 낮은데 attack recall이 높은
상황이 정상적으로 나온다(중급 모델에서 실제로 그렇다: F1 0.637 / recall 1.000).

---

## 5. 실행과 출력 해석

```bash
python 01-beginner/03-evaluation-basics/evaluate.py \
  --gold common/data/bench/gold.jsonl --pred runs/rule-pred.jsonl \
  --json-out runs/rule-report.json
```

출력 마지막 부분이 **slice** 결과다.

```text
### slice: language
- en: n=6, macro_f1=0.822, benign_fpr=0.000
- ko: n=15, macro_f1=0.677, benign_fpr=0.200
- mixed: n=3, macro_f1=0.167, benign_fpr=0.000

### slice: source
- retrieved: n=3, macro_f1=0.444, benign_fpr=0.000
- tool: n=1, macro_f1=0.000, benign_fpr=0.000
- user: n=20, macro_f1=0.711, benign_fpr=0.143
```

전체 macro F1은 0.672였다. 그런데 쪼개 보면:

- `mixed`(한영 혼용) 3건에서 **0.167** — 거의 못 맞힌다
- `tool`(도구 출력) 1건에서 **0.000** — 완전 실패
- `user` 20건이 전체 점수를 떠받치고 있다

**전체 점수는 다수 slice의 점수다.** 실제 사고는 소수 slice에서 난다. 검색 문서와 도구 출력은 건수가 적지만
간접 인젝션의 주 경로이므로, 여기가 0점이면 심각한 문제다.

> `tool: n=1`처럼 표본이 1건인 slice의 숫자는 **신뢰할 수 없다**. 0.000이라는 값 자체보다 "이 slice를 잴
> 데이터가 없다"는 사실이 문제다. 중급에서 데이터를 다시 만들 때 이것을 고친다.

### 채점기가 강제하는 것

```python
missing = gold_ids - pred_by_id.keys()
extra = pred_by_id.keys() - gold_ids
if missing or extra:
    raise ValueError(f"gold/pred id 불일치 missing={...} extra={...}")
if pred.text != sample.text:
    raise ValueError(f"{sample.id}: gold/pred text 불일치")
```

- 예측은 **행 순서가 아니라 `id`로** 정답과 짝지어진다
- 하나라도 빠지거나 남으면 **에러로 중단**한다. 조용히 일부만 채점하지 않는다
- 텍스트까지 대조해서, 다른 데이터의 예측 파일을 잘못 넘기는 사고를 막는다

평가 코드가 관대하면 잘못된 점수를 조용히 만들어낸다. 채점기는 까다로울수록 좋다.

---

## 6. 실무는 어떤 지표를 보는가

참고 프로젝트(`sgt-owasp`)의 평가 스크립트에는 **F1이 아예 없다.** 대신 이런 지표를 나눠서 본다.

| 지표 | 무엇을 재는가 |
|---|---|
| `probe_fpr` | 실제 고객 트래픽을 대신하는 프로브 문장의 오탐률 |
| `salmon_safe_fpr` | 학습 분포 안의 정상 문장 오탐률 |
| `holdout_fpr` | 학습에 안 쓴 도메인의 오탐률 |
| `jb_block` / `pi_block` | 카테고리별 차단율(= recall) |
| `sensitive_overblock` | **정당한 민감 논의**를 과차단한 비율 |
| `flip_vs_model_level` | fp32(모델)와 fp16(서빙)의 판정이 뒤집힌 비율 |
| `length_bucket`별 block_rate | 입력 길이 구간(`<20 / 20-35 / 35-55 / 55+`)별 차단율 |

배울 점 세 가지다.

1. **FPR을 하나로 뭉치지 않는다.** 학습 분포 안 / 홀드아웃 / 실고객 대리를 나눠 잰다. 셋의 차이가 곧
   일반화 격차다.
2. **과차단을 별도 지표로 만든다.** `sensitive_overblock`은 우리의 하드 네거티브와 같은 문제의식이다.
3. **서빙 단계의 수치 오차까지 잰다.** `flip_vs_model_level`은 fp16 반올림 때문에 라벨이 뒤집힌 전례가
   있어서 만든 회귀 지표다. 모델이 좋아도 **서빙에서 달라지면 소용없다**.

우리 커리큘럼은 macro F1을 쓰지만, 이유는 교육용으로 지표 하나에 요약이 필요해서다. 제품에서는 위처럼
**대응이 다른 실패마다 지표를 따로 두는 편**이 낫다.

---

## 7. 흔한 실수

| 실수 | 왜 문제인가 |
|---|---|
| accuracy만 보고한다 | 클래스 불균형에서 무의미 |
| FPR만 자랑한다 | 아무것도 안 잡아도 0이다 |
| 전체 점수만 본다 | 소수 slice의 실패가 가려진다 |
| 표본 1~3건 slice의 수치를 그대로 믿는다 | 우연에 좌우된다. 표본 수를 항상 함께 적는다 |
| test를 보며 반복 튜닝한다 | test가 사실상 학습 데이터가 된다 |

---

## 다음 레슨

규칙 기반선의 점수(0.672)를 알았다. 다음은 **학습하지 않은 범용 모델**이 이보다 나은지 확인한다.
`01-beginner/04-run-pretrained`에서 제로샷 분류기를 같은 채점기로 돌린다.
