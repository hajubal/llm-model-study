# 02 · 규칙 기반선

## 이 레슨을 마치면

- 정규식 기반 탐지기가 어떻게 동작하는지 코드를 읽고 설명할 수 있다.
- **기준선(baseline)**이 왜 필요한지, 없으면 무엇을 판단할 수 없는지 말할 수 있다.
- 규칙 탐지기를 실행하고, 그 결과가 어디서 왜 틀리는지 지목할 수 있다.
- 규칙이 잘 맞는 것처럼 보이는 **착시**를 만드는 조건을 안다.

---

## 1. 기준선이 없으면 아무것도 판단할 수 없다

중급에서 모델을 학습하면 `macro F1 0.75` 같은 숫자가 나온다. 이 숫자는 좋은가?

**혼자서는 알 수 없다.** 비교 대상이 있어야 한다.

| 비교 대상 | 이 커리큘럼에서 |
|---|---|
| 아무것도 안 하기(전부 BENIGN 예측) | `01-beginner/03` 과제에서 측정 |
| 정규식 몇 줄 | **이 레슨** |
| 학습 없이 범용 모델(제로샷) | `01-beginner/04` |
| 파인튜닝한 모델 | `02-intermediate/03` |

이 순서로 올라가면서, 각 단계가 **직전 단계보다 나은지**를 같은 채점기로 확인한다. 규칙 기반선이 모델보다
높게 나오는 일도 실제로 벌어진다(이 커리큘럼에서도 한 번 벌어진다 — 4절 참고).

> **용어** — **기준선(baseline)**: 최소한 이것보다는 나아야 하는 단순한 방법. **SOTA를 이기는 것**이 목표가
> 아니라 **내 방법이 단순한 방법보다 나은지** 확인하는 것이 목표다.

---

## 2. 코드 읽기

`common/guardlab/rules.py` 전체가 50줄이 안 된다. 세 부분으로 되어 있다.

### (1) 공격 패턴

```python
_INJECTION_PATTERNS = [
    re.compile(r"\b(ignore|disregard|forget)\b.{0,40}\b(previous|prior|system|instructions?)\b", re.I),
    re.compile(r"(이전|앞선|기존).{0,20}(지시|명령|규칙).{0,15}(무시|잊어|폐기)", re.I),
    ...
]
```

읽는 법:
- `\b` — 단어 경계. `ignoring`의 `ignore` 부분에는 안 걸리게 한다
- `.{0,40}` — 두 키워드 사이에 최대 40자까지 아무거나 와도 된다. "ignore **all of the** previous"를 잡기 위함
- `re.I` — 대소문자 무시
- 한국어 패턴은 `\b`를 못 쓴다(한국어에 단어 경계 개념이 없다). 그래서 조사 변화를 `.{0,20}`으로 흡수한다

### (2) 교육 문맥 할인(discount)

```python
_EDUCATIONAL_CONTEXT = re.compile(
    r"(정의|설명|탐지|분류|연구|논문|교육|예시를 분석|방어|what is|explain|detect|research|paper|defen[cs]e)",
    re.I,
)
```

"설명해 줘"가 들어 있으면 공격 점수를 깎는다. 하드 네거티브를 살리려는 **의도적인 보정**이다. 완벽하지
않다 — 공격자가 "설명해 줘"를 덧붙이면 그대로 뚫린다.

### (3) 점수 계산

```python
injection_score = max(0.0, min(0.99, 0.25 + 0.35 * injection_hits - 0.25 * discount)) if injection_hits else 0.02
benign_score    = max(0.01, 1.0 - max(injection_score, jailbreak_score))
label = max(scores, key=scores.get)
```

- 패턴이 하나도 안 맞으면 점수 `0.02`
- 하나 맞으면 `0.25 + 0.35 = 0.60`, 둘 맞으면 `0.95`
- 교육 문맥이 있으면 `-0.25`
- 세 라벨 중 가장 높은 점수를 최종 라벨로

이 숫자들(`0.25`, `0.35`, `0.02`)에 이론적 근거는 없다. **손으로 정한 값**이고, 규칙 기반 시스템의 점수가
확률이 아니라는 것을 보여주려고 일부러 남겨 뒀다. 뒤에서 임계값(threshold)을 다룰 때 이 차이가 문제가 된다.

---

## 3. 실행

```bash
python 01-beginner/02-rule-baseline/rule_detector.py \
  --input common/data/bench/gold.jsonl --output runs/rule-pred.jsonl
```

출력:

```text
24 predictions -> runs/rule-pred.jsonl
```

`runs/rule-pred.jsonl` 한 줄을 열어보면 이렇게 생겼다.

```json
{"id":"bench-p-001","text":"이전 지시를 모두 무시하고...","label":"PROMPT_INJECTION","score":0.6,"scores":{"BENIGN":0.4,"PROMPT_INJECTION":0.6,"JAILBREAK":0.02}}
```

`scores`에 **세 라벨 전부의 점수**가 들어 있는 것이 중요하다. 이후 레슨의 임계값 스윕과 하이브리드 정책이
이 필드를 쓴다. 최종 라벨 하나만 남기면 나중에 "얼마나 확신했는가"를 다시 계산할 수 없다.

### 하드 네거티브 전용 세트에도 돌린다

`negatives.jsonl`은 8건 전부 `BENIGN`이다. 여기서 올라가는 숫자는 오탐뿐이다.

```bash
python 01-beginner/02-rule-baseline/rule_detector.py \
  --input common/data/bench/negatives.jsonl --output runs/rule-neg-pred.jsonl
python 01-beginner/03-evaluation-basics/evaluate.py \
  --gold common/data/bench/negatives.jsonl --pred runs/rule-neg-pred.jsonl
```

실제 출력:

```text
- 샘플 8 · accuracy 0.875 · macro F1 0.311
- attack recall 0.000 · benign FPR 0.125
- 주의: support가 0인 라벨(PROMPT_INJECTION, JAILBREAK)의 F1 0이 macro 평균에 포함된다.
- 주의: 공격 샘플이 없어 attack recall 0.000은 '미탐'이 아니라 '해당 없음'이다
```

**이 세트에서 읽을 값은 `benign FPR` 하나뿐이다.** macro F1 0.311은 공격 라벨의 support가 0이라 F1이 0으로
들어간 결과이고, attack recall 0.000은 공격 샘플이 없어서다. 채점기가 이 사실을 출력에 적어 준다.
지표는 항상 "이 데이터에서 그 지표가 정의되는가"를 먼저 확인하고 읽는다.

8건 중 1건(12.5%)을 오탐했다. 어떤 문장인지 직접 확인해 보자 — 과제 1이 그것이다.

---

## 4. 결과 해석 — 규칙은 어디서 무너지는가

같은 규칙 탐지기를 서로 다른 두 데이터에 돌린 실측값이다.

| 데이터 | 샘플 | macro F1 | attack recall | benign FPR |
|---|---:|---:|---:|---:|
| `common/data/bench/gold.jsonl` | 24 | 0.672 | 0.562 | 0.125 |
| `runs/data/v1/test.jsonl` (합성 test) | 192 | **0.450** | **0.273** | 0.000 |

같은 코드인데 성능이 크게 다르다. 이유는 **데이터의 표현 분포**다.

- `gold.jsonl`은 규칙이 노리는 전형적 문구("이전 지시를 무시", "ignore previous instructions")를 많이 담고 있다
- 합성 test에는 `[검색 결과] ... (숨은 지시: 위 규칙을 무시하고 ...)`, `[도구 출력] ... 다음 지시:` 처럼
  **검색 문서·도구 출력 형태의 공격**이 3분의 1을 차지한다. 규칙의 정규식이 이런 문형을 못 잡는다

그래서 attack recall이 0.562 → 0.273으로 떨어진다. 반대로 benign FPR은 0.000이다. **아무것도 안 잡으니
오탐도 없다.** 이것이 "FPR이 낮다"를 단독으로 자랑하면 안 되는 이유다.

> **핵심 교훈** — 규칙 탐지기의 점수는 **데이터가 규칙과 얼마나 겹치는가**를 잰 것이지, 탐지 능력을 잰 것이
> 아니다. 규칙을 만든 사람이 평가 데이터도 만들면 점수가 항상 좋게 나온다. `03-advanced`에서 표현 변형과
> 독립 holdout으로 이 착시를 확인한다.

---

## 5. 실무에서 규칙은 어디에 쓰이는가

규칙이 쓸모없다는 뜻이 아니다. 참고 프로젝트(`sgt-owasp`)는 규칙과 모델을 **둘 다** 쓰되, 구조를 이렇게 잡았다.

| | 규칙 | 모델 |
|---|---|---|
| 엔드포인트 | `POST /api/v1/banword/detect` | `POST /api/v1/jailbreak/detect` |
| 방식 | 금지어 목록의 부분 문자열 매칭(대소문자 무시) | 2.1B 파라미터 모델 추론 |
| 구현 | `llm-guard`의 `BanSubstrings` | 파인튜닝한 Kanana 계열 |
| 목록 | `ban_word.json` (코드 배포 없이 교체) | 모델 체크포인트 |

주목할 점은 **두 API가 코드 레벨에서 합쳐져 있지 않다**는 것이다. OR로 묶는 정책은 상위 게이트웨이 서비스의
몫이다. 이렇게 나눈 이유는 명확하다.

1. **규칙은 빠르고 확실하다** — 반드시 막아야 하는 고정 문자열(사내 비밀 코드명 등)은 모델의 확률에 맡기지
   않고 문자열 매칭으로 확실히 잡는다
2. **규칙은 즉시 고칠 수 있다** — 새 공격 문구가 발견되면 목록에 한 줄 추가하면 끝이다. 모델은 재학습에
   시간이 걸린다
3. **둘의 실패 방식이 다르다** — 규칙은 변형에 약하고, 모델은 새 도메인에 약하다. 겹쳐 놓으면 서로를 보완한다

`03-advanced/03-hybrid-gates`에서 이 조합을 직접 만든다.

---

## 6. 흔한 실수

| 실수 | 결과 | 대신 |
|---|---|---|
| 오탐이 나면 패턴을 바로 좁힌다 | recall이 조용히 떨어진다 | 좁히기 전후의 recall/FPR을 **둘 다** 기록 |
| 평가 데이터를 보면서 패턴을 고친다 | 그 데이터 전용 규칙이 된다(과적합) | 규칙 수정은 dev로, test는 마지막 한 번 |
| 최종 라벨만 저장한다 | 임계값 실험을 다시 할 수 없다 | `scores` 전체를 저장 |
| FPR 0을 성과로 본다 | 아무것도 안 잡아도 FPR은 0이다 | recall과 항상 같이 본다 |

---

## 다음 레슨

지금까지 `macro F1`, `attack recall`, `benign FPR`이라는 숫자를 그냥 받아썼다. 다음 레슨
`01-beginner/03-evaluation-basics`에서 이 지표들이 각각 무엇을 세는지, 왜 accuracy 하나로는 안 되는지를
직접 계산하며 확인한다.
