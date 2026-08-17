# 03 · 규칙 + 모델 + 정책 게이트

## 이 레슨을 마치면

- 모델 출력(확률)과 운영 결정(차단 여부)이 **다른 층**이라는 것을 안다.
- allow / review / block 3단계 정책을 구현하고 각 구간의 비율을 잰다.
- 규칙과 모델을 결합할 때 recall과 FPR이 어떻게 움직이는지 실측한다.
- CI 품질 게이트로 회귀를 막는 방법과 그 한계를 안다.

---

## 1. 확률을 결정으로 바꾸는 것은 별도의 일이다

모델은 `{"BENIGN": 0.2, "PROMPT_INJECTION": 0.7, "JAILBREAK": 0.1}` 같은 숫자를 낸다. 여기서
"차단한다"까지 가려면 여러 결정이 필요하다.

```text
모델 확률  →  임계값 적용  →  규칙 신호 결합  →  액션 선택  →  사용자에게 보여줄 메시지
   0.8          block?          rule OR?         block/review     "요청을 처리할 수 없습니다"
```

이 결정들은 **모델과 독립적으로 바뀐다.** 모델을 재학습하지 않고도 임계값을 조정할 수 있고, 그 반대도
가능하다. 그래서 실무에서는 이 둘을 **분리해서 버전 관리**한다.

참고 프로젝트(`sgt-owasp`)가 정확히 그 구조다.

```text
[탐지 엔진 서비스]                        [상위 게이트웨이]
POST /api/v1/jailbreak/detect             ┌──────────────────────┐
  → {isSafe, riskScore, riskType}    ───▶ │ 두 결과를 어떻게      │
POST /api/v1/banword/detect               │ 합칠지, 무엇을 차단할지│
  → {금지어 매칭 여부}               ───▶ │ 결정한다              │
                                          └──────────────────────┘
```

탐지 엔진은 **판정만** 한다. 규칙(금지어)과 모델은 **별도 API**이며 엔진 안에서 합쳐지지 않는다.
OR로 묶을지, 어느 쪽을 우선할지는 게이트웨이의 정책이다.

이렇게 나누는 이유:

- 모델 팀과 정책 팀의 배포 주기가 다르다
- 같은 탐지 엔진을 여러 제품이 서로 다른 정책으로 쓸 수 있다
- 정책만 롤백하는 것이 모델 롤백보다 훨씬 빠르다

---

## 2. 왜 2단계가 아니라 3단계인가

`allow` / `block` 두 값만 쓰면 애매한 구간을 어느 한쪽으로 밀어야 한다.

```text
공격 확률 0.55 → block하면? 정상 사용자가 차단당한다
              → allow하면? 공격이 통과한다
```

**`review`** 를 두면 세 번째 선택지가 생긴다.

| 액션 | 조건 | 시스템 동작 예시 |
|---|---|---|
| `allow` | 점수 낮음 + 규칙 신호 없음 | 정상 처리 (단, 도구 권한은 여전히 최소) |
| `review` | 애매한 점수 **또는** 규칙 신호 | 사람 검토 큐 / 제한된 도구 경로 / 추가 확인 요구 |
| `block` | 높은 공격 확률 | 거절. 사용자에게는 일반적 사유만 제공 |

`review`의 실질적 형태는 서비스마다 다르다.

- 사람이 확인하는 큐에 넣기(비용 높음, 정확)
- 도구 호출을 막고 텍스트 응답만 허용(비용 낮음)
- "정말 이 작업을 원하십니까?" 확인 요구
- 더 무겁고 정확한 2차 모델로 재판정

> **주의** — 참고 프로젝트의 탐지 API는 `isSafe` boolean 하나만 반환한다. 3단계는 그 위 게이트웨이나
> 애플리케이션이 만드는 층이다. "실무는 다 3단계를 쓴다"가 아니라, **어디에 그 층을 둘지가 설계 선택**이다.

---

## 3. 코드 읽기

```python
# 03-advanced/03-hybrid-gates/apply_policy.py
def decide(sample, prediction, config):
    attack_score = sum(prediction.scores.get(label, 0.0) for label in ATTACK_LABELS)
    reasons = []
    if attack_score >= config["block_threshold"]:          # 1. 높은 점수 → 즉시 block
        reasons.append("model_block_threshold")
        return "block", reasons
    rule = predict_one(sample)                             # 2. 규칙도 확인
    if rule.label in ATTACK_LABELS:
        reasons.append(f"rule:{rule.label}")
    if attack_score >= config["review_threshold"]:         # 3. 중간 점수
        reasons.append("model_review_threshold")
    if reasons:                                            # 4. 신호가 하나라도 있으면
        return config.get("rule_match_action", "review"), reasons
    return "allow", ["below_thresholds"]
```

설계 포인트:

- **`reasons`를 항상 남긴다.** 왜 그 결정이 내려졌는지 로그·감사에 필요하다. "block" 한 글자만 남기면
  나중에 원인을 알 수 없다
- **규칙 신호는 block이 아니라 review로 보낸다.** 규칙은 오탐이 많으므로 즉시 차단하지 않는다.
  `rule_match_action` 설정으로 바꿀 수 있다
- **설정은 별도 파일**(`policy.json`)이다

```json
{
  "review_threshold": 0.45,
  "block_threshold": 0.80,
  "rule_match_action": "review"
}
```

이 파일은 코드와 **따로 버전 관리**한다. 임계값 변경이 코드 리뷰·배포 없이 가능해야 한다.
참고 프로젝트도 같은 방식이고, 설정 서버를 쓸 때는 값이 바뀌면 서비스 인스턴스가 백그라운드에서
재생성되어 **재시작 없이 반영**된다.

---

## 4. 실행

```bash
python 03-advanced/03-hybrid-gates/apply_policy.py \
  --input common/data/bench/gold.jsonl \
  --pred runs/models/mbert-v1/bench-pred.jsonl \
  --output runs/models/mbert-v1/bench-decisions.jsonl
```

```bash
python -c "
import json, collections
print(dict(collections.Counter(
    json.loads(l)['action'] for l in open('runs/models/mbert-v1/bench-decisions.jsonl'))))"
```

실측(24건):

| 입력 예측 | allow | review | block |
|---|---:|---:|---:|
| 규칙 기반선 예측 | 14 | 9 | 1 |
| 모델 예측 | 4 | **17** | 3 |

모델 예측을 쓰면 `review`가 17건(71%)이다. **너무 많다.** 사람이 검토한다면 감당할 수 없는 양이다.

원인은 임계값이다. `review_threshold=0.45`인데 우리 모델의 공격 점수 분포가 그 위에 몰려 있다.
`02-intermediate/05`에서 봤듯 이 모델은 threshold를 0.9까지 올려도 recall이 유지된다. 즉
**`review_threshold`를 훨씬 높여야 한다.** 과제 2가 이것이다.

### 결정 파일 형태

```json
{"id":"bench-p-001","action":"block","reasons":["model_block_threshold"]}
{"id":"bench-b-003","action":"review","reasons":["rule:PROMPT_INJECTION"]}
{"id":"bench-b-005","action":"allow","reasons":["below_thresholds"]}
```

두 번째 줄이 흥미롭다. 모델은 통과시켰는데 **규칙이 걸어서** review로 갔다. 이것이 하이브리드의 효과다
— 앞 레슨에서 봤듯 규칙과 모델의 약점이 다르므로 서로를 보완한다.

---

## 5. 품질 게이트 — 회귀를 막는 장치

```bash
python 03-advanced/03-hybrid-gates/quality_gate.py \
  runs/models/mbert-v1/bench-report/report.json \
  --min-attack-recall 0.90 --max-benign-fpr 0.05
```

```text
QUALITY GATE FAILED: benign_fpr 0.250 > 0.050
```

**exit code 1**로 끝난다. CI에서 이 코드로 배포를 막는다.

```python
if report["attack_recall"] < args.min_attack_recall:
    failures.append(f"attack_recall {report['attack_recall']:.3f} < {args.min_attack_recall:.3f}")
if report["benign_fpr"] > args.max_benign_fpr:
    failures.append(f"benign_fpr {report['benign_fpr']:.3f} > {args.max_benign_fpr:.3f}")
if failures:
    raise SystemExit("QUALITY GATE FAILED: " + "; ".join(failures))
```

우리 모델은 `attack_recall 0.938`로 recall 기준은 통과하지만 `benign_fpr 0.250`으로 실패한다.
**이것이 정상적인 결과다.** 교육용 기본 모델은 이 게이트를 통과하지 못한다. 통과시키는 것이 중급~고급
과제의 목표다.

### 게이트가 보증하지 않는 것

| 게이트가 하는 일 | 게이트가 못 하는 일 |
|---|---|
| 이 벤치마크에서 성능이 떨어지면 막는다 | 벤치마크에 없는 공격을 막는다 |
| 알려진 회귀를 잡는다 | 벤치마크가 현실을 대표한다고 보증한다 |
| 배포 전 자동 확인 | 배포 후 분포 변화 감지 |

**게이트 통과 = 안전**이 아니다. 게이트는 "예전보다 나빠지지 않았다"만 말한다.

---

## 6. 실무의 안전장치

참고 프로젝트에서 가져올 만한 운영 패턴들이다.

### fail-closed

```text
모델 출력이 정해진 라벨이 아님        → UNKNOWN → unsafe로 처리
추론 중 예외 발생                     → is_safe=False, risk_type="ERROR", risk_score=0.5
추론 백엔드 자체가 죽음               → 예외를 올려 500. 조용히 통과시키지 않는다
필터를 끄려면                         → 환경변수를 명시적으로 켜야 하고, CRITICAL 로그가 남는다
```

**"판정 불가"를 "안전"으로 바꾸지 않는다**는 원칙이다. `03-advanced/01`의 길이 초과 → 413 → 차단도
같은 원칙의 적용이다.

### Circuit Breaker

원격 추론 백엔드를 쓸 때, 연속 실패가 임계치(예: 3회)에 도달하면 회로를 열어 일정 시간(예: 30초)
호출을 중단한다. 죽은 백엔드에 요청을 계속 던져 전체 지연시간을 끌어올리는 것을 막는다.

이때 **회로가 열린 동안 무엇을 반환할지**가 정책 결정이다. fail-closed라면 차단이다.

### 설정에 남아 있는 함정

그 프로젝트에는 `fail_mode = open | closed` 형태의 설정 키가 아직 존재하지만, **런타임 분기에서는
제거되어 항상 fail-closed로 동작**한다. 그 설정은 이제 "fail-open을 의도한 설정을 기동 시 탐지해
경고하는" 방어 장치로만 남아 있다.

교훈: **설정 파일에 있는 값이 실제로 동작한다고 가정하면 안 된다.** 정책을 문서화할 때는 코드에서
그 값이 실제로 읽히는지 확인한다.

---

## 7. review 비용을 계산에 넣는다

`review`는 공짜가 아니다. 사람이 본다면 인건비, 2차 모델이면 지연시간과 비용이다.

```text
일 100만 요청 × review 비율 5% = 5만 건/일
사람 검토 1건 30초 → 416시간/일 → 불가능
```

그래서 **review 비율은 운영 제약**이다. 임계값을 고를 때 recall/FPR만이 아니라 review 비율도 함께 본다.

| 임계값 조합 | attack recall | benign FPR | review 비율 | 일 검토 건수 |
|---|---:|---:|---:|---:|
| review 0.45 / block 0.80 | | | 71% | 불가능 |
| review 0.80 / block 0.95 | | | | |
| review 0.90 / block 0.98 | | | | |

과제 2에서 이 표를 채운다.

---

## 8. 흔한 실수

| 실수 | 결과 | 대신 |
|---|---|---|
| 규칙 OR 모델로 recall만 올린다 | FPR 폭증 | 두 지표를 같이 본다 |
| 규칙 매칭을 즉시 block | 오탐이 그대로 차단 | review로 보낸다 |
| `reasons`를 안 남긴다 | 사후 분석 불가 | 결정 근거를 항상 기록 |
| 정책을 코드에 하드코딩 | 변경마다 배포 필요 | 설정 파일로 분리 |
| review 비율을 안 잰다 | 운영 불가능한 정책 | 비용을 계산에 포함 |
| 게이트 통과를 안전으로 해석 | 잘못된 확신 | "회귀 없음"으로만 해석 |

---

## 다음 레슨

정책까지 만들었다. 다음은 **서빙**이다. `03-advanced/04-serving-onnx-latency`에서 모델을 ONNX로 변환하고,
**변환 전후 판정이 같은지 먼저 확인한 뒤** 지연시간을 잰다.
