# 과제 · 운영 가능한 정책을 찾는다

## 목표

recall, FPR, **review 비용** 세 가지를 동시에 만족하는 임계값 조합을 찾고, 그 결정을 문서로 남긴다.

## 선행 조건

```bash
python 03-advanced/03-hybrid-gates/apply_policy.py \
  --input common/data/bench/gold.jsonl \
  --pred runs/models/mbert-v1/bench-pred.jsonl \
  --output runs/models/mbert-v1/bench-decisions.jsonl
```

기준 실측(24건, review 0.45 / block 0.80): allow 4 / **review 17** / block 3

---

## 과제 1 · 네 가지 판정 방식의 액션 분포를 비교한다

```bash
python - <<'PY'
import json, collections
from guardlab.io import read_jsonl
from guardlab.rules import predict_one
from guardlab.schema import ATTACK_LABELS

gold = read_jsonl("common/data/bench/gold.jsonl")
model = {json.loads(l)["id"]: json.loads(l) for l in open("runs/models/mbert-v1/bench-pred.jsonl")}

def stats(decide, name):
    at = ah = bt = bf = 0
    for s in gold:
        flagged = decide(s)
        if s.label == "BENIGN":
            bt += 1; bf += flagged
        else:
            at += 1; ah += flagged
    print(f"{name:14} attack_recall={ah/at:.3f}  benign_fpr={bf/bt:.3f}  차단={ah+bf}건")

def model_attack(s, t=0.5):
    return sum(model[s.id]["scores"][k] for k in ATTACK_LABELS) >= t

stats(lambda s: predict_one(s).label in ATTACK_LABELS, "규칙 단독")
stats(lambda s: model_attack(s), "모델 단독")
stats(lambda s: predict_one(s).label in ATTACK_LABELS or model_attack(s), "OR")
stats(lambda s: predict_one(s).label in ATTACK_LABELS and model_attack(s), "AND")
PY
```

| 방식 | attack recall | benign FPR | 차단 건수 |
|---|---:|---:|---:|
| 규칙 단독 | | | |
| 모델 단독 | | | |
| OR | | | |
| AND | | | |

**답할 것**:

1. OR은 recall과 FPR을 각각 어느 방향으로 움직이는가? 그것이 논리적으로 당연한 이유는?
2. AND는 어떤가?
3. 3단계 정책(`apply_policy.py`)은 OR·AND 중 어느 쪽에 가까운가? 왜 그 사이에 있다고 볼 수 있는가?

---

## 과제 2 · 운영 가능한 임계값을 찾는다 (핵심 과제)

`policy.json`을 복사해 세 가지 조합을 만들고 각각의 액션 비율을 잰다.

```bash
mkdir -p runs/exercise/03-gates
for cfg in "0.45 0.80" "0.80 0.95" "0.90 0.98"; do
  set -- $(echo $cfg)
  r=$1; b=$2
  echo "{\"review_threshold\": $r, \"block_threshold\": $b, \"rule_match_action\": \"review\"}" \
    > runs/exercise/03-gates/policy-$r-$b.json
  python 03-advanced/03-hybrid-gates/apply_policy.py \
    --input common/data/bench/gold.jsonl \
    --pred runs/models/mbert-v1/bench-pred.jsonl \
    --config runs/exercise/03-gates/policy-$r-$b.json \
    --output runs/exercise/03-gates/decisions-$r-$b.jsonl > /dev/null
  echo -n "review=$r block=$b  "
  python - <<PY
import json, collections
from guardlab.io import read_jsonl
gold = {s.id: s for s in read_jsonl("common/data/bench/gold.jsonl")}
rows = [json.loads(l) for l in open("runs/exercise/03-gates/decisions-$r-$b.jsonl")]
c = collections.Counter(x["action"] for x in rows)
n = len(rows)
# 공격이 allow로 빠져나간 비율 = 실질 미탐
leaked = sum(1 for x in rows if x["action"] == "allow" and gold[x["id"]].label != "BENIGN")
attacks = sum(1 for s in gold.values() if s.label != "BENIGN")
blocked_benign = sum(1 for x in rows if x["action"] == "block" and gold[x["id"]].label == "BENIGN")
benign = sum(1 for s in gold.values() if s.label == "BENIGN")
print(f"allow={c['allow']:2} review={c['review']:2} block={c['block']:2} | "
      f"review비율={c['review']/n:.1%} 공격유출={leaked}/{attacks} 정상차단={blocked_benign}/{benign}")
PY
done
```

| review / block | allow | review | block | review 비율 | 공격 유출 | 정상 차단 |
|---|---:|---:|---:|---:|---:|---:|
| 0.45 / 0.80 | | | | | | |
| 0.80 / 0.95 | | | | | | |
| 0.90 / 0.98 | | | | | | |

### 운영 비용표

일 100만 요청, 검토 1건당 30초, 검토 인력 1명이 하루 6시간 검토 가능하다고 가정한다.

| review 비율 | 일 검토 건수 | 필요 인력 | 실현 가능? |
|---:|---:|---:|---|
| 71% | 710,000 | ~985명 | ✗ |
| | | | |
| | | | |

**답할 것**: 세 조합 중 운영 가능한 것은? 그때 **공격이 몇 건 유출**되는가? 그 유출을 감수할 수 있는가?

---

## 과제 3 · 결정 근거(reasons)를 분석한다

```bash
python - <<'PY'
import json, collections
rows = [json.loads(l) for l in open("runs/models/mbert-v1/bench-decisions.jsonl")]
by_reason = collections.Counter()
for r in rows:
    by_reason[(r["action"], tuple(sorted(r["reasons"])))] += 1
for (action, reasons), n in sorted(by_reason.items(), key=lambda kv: -kv[1]):
    print(f"{n:3}건  {action:7} ← {', '.join(reasons)}")
PY
```

**답할 것**:

1. `rule:...`만으로 review가 된 건수는? 이 건들은 모델이 통과시켰는데 규칙이 잡은 것이다.
   실제로 공격이었는가?
2. `model_block_threshold`로 block된 건들은 실제로 공격이었는가?
3. `reasons`가 없다면 이 분석이 가능한가? 로그 설계에서 무엇을 배울 수 있는가?

---

## 과제 4 · 품질 게이트를 CI에 넣는다면

```bash
python 03-advanced/03-hybrid-gates/quality_gate.py \
  runs/models/mbert-v1/bench-report/report.json \
  --min-attack-recall 0.90 --max-benign-fpr 0.05
echo "exit code: $?"
```

기준을 현실적으로 조정해 통과하는 값을 찾는다.

```bash
for fpr in 0.05 0.15 0.30; do
  echo -n "max-benign-fpr=$fpr → "
  python 03-advanced/03-hybrid-gates/quality_gate.py \
    runs/models/mbert-v1/bench-report/report.json \
    --min-attack-recall 0.90 --max-benign-fpr $fpr 2>&1 | tail -1
done
```

**답할 것**:

1. 지금 모델이 통과하는 최소 기준은 무엇인가?
2. **기준을 낮춰서 통과시키는 것**과 **모델을 고쳐서 통과시키는 것** 중 무엇을 해야 하는가?
   기준을 낮추는 것이 정당한 경우가 있는가?
3. 이 게이트를 CI에 넣으면 무엇을 막을 수 있고 무엇을 못 막는가?

### 실물 CI를 읽는다

이 저장소에는 동작하는 워크플로가 들어 있다: [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml).
게이트의 핵심은 **사람이 잊어도 자동으로 막히는 것**이므로, `quality_gate.py`를 손으로
돌리는 것만으로는 이 레슨이 끝나지 않는다.

파일을 읽고 세 가지 설계 결정을 확인한다.

1. **왜 파인튜닝 모델이 아니라 규칙 기반선을 검사하는가?**
   (힌트: 520MB 가중치는 git에 없고, 학습은 몇 분 걸리며, CI는 매 PR마다 돈다)
2. **왜 CI 기준값이 운영 목표치(`recall >= 0.90`)가 아니라 `--min-macro-f1 0.60`인가?**
   운영 목표치를 쓰면 어떻게 되는가?
3. **왜 하드 네거티브 단계에서는 `--max-benign-fpr`만 검사하는가?**
   (힌트: 그 파일에는 BENIGN만 있다. attack recall 0.000이 무슨 뜻인지 다시 본다)

세 번째는 실제로 이 저장소에서 났던 버그다. `quality_gate.py`의 `--min-attack-recall`에
기본값 0.90이 있던 시절, 하드 네거티브 게이트가 항상 실패했다 — 공격 샘플이 없어
"해당 없음"인 0.000을 "미탐"으로 읽었기 때문이다. 지금은 **명시한 기준만 검사**한다.

**추가 과제**: 게이트가 실제로 실패하는 것을 한 번 확인한다. `--min-macro-f1 0.99`처럼
통과 불가능한 값을 넣고 종료 코드가 0이 아닌지 본다. **실패하는 것을 본 적 없는 게이트는
게이트가 아니다.**

---

## 과제 5 · source를 서버가 붙여야 하는 이유

`apply_policy.py`는 입력의 `source` 필드를 그대로 신뢰한다. 만약 그 값을 클라이언트가 보낸다면?

```bash
python - <<'PY'
import json
from guardlab.io import read_jsonl
from guardlab.rules import predict_one

rows = read_jsonl("common/data/bench/gold.jsonl")
attack = next(r for r in rows if r.label == "PROMPT_INJECTION")
print("원본:", attack.source, "|", attack.text[:50])

# 공격자가 source를 'system'으로 위장한다면?
attack.source = "system"
print("위장:", attack.source)
print()
print("만약 정책이 source=system을 신뢰해 검사를 건너뛴다면?")
PY
```

**답할 것** (정책 문서로 작성):

```markdown
## 신뢰 경계 정책

- `source` 값은 ___가 부여한다. 클라이언트 입력의 해당 필드는 ___한다.
- 이유:
- source별 처리 차이:
  - `user`: ___
  - `retrieved` / `tool`: ___
  - `system`: ___
- 이 정책이 없을 때 가능한 공격 시나리오:
```

---

## 과제 6 · 정책 문서 작성

최종 산출물이다. `runs/exercise/03-gates/POLICY.md`에 작성한다.

```markdown
# 탐지 정책 v1

## 임계값
- review_threshold: ___
- block_threshold: ___
- rule_match_action: ___
- 근거: (과제 2의 표를 인용)

## 액션별 시스템 동작
| 액션 | 동작 | 사용자에게 보이는 것 |
|---|---|---|
| allow | | |
| review | | |
| block | | |

## 실패 처리 (fail 정책)
- 모델 추론 실패 시: ___
- 입력 길이 초과 시: ___
- 백엔드 다운 시: ___
- 선택 이유: fail-open / fail-closed 중 ___를 고른 이유

## 회귀 게이트
- 최소 attack recall: ___
- 최대 benign FPR: ___
- 게이트 실패 시 절차: ___

## 롤백 조건
- 다음 중 하나면 이전 버전으로 되돌린다: ___

## 모니터링 지표
- ___ (차단률, review 비율, source별 분포, 지연시간 등)

## 이 정책이 보증하지 않는 것
- ___
```

---

## 정답 확인

- [ ] 규칙/모델/OR/AND 네 방식의 recall·FPR을 모두 쟀는가?
- [ ] 임계값 3조합의 **review 비율**과 운영 비용을 계산했는가?
- [ ] 운영 가능한 조합을 고르고, 그때의 **공격 유출 건수**를 명시했는가?
- [ ] `reasons` 분석으로 규칙과 모델의 기여를 분리했는가?
- [ ] 기준을 낮추는 것과 모델을 고치는 것의 차이를 판단했는가?
- [ ] `source`를 서버가 붙여야 하는 이유를 시나리오로 설명했는가?
- [ ] 정책 문서에 **fail 정책**과 **롤백 조건**을 적었는가?

## 막혔을 때

- **`apply_policy.py`가 KeyError를 낸다** → `--input`과 `--pred`의 id 집합이 달라서다. 같은 데이터의
  예측인지 확인한다.
- **review 비율이 계산과 다르다** → 규칙 신호도 review를 만든다. 모델 임계값만 올려도 규칙이 걸면
  review로 간다. `rule_match_action`을 `allow`로 바꿔 비교해 본다.
- **24건은 너무 적다** → 맞다. 합성 test(192건)로도 같은 실험을 돌려 본다.

## 제출물

- 과제 1의 결합 방식 비교표
- 과제 2의 임계값 3조합 표 + **운영 비용표**
- 과제 3의 reasons 분석
- 과제 4의 답 (기준을 낮출 것인가 모델을 고칠 것인가)
- 과제 5의 신뢰 경계 정책
- `POLICY.md`
