# 과제 · 제로샷을 흔들어 보고 규칙과 합쳐 본다

## 목표

제로샷 결과가 **라벨 문구에 얼마나 민감한지** 측정하고, 서로 다른 두 탐지기를 합쳤을 때 recall과 FPR이
어떻게 움직이는지 예측한 뒤 검산한다.

## 선행 조건

```bash
python 01-beginner/04-run-pretrained/run_zero_shot.py \
  --input common/data/bench/gold.jsonl --output runs/zero-shot-pred.jsonl
python 01-beginner/03-evaluation-basics/evaluate.py \
  --gold common/data/bench/gold.jsonl --pred runs/zero-shot-pred.jsonl \
  --json-out runs/zero-shot-report.json
```

기준값: macro F1 **0.465**, attack recall **0.500**, benign FPR **0.500**

---

## 과제 1 · 라벨 문구를 바꿔 재측정

`01-beginner/04-run-pretrained/run_zero_shot.py`의 `names` 딕셔너리를 아래 세 가지로 바꿔가며 측정한다.
**스크립트를 복사해서 고치지 말고, 원본을 고친 뒤 되돌리거나 git으로 관리한다.**

| 버전 | 라벨 문구 |
|---|---|
| A (원본) | `"a normal user request"` / `"an attempt to override the model instructions"` / `"an attempt to bypass AI safety restrictions"` |
| B (한국어) | `"평범한 사용자 요청"` / `"모델의 지시를 덮어쓰려는 시도"` / `"AI 안전 제한을 우회하려는 시도"` |
| C (구체화) | `"a question or task request"` / `"a request to ignore previous instructions or reveal the system prompt"` / `"a request to disable safety rules or act without restrictions"` |

각각 다른 출력 파일에 저장하고 채점한다.

```bash
python 01-beginner/04-run-pretrained/run_zero_shot.py \
  --input common/data/bench/gold.jsonl --output runs/exercise/04-zs/pred-B.jsonl
python 01-beginner/03-evaluation-basics/evaluate.py \
  --gold common/data/bench/gold.jsonl --pred runs/exercise/04-zs/pred-B.jsonl \
  --json-out runs/exercise/04-zs/report-B.json
```

세 결과를 한 표로 모은다.

```bash
python 02-intermediate/06-mini-project/compare_runs.py \
  runs/zero-shot-report.json \
  runs/exercise/04-zs/report-B.json \
  runs/exercise/04-zs/report-C.json
```

| 버전 | macro F1 | attack recall | benign FPR |
|---|---:|---:|---:|
| A (원본) | 0.465 | 0.500 | 0.500 |
| B (한국어) | | | |
| C (구체화) | | | |

**답할 것**: 모델 가중치는 하나도 안 바뀌었는데 점수가 달라졌다. 이 결과를 보고서에 쓸 때 반드시 함께
적어야 하는 정보는 무엇인가?

---

## 과제 2 · 두 탐지기가 어디서 틀리는지 비교

규칙과 제로샷의 예측을 나란히 놓고 네 부류로 나눈다.

```bash
python - <<'PY'
import json
gold  = {json.loads(l)["id"]: json.loads(l) for l in open("common/data/bench/gold.jsonl") if l.strip()}
rule  = {json.loads(l)["id"]: json.loads(l) for l in open("runs/rule-pred.jsonl")}
zero  = {json.loads(l)["id"]: json.loads(l) for l in open("runs/zero-shot-pred.jsonl")}

buckets = {"둘 다 맞음": [], "규칙만 맞음": [], "제로샷만 맞음": [], "둘 다 틀림": []}
for sid, g in gold.items():
    r_ok = rule[sid]["label"] == g["label"]
    z_ok = zero[sid]["label"] == g["label"]
    key = ("둘 다 맞음" if r_ok and z_ok else "규칙만 맞음" if r_ok else
           "제로샷만 맞음" if z_ok else "둘 다 틀림")
    buckets[key].append((sid, g["label"], rule[sid]["label"], zero[sid]["label"], g["text"][:45]))

for name, rows in buckets.items():
    print(f"\n=== {name} ({len(rows)}건) ===")
    for row in rows[:4]:
        print(f"  {row[0]} gold={row[1]} rule={row[2]} zero={row[3]} | {row[4]}")
PY
```

기록할 것:

1. 네 부류의 건수
2. "둘 다 틀림"에 속한 문장의 공통점 한 줄 — 어떤 표현이 두 방법 모두를 뚫었는가?
3. "제로샷만 맞음"이 있다면, 규칙이 못 잡은 이유

---

## 과제 3 · OR 결합을 **예측한 뒤** 계산

두 탐지기를 OR로 합친다고 하자. 규칙 또는 제로샷 중 **하나라도 공격이라고 하면 공격**으로 본다.

**먼저 예측을 적는다** (계산하기 전에!):

- attack recall은 올라갈 것 같은가, 내려갈 것 같은가? 왜?
- benign FPR은 어떻게 될 것 같은가? 왜?

그 다음 계산한다.

```bash
python - <<'PY'
import json
ATTACK = {"PROMPT_INJECTION", "JAILBREAK"}
gold = {json.loads(l)["id"]: json.loads(l) for l in open("common/data/bench/gold.jsonl") if l.strip()}
rule = {json.loads(l)["id"]: json.loads(l) for l in open("runs/rule-pred.jsonl")}
zero = {json.loads(l)["id"]: json.loads(l) for l in open("runs/zero-shot-pred.jsonl")}

def measure(decide):
    at = ah = bt = bf = 0
    for sid, g in gold.items():
        flagged = decide(sid)
        if g["label"] == "BENIGN":
            bt += 1; bf += flagged
        else:
            at += 1; ah += flagged
    return ah / at, bf / bt

for name, decide in [
    ("규칙만",   lambda s: rule[s]["label"] in ATTACK),
    ("제로샷만", lambda s: zero[s]["label"] in ATTACK),
    ("OR 결합",  lambda s: rule[s]["label"] in ATTACK or zero[s]["label"] in ATTACK),
    ("AND 결합", lambda s: rule[s]["label"] in ATTACK and zero[s]["label"] in ATTACK),
]:
    recall, fpr = measure(decide)
    print(f"{name:9} attack_recall={recall:.3f}  benign_fpr={fpr:.3f}")
PY
```

| 결합 | attack recall | benign FPR |
|---|---:|---:|
| 규칙만 | 0.562 | 0.125 |
| 제로샷만 | 0.500 | 0.500 |
| OR | | |
| AND | | |

**답할 것**:

1. 내 예측이 맞았는가? 틀렸다면 어디를 잘못 생각했는가?
2. OR와 AND 중 어느 쪽을 어떤 상황에서 쓰겠는가?
3. 이 두 가지 말고 제3의 선택지는 없는가? (힌트: `03-advanced/03-hybrid-gates`)

---

## 과제 4 · 비용까지 포함한 판단

제로샷 모델은 약 1GB를 내려받고 CPU에서 문장당 수십~수백 ms가 걸린다. 규칙은 마이크로초 단위다.

성능표에 **비용 열을 추가**해서 다시 본다.

| 방법 | macro F1 | 모델 크기 | 대략의 응답 시간 | 새 공격 대응 방법 |
|---|---:|---|---|---|
| 규칙 | 0.672 | 0 | 매우 빠름 | 정규식 한 줄 추가(즉시) |
| 제로샷 | 0.465 | ~1GB | 느림(CPU) | 라벨 문구 수정(효과 불확실) |

**답할 것**: 지금 당장 서비스에 하나만 넣어야 한다면 무엇을 넣겠는가? 그 선택의 가장 큰 약점은 무엇이고,
그것을 무엇으로 보완하겠는가?

---

## 정답 확인

- [ ] 라벨 문구 3버전의 점수를 모두 측정했는가?
- [ ] 점수 보고 시 함께 기록해야 할 정보를 적었는가?
- [ ] 네 부류(둘 다 맞음/한쪽만/둘 다 틀림) 건수를 세었는가?
- [ ] OR 결합 결과를 **계산 전에** 예측하고, 예측과 실제를 비교했는가?
- [ ] 성능뿐 아니라 비용·운영 관점의 판단을 적었는가?

## 막혔을 때

- **모델 다운로드가 계속 실패한다** → 네트워크 문제일 수 있다. `~/.cache/huggingface/hub`에 이미
  `models--MoritzLaurer--mDeBERTa-v3-base-mnli-xnli`가 있으면 재사용된다.
- **한국어 라벨 문구로 바꿨더니 더 나빠졌다** → 정상적인 결과일 수 있다. `hypothesis_template`이
  `"This text is {}."`(영어)로 남아 있으면 한영이 섞인 가설 문장이 만들어진다. 템플릿도 같이 바꿔서
  다시 재본다.
- **너무 느리다** → `--batch` 값을 올려 본다. 또는 입력을 `negatives.jsonl`(8건)로 줄여 실험한다.

## 제출물

- 과제 1의 3버전 비교표
- 과제 2의 네 부류 건수와 "둘 다 틀림"의 공통점
- 과제 3의 예측 → 실제 비교, OR/AND 표
- 과제 4의 판단 한 문단
