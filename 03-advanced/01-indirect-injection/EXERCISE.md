# 과제 · 긴 문서에서 recall과 FPR을 동시에 본다

## 목표

sliding window가 recall을 올리는 대신 **무엇을 대가로 치르는지** 문서 길이별로 측정한다.

## 선행 조건

```bash
python 03-advanced/01-indirect-injection/make_indirect_eval.py
python 02-intermediate/03-first-finetune/predict.py \
  --model runs/models/mbert-v1 --input runs/data/indirect-eval.jsonl \
  --output runs/models/mbert-v1/indirect-single.jsonl
python 03-advanced/01-indirect-injection/chunk_predict.py \
  --model runs/models/mbert-v1 --input runs/data/indirect-eval.jsonl \
  --output runs/models/mbert-v1/indirect-chunk.jsonl
```

기준 실측: 단일 truncation은 공격 3건 **전부 미탐**, sliding window는 3건 다 탐지하되 **정상 문서도 오탐**

---

## 과제 1 · 공격 위치별 결과를 정리한다

```bash
python - <<'PY'
import json
single = {json.loads(l)["id"]: json.loads(l) for l in open("runs/models/mbert-v1/indirect-single.jsonl")}
chunk  = {json.loads(l)["id"]: json.loads(l) for l in open("runs/models/mbert-v1/indirect-chunk.jsonl")}
gold   = {json.loads(l)["id"]: json.loads(l) for l in open("runs/data/indirect-eval.jsonl")}
print(f"{'id':18} {'gold':18} {'단일':22} {'window':22}")
for k in gold:
    s, c = single[k], chunk[k]
    print(f"{k:18} {gold[k]['label']:18} "
          f"{s['label'] + ' ' + format(s['score'], '.2f'):22} "
          f"{c['label'] + ' ' + format(c['score'], '.2f'):22}")
PY
```

| 문서 | 공격 위치 | 단일 예측 | window 예측 | 단일이 놓친 이유 |
|---|---|---|---|---|
| `indirect-start` | 맨 앞 | | | |
| `indirect-middle` | 중간 | | | |
| `indirect-end` | 맨 끝 | | | |
| `indirect-benign` | 없음 | | | — |

**답할 것**: `middle`과 `end`가 미탐된 이유는 명확하다(토큰 범위 밖). 그런데 `start`도 놓쳤다.
공격이 맨 앞 192토큰 안에 있는데 왜 놓쳤는가?

토큰 수를 직접 확인해 본다.

```bash
python - <<'PY'
from transformers import AutoTokenizer
from guardlab.io import read_jsonl
tok = AutoTokenizer.from_pretrained("runs/models/mbert-v1")
for r in read_jsonl("runs/data/indirect-eval.jsonl"):
    n = len(tok(r.text)["input_ids"])
    print(f"{r.id:18} {n:6}토큰  → window {(n // (192 - 48)) + 1}개 예상")
PY
```

---

## 과제 2 · stride를 바꿔 본다

`stride`는 window끼리 겹치는 토큰 수다. 0이면 겹치지 않는다.

```bash
for s in 0 32 64 96; do
  python 03-advanced/01-indirect-injection/chunk_predict.py \
    --model runs/models/mbert-v1 --input runs/data/indirect-eval.jsonl \
    --output runs/models/mbert-v1/indirect-s$s.jsonl --stride $s > /dev/null 2>&1
  echo -n "stride=$s: "
  python -c "
import json
rows = [json.loads(l) for l in open('runs/models/mbert-v1/indirect-s$s.jsonl')]
print(', '.join(f\"{r['id'].replace('indirect-','')}={r['label'][:2]}({r['score']:.2f})\" for r in rows))"
done
```

| stride | start | middle | end | benign | 공격 3건 탐지 수 |
|---:|---|---|---|---|---:|
| 0 | | | | | |
| 32 | | | | | |
| 64 | | | | | |
| 96 | | | | | |

**답할 것**:

1. stride를 키우면 window 수가 늘어나는가 줄어드는가? 왜?
2. stride=0일 때 생길 수 있는 문제는? (힌트: 공격 문장이 window 경계에 걸치면?)
3. stride를 키우면 지연시간은 어떻게 되는가?

---

## 과제 3 · 문서 길이별 오탐률 측정 (핵심 과제)

**정상 문서만** 여러 길이로 만들어 sliding window의 오탐률을 잰다.

```bash
python - <<'PY'
import json
from pathlib import Path
paragraph = "이 문서는 제품 배포 절차, 테스트 결과, 담당자 체크리스트를 설명한다. "
rows = []
for repeat in (1, 5, 20, 100):
    for i in range(5):                      # 길이당 5건
        rows.append({
            "id": f"len{repeat:03d}-{i}", "text": (paragraph * repeat) + f" 문서 번호 {i}.",
            "label": "BENIGN", "source": "retrieved", "language": "ko",
            "group_id": f"len-{repeat}", "meta": {"synthetic": True, "repeat": repeat},
        })
Path("runs/data/length-benign.jsonl").write_text(
    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
print(f"{len(rows)}건 생성")
PY

python 03-advanced/01-indirect-injection/chunk_predict.py \
  --model runs/models/mbert-v1 --input runs/data/length-benign.jsonl \
  --output runs/models/mbert-v1/length-chunk.jsonl

python - <<'PY'
import json, collections
gold = {json.loads(l)["id"]: json.loads(l) for l in open("runs/data/length-benign.jsonl")}
pred = {json.loads(l)["id"]: json.loads(l) for l in open("runs/models/mbert-v1/length-chunk.jsonl")}
ATTACK = {"PROMPT_INJECTION", "JAILBREAK"}
by = collections.defaultdict(lambda: [0, 0])
for sid, g in gold.items():
    bucket = by[g["meta"]["repeat"]]
    bucket[0] += 1
    bucket[1] += pred[sid]["label"] in ATTACK
print("문단 반복  건수  오탐  FPR")
for repeat in sorted(by):
    total, fp = by[repeat]
    print(f"{repeat:8}  {total:4}  {fp:4}  {fp/total:.3f}")
PY
```

| 문단 반복 | 대략 토큰 수 | window 수 | 정상 문서 오탐률 |
|---:|---:|---:|---:|
| 1 | | | |
| 5 | | | |
| 20 | | | |
| 100 | | | |

**답할 것**:

1. 문서가 길어질수록 오탐률이 어떻게 변하는가?
2. window 하나의 오탐률을 `p`라 하고 window가 `N`개일 때, 문서 오탐률의 이론값은?
   실측과 비교하면?
3. 이 문제를 완화하려면 어떤 방법을 쓰겠는가? (레슨 4절의 표에서 하나 고르고 이유를 적는다)

---

## 과제 4 · 완화책을 하나 구현한다

레슨 4절의 완화책 중 하나를 골라 직접 구현하고 재측정한다. 예: **상위 3개 window 평균**

```bash
python - <<'PY'
import json, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from guardlab.io import read_jsonl
from guardlab.schema import ATTACK_LABELS, LABELS

tok = AutoTokenizer.from_pretrained("runs/models/mbert-v1")
model = AutoModelForSequenceClassification.from_pretrained("runs/models/mbert-v1").eval()
attack_idx = [LABELS.index(l) for l in ATTACK_LABELS]
TOP_K = 3

for path in ("runs/data/indirect-eval.jsonl", "runs/data/length-benign.jsonl"):
    print(f"--- {path} ---")
    for row in read_jsonl(path):
        w = tok(row.text, truncation=True, max_length=192, stride=48,
                return_overflowing_tokens=True, padding=True, return_tensors="pt")
        w.pop("overflow_to_sample_mapping", None)
        with torch.inference_mode():
            probs = torch.softmax(model(**w).logits, dim=-1)
        attack = probs[:, attack_idx].sum(dim=1)
        top_k = torch.topk(attack, min(TOP_K, len(attack))).values.mean().item()
        print(f"  {row.id:18} gold={row.label:17} windows={len(attack):3} "
              f"max={attack.max():.3f} top{TOP_K}avg={top_k:.3f}")
PY
```

| 방식 | 공격 3건 탐지 | 정상 문서 오탐 |
|---|---:|---:|
| max (기본) | | |
| 상위 3개 평균 | | |

**답할 것**: 완화책이 FPR을 낮췄는가? recall은 유지됐는가? 어떤 threshold를 쓰면 둘 다 만족하는가?

---

## 과제 5 · 길이 초과를 어떻게 처리할지 정한다

우리 `predict.py`는 `truncation=True`로 조용히 자른다. 실무(참고 프로젝트)는 상한 초과 시 **413 에러**를
내고 게이트웨이가 그것을 차단으로 처리한다.

**답할 것** (정책 문서 형태로 작성):

```markdown
## 긴 입력 처리 정책

- 입력 상한: ___ 토큰
- 상한 초과 시: (a) 잘라서 판정 / (b) 거부 / (c) chunk 처리 후 판정
- 선택 이유:
- (a)를 고르면 안 되는 이유:
- chunk 처리 시 지연시간 상한: ___ ms, 초과하면 ___
- 이 정책이 실패할 때의 fallback: ___
```

힌트: (a)를 고르면 공격자가 앞에 정상 텍스트를 잔뜩 채워 넣어 항상 통과시킬 수 있다.

---

## 정답 확인

- [ ] 단일 truncation의 미탐 3건과 그 이유를 토큰 수로 설명했는가?
- [ ] stride를 바꿔가며 결과 변화를 기록했는가?
- [ ] **문서 길이별 오탐률**을 실측했는가? (이것이 이 레슨의 핵심)
- [ ] 완화책을 하나 구현하고 recall/FPR을 함께 봤는가?
- [ ] 길이 초과 처리 정책을 문서로 정했는가?

## 막혔을 때

- **chunk_predict가 OOM으로 죽는다** → 문서가 너무 길어 window가 많다. `--repeat`를 줄이거나
  배치를 나눠 처리한다.
- **window가 1개만 나온다** → 문서가 짧아서다. `make_indirect_eval.py --repeat 200`처럼 늘린다.
- **stride를 늘렸는데 window가 줄어든다** → `stride`는 겹침 크기다. 겹침이 크면 진행 폭
  (`max_length - stride`)이 작아져 window가 **늘어난다.** 결과가 반대라면 값을 다시 확인한다.

## 제출물

- 과제 1의 위치별 비교표 + `start` 미탐 이유
- 과제 2의 stride 표
- 과제 3의 **길이별 FPR 표** (필수)
- 과제 4의 완화책 구현 결과
- 과제 5의 긴 입력 처리 정책 문서
