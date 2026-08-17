# 과제 · 채점기를 신뢰할 수 있는지 검증한다

## 목표

평가 하네스가 **잘못된 입력을 조용히 넘기지 않는지** 직접 확인하고, slice 리포트에서 모델의 진짜 약점을
찾아낸다.

## 선행 조건

```bash
python 02-intermediate/04-evaluation-harness/evaluate.py \
  --gold runs/data/v1/test.jsonl \
  --pred runs/models/mbert-v1/test-pred.jsonl \
  --out runs/models/mbert-v1/test-report
```

기준값: macro F1 **0.750**, attack recall **0.914**, benign FPR **0.328**

---

## 과제 1 · 채점기를 속여 본다

세 가지 방식으로 잘못된 입력을 넣고, 채점기가 **멈추는지** 확인한다.

```bash
mkdir -p runs/exercise/04-harness

# (a) 예측에서 한 줄 제거
head -191 runs/models/mbert-v1/test-pred.jsonl > runs/exercise/04-harness/missing.jsonl
python 02-intermediate/04-evaluation-harness/evaluate.py \
  --gold runs/data/v1/test.jsonl --pred runs/exercise/04-harness/missing.jsonl \
  --out runs/exercise/04-harness/r1

# (b) 다른 데이터의 예측을 넘김
python 02-intermediate/04-evaluation-harness/evaluate.py \
  --gold runs/data/v1/test.jsonl --pred runs/models/mbert-v1/bench-pred.jsonl \
  --out runs/exercise/04-harness/r2

# (c) 텍스트만 살짝 바꿈
python - <<'PY'
import json
rows = [json.loads(l) for l in open("runs/models/mbert-v1/test-pred.jsonl")]
rows[0]["text"] += " "                                   # 공백 하나 추가
with open("runs/exercise/04-harness/tampered.jsonl", "w") as f:
    f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
PY
python 02-intermediate/04-evaluation-harness/evaluate.py \
  --gold runs/data/v1/test.jsonl --pred runs/exercise/04-harness/tampered.jsonl \
  --out runs/exercise/04-harness/r3
```

| 케이스 | 멈췄는가? | 에러 메시지 요지 |
|---|---|---|
| (a) 예측 1건 누락 | | |
| (b) 다른 데이터의 예측 | | |
| (c) 텍스트에 공백 1개 추가 | | |

**답할 것**: (c)의 검사가 없다면 어떤 사고가 조용히 지나갈 수 있는가?
(힌트: v1 test와 v2 test의 id 체계가 같다면?)

---

## 과제 2 · slice에서 최악의 구간을 찾는다

```bash
python -c "
import json
r = json.load(open('runs/models/mbert-v1/test-report/report.json'))
print('전체:', r['macro_f1'], 'FPR:', r['benign_fpr'])
for field, slices in r['slices'].items():
    print(f'--- {field} ---')
    for value, s in sorted(slices.items(), key=lambda kv: kv[1]['macro_f1']):
        print(f\"  {value:12} n={s['n_samples']:3} macro_f1={s['macro_f1']:.3f} benign_fpr={s['benign_fpr']:.3f}\")
"
```

| slice | n | macro F1 | benign FPR | 전체 대비 |
|---|---:|---:|---:|---|
| | | | | |

**답할 것**:

1. 가장 나쁜 slice는 무엇이고 전체 대비 얼마나 떨어지는가?
2. `tool` slice의 benign FPR이 유난히 높다. 정상 도구 출력 문장 몇 건이 오탐되는가?
   (`n × FPR`로 계산)
3. 이 약점을 고치려면 데이터를 어떻게 바꿔야 하는가? 한 문장으로 가설을 적는다

---

## 과제 3 · 형식이 아니라 내용을 보는지 확인한다

과제 2의 가설이 "모델이 `[도구 출력]` 같은 **접두사**를 보고 판단한다"였다면, 실험으로 확인한다.

```bash
python - <<'PY'
import json, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from guardlab.schema import LABELS

tok = AutoTokenizer.from_pretrained("runs/models/mbert-v1")
model = AutoModelForSequenceClassification.from_pretrained("runs/models/mbert-v1").eval()

pairs = [
    ("[도구 출력] 배포 체크리스트 완료. 오류 없음.", "배포 체크리스트 완료. 오류 없음."),
    ("[검색 결과] 접근 권한 관련 자주 묻는 질문 모음이다.", "접근 권한 관련 자주 묻는 질문 모음이다."),
]
for with_prefix, without in pairs:
    for text in (with_prefix, without):
        p = torch.softmax(model(**tok(text, return_tensors="pt")).logits, -1)[0]
        scores = {LABELS[i]: round(float(v), 3) for i, v in enumerate(p)}
        print(f"{max(scores, key=scores.get):17} {scores}  | {text[:45]}")
    print()
PY
```

**답할 것**: 접두사만 떼면 판정이 바뀌는가? 바뀐다면 모델이 무엇을 학습한 것인가?
그것이 실제 서비스에서 왜 위험한가?

---

## 과제 4 · macro F1이 같은데 운영 판단이 다른 두 리포트

같은 macro F1을 내지만 benign FPR이 크게 다른 두 예측을 만든다. 가장 쉬운 방법은 **임계값을 다르게 적용**해
예측 라벨을 다시 만드는 것이다.

```bash
python - <<'PY'
import json
ATTACK = {"PROMPT_INJECTION", "JAILBREAK"}
rows = [json.loads(l) for l in open("runs/models/mbert-v1/test-pred.jsonl")]

for tag, threshold in [("t50", 0.50), ("t90", 0.90)]:
    out = []
    for r in rows:
        attack_score = sum(r["scores"][k] for k in ATTACK)
        if attack_score >= threshold:
            label = max(ATTACK, key=lambda k: r["scores"][k])
        else:
            label = "BENIGN"
        out.append({**r, "label": label, "score": r["scores"][label]})
    path = f"runs/exercise/04-harness/pred-{tag}.jsonl"
    with open(path, "w") as f:
        f.writelines(json.dumps(o, ensure_ascii=False) + "\n" for o in out)
    print("wrote", path)
PY

for t in t50 t90; do
  python 02-intermediate/04-evaluation-harness/evaluate.py \
    --gold runs/data/v1/test.jsonl \
    --pred runs/exercise/04-harness/pred-$t.jsonl \
    --out runs/exercise/04-harness/report-$t > /dev/null
done
python 02-intermediate/06-mini-project/compare_runs.py \
  runs/exercise/04-harness/report-t50/report.json \
  runs/exercise/04-harness/report-t90/report.json
```

| threshold | macro F1 | attack recall | benign FPR |
|---:|---:|---:|---:|
| 0.50 | | | |
| 0.90 | | | |

**답할 것**: 모델 가중치는 하나도 안 바뀌었는데 지표가 달라졌다. 보고서에 성능을 쓸 때 반드시 함께
적어야 하는 것은 무엇인가?

---

## 과제 5 · 서빙 레벨 재현성 미리보기

지금 예측은 PyTorch로 만들었다. `03-advanced/04`에서 ONNX로 변환한 뒤 **같은 입력에 같은 라벨이 나오는지**
확인할 것이다. 그 준비로, 지금 예측 결과를 기준 파일로 고정해 둔다.

```bash
cp runs/models/mbert-v1/test-pred.jsonl runs/models/mbert-v1/test-pred-fp32-reference.jsonl
python -c "
import json, collections
rows = [json.loads(l) for l in open('runs/models/mbert-v1/test-pred-fp32-reference.jsonl')]
print('기준 예측 저장:', len(rows), '건')
print('라벨 분포:', dict(collections.Counter(r['label'] for r in rows)))
"
```

**답할 것**: 나중에 ONNX 변환 후 라벨이 3건 뒤집혔다면, 그것을 "버그"로 볼 것인가 "허용 오차"로 볼
것인가? 판단 기준을 미리 정해 적는다. (힌트: 뒤집힌 방향이 미탐 쪽인가 오탐 쪽인가)

---

## 정답 확인

- [ ] 세 가지 잘못된 입력에 대해 채점기가 모두 멈추는 것을 확인했는가?
- [ ] 가장 나쁜 slice를 찾고 오탐 건수를 실제 숫자로 계산했는가?
- [ ] 모델이 내용이 아니라 형식을 보는지 실험으로 확인했는가?
- [ ] 임계값만 바꿔도 지표가 달라진다는 것을 표로 보였는가?
- [ ] 서빙 재현성 판단 기준을 미리 적어 두었는가?

## 막혔을 때

- **(b)가 안 멈춘다** → `bench-pred.jsonl`이 없을 수 있다. 먼저 bench 예측을 만든다.
- **과제 3의 모델 로딩이 느리다** → 정상이다. 한 번 로드에 몇 초 걸린다.
- **`report.json`에 `slices`가 없다** → `01-beginner/03`의 `evaluate.py`가 아니라
  `02-intermediate/04`의 것을 써야 한다. 전자는 `--json-out`, 후자는 `--out`이다.

## 제출물

- 과제 1의 표 (세 케이스 모두 멈췄는지)
- 과제 2의 slice 표와 개선 가설
- 과제 3의 실험 결과 (접두사 유무에 따른 판정 변화)
- 과제 4의 임계값 비교표
- 과제 5의 판단 기준 한 문단
