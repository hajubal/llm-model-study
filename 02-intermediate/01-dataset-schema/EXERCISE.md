# 과제 · 누수를 직접 만들어 보고 검사기로 잡는다

## 목표

검사기가 **무엇을 잡고 무엇을 못 잡는지** 직접 확인한다. 검사를 통과했다는 것이 데이터가 깨끗하다는 뜻이
아님을 실험으로 안다.

## 선행 조건

```bash
python 02-intermediate/02-synthetic-data/gen_synth.py --out runs/data/v1
python 02-intermediate/01-dataset-schema/inspect_data.py runs/data/v1
```

마지막 줄에 `OK: id 중복 없음, group_id leakage 없음`이 나와야 한다.

---

## 과제 1 · 의도적으로 누수를 만든다

`runs/data/v1`을 복사한 뒤 test의 한 샘플에 train의 `group_id`를 붙인다.

```bash
mkdir -p runs/exercise/01-schema/leaky
cp runs/data/v1/*.jsonl runs/exercise/01-schema/leaky/
python - <<'PY'
import json
train = [json.loads(l) for l in open("runs/exercise/01-schema/leaky/train.jsonl")]
test  = [json.loads(l) for l in open("runs/exercise/01-schema/leaky/test.jsonl")]
victim = test[0]["group_id"]
test[0]["group_id"] = train[0]["group_id"]           # 누수 주입
with open("runs/exercise/01-schema/leaky/test.jsonl", "w") as f:
    f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in test)
print(f"{victim} -> {train[0]['group_id']} 로 변경")
PY
python 02-intermediate/01-dataset-schema/inspect_data.py runs/exercise/01-schema/leaky
```

기대 결과: `ValueError: group_id '...'가 train/test에 중복되었습니다`로 **중단**된다.

**답할 것**: 이 검사가 없었다면 학습·평가는 정상적으로 끝났을 것이다. 그때 test 점수는 실제보다 높게
나올까 낮게 나올까? 왜?

---

## 과제 2 · 검사기를 통과하는 누수 만들기

이번에는 **검사기가 못 잡는 누수**를 만든다. 같은 문장을 복제하되 `group_id`만 다르게 붙인다.

```bash
mkdir -p runs/exercise/01-schema/sneaky
cp runs/data/v1/*.jsonl runs/exercise/01-schema/sneaky/
python - <<'PY'
import json
train = [json.loads(l) for l in open("runs/exercise/01-schema/sneaky/train.jsonl")]
test  = [json.loads(l) for l in open("runs/exercise/01-schema/sneaky/test.jsonl")]

# test에서 5건을 골라 train에 복제한다. id와 group_id만 바꾼다.
for i, row in enumerate(test[:5]):
    clone = dict(row)
    clone["id"] = f"clone-{i:03d}"
    clone["group_id"] = f"clone-group-{i:03d}"       # 다른 그룹으로 위장
    train.append(clone)

with open("runs/exercise/01-schema/sneaky/train.jsonl", "w") as f:
    f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in train)
print("train에 test 문장 5건 복제 완료")
PY
python 02-intermediate/01-dataset-schema/inspect_data.py runs/exercise/01-schema/sneaky
```

기대 결과: **통과한다.** `OK: id 중복 없음, group_id leakage 없음`

**답할 것**:

1. 검사기는 왜 이것을 못 잡았는가?
2. 이 상태로 학습하면 test 점수는 어떻게 되는가?
3. 이런 누수를 잡으려면 무엇을 추가해야 하는가?

### 직접 잡아 보기

텍스트 중복을 검사하는 코드를 짜 본다.

```bash
python - <<'PY'
import json, collections
rows = {}
for split in ("train", "dev", "test"):
    for line in open(f"runs/exercise/01-schema/sneaky/{split}.jsonl"):
        rows.setdefault(json.loads(line)["text"], []).append(split)

overlap = {t: s for t, s in rows.items() if len(set(s)) > 1}
print(f"split을 넘나드는 동일 문장: {len(overlap)}건")
for text, splits in list(overlap.items())[:3]:
    print(f"  {sorted(set(splits))}: {text[:60]}")
PY
```

---

## 과제 3 · near-duplicate는 왜 더 어려운가

과제 2의 검사(정확히 같은 문자열)로도 못 잡는 경우를 만든다.

```bash
python - <<'PY'
import json
test = [json.loads(l) for l in open("runs/data/v1/test.jsonl")]
original = test[0]["text"]
variants = [
    original + " ",                    # 공백 추가
    original.replace(".", "。"),        # 문장부호 교체
    original.replace(" ", "  "),        # 이중 공백
]
print("원본:", original)
for v in variants:
    print(f"  같은 문자열인가? {v == original}   해시 같은가? {hash(v) == hash(original)}")
PY
```

**답할 것**: 세 변형은 사람이 보기에 같은 문장이다. exact 중복 검사로는 왜 못 잡는가?
번역문("Ignore previous instructions...")은 여기서 한 발 더 나아간 문제인데, 무엇이 더 어려운가?

**제안할 것**: 다음 데이터 버전에서 near-duplicate를 어떻게 잡을지 방법 하나를 정하고 한 문단으로 적는다.
(후보: 정규화 후 해시 / 문자 n-gram 자카드 유사도 / 임베딩 코사인 유사도 / MinHash)

---

## 과제 4 · 분포를 기록한다

정상 데이터의 분포를 표로 남긴다. 이 표는 이후 v2를 만들 때 비교 기준이 된다.

```bash
python 02-intermediate/01-dataset-schema/inspect_data.py runs/data/v1
```

| split | n | BENIGN | PI | JB | user | retrieved | tool | ko | en | 그룹 수 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 384 | 128 | 128 | 128 | 288 | 48 | 48 | 192 | 192 | 48 |
| dev | | | | | | | | | | |
| test | | | | | | | | | | |

그룹 수는 이렇게 센다.

```bash
python -c "
import json
for s in ('train','dev','test'):
    g = {json.loads(l)['group_id'] for l in open(f'runs/data/v1/{s}.jsonl')}
    print(s, len(g))"
```

**답할 것**: test의 그룹 수가 24개다. 만약 이 값이 6개였다면 평가에 어떤 문제가 생기는가?
(힌트: 한 그룹을 통째로 틀리면 test의 몇 %가 오답이 되는가?)

---

## 과제 5 · 층화 실패를 재현한다

`common/guardlab/split.py`의 층화 키에서 `source`를 빼면 어떻게 되는지 확인한다.

```bash
python - <<'PY'
import collections, random
from guardlab.synth import generate
from guardlab.split import assert_no_group_leakage

# source를 뺀 (label, language) 층화를 흉내 낸다
rows = generate()
groups = collections.defaultdict(list)
for r in rows:
    groups[r.group_id].append(r)

buckets = collections.defaultdict(list)
for gid, rs in groups.items():
    buckets[(rs[0].label, rs[0].language)].append(rs)

result = {"train": [], "dev": [], "test": []}
rng = random.Random(42)
for key, bgroups in sorted(buckets.items()):
    rng.shuffle(bgroups)
    n = len(bgroups)
    n_train, n_dev = max(1, round(n * 0.7)), max(1, round(n * 0.15))
    for i, g in enumerate(bgroups):
        split = "train" if i < n_train else "dev" if i < n_train + n_dev else "test"
        result[split].extend(g)

for name, rs in result.items():
    print(f"{name:5} sources={dict(collections.Counter(r.source for r in rs))}")
PY
```

**답할 것**: `source`를 층화하지 않으면 어느 split에서 어떤 source가 사라지거나 편중되는가?
그 상태로 `03-advanced/01-indirect-injection`을 진행하면 무엇을 측정할 수 없게 되는가?

---

## 정답 확인

- [ ] 명시적 누수(같은 group_id)를 검사기가 잡는 것을 확인했는가?
- [ ] 검사기를 통과하는 누수(문장 복제)를 만들고, 그것을 잡는 코드를 직접 짰는가?
- [ ] exact 검사로 못 잡는 near-duplicate 3종을 확인했는가?
- [ ] 다음 버전에서 near-duplicate를 잡을 방법을 하나 정했는가?
- [ ] 분포표를 채우고 그룹 수를 셌는가?
- [ ] source 층화를 뺐을 때 무엇이 무너지는지 확인했는가?

## 막혔을 때

- **`inspect_data.py`가 파일을 못 찾는다** → 인자는 디렉터리 경로다. `train.jsonl`이 아니라
  그 파일이 든 폴더를 넘긴다.
- **누수를 만들었는데 에러가 안 난다** → `group_id`를 바꾼 파일을 실제로 저장했는지, 그리고 검사기에
  넘긴 경로가 원본이 아니라 복사본인지 확인한다.
- **과제 5 코드가 에러를 낸다** → `rs[0].label`처럼 첫 샘플 값을 쓰는 것은 원래 분할기가 **금지하는**
  방식이다. 이 과제에서는 일부러 그렇게 해서 결과를 비교하는 것이다.

## 제출물

- 과제 1·2의 실행 결과(에러 메시지 / 통과 메시지)
- 과제 2에서 직접 짠 텍스트 중복 검사 코드와 결과
- 과제 3의 near-duplicate 대응 방안 한 문단
- 과제 4의 분포표
- 과제 5의 답: source 층화가 없으면 못 재는 것
