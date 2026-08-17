# 중급 · 예시 답안

**먼저 스스로 답한 뒤에 열 것.** 중급의 수료 기준은 숫자가 아니라 절차다. 아래 답이
설명하는 것도 "무엇이 정답인가"가 아니라 **"어떤 절차를 거쳤는가"**다.

---

## 수료 기준 1 · 누수를 두 종류 이상 설명한다

세 종류를 직접 만들어 보는 것이 가장 빠르다.

```bash
python 02-intermediate/01-dataset-schema/solution/make_leak.py \
  --source runs/data/v1 --out runs/leak-group --mode group   # 잡힌다
python 02-intermediate/01-dataset-schema/solution/make_leak.py \
  --source runs/data/v1 --out runs/leak-dup --mode dup       # 통과한다
python 02-intermediate/01-dataset-schema/solution/make_leak.py \
  --source runs/data/v1 --out runs/leak-near --mode near     # 통과한다
```

| 종류 | 무엇인가 | 검사기가 잡는가 | 왜 위험한가 |
|---|---|---|---|
| **group 누수** | 같은 템플릿의 파생 문장이 train과 test에 갈라져 들어감 | **예** — `group_id` 겹침 | 모델이 템플릿을 외우고 test에서 점수를 받는다 |
| **텍스트 복제** | 완전히 같은 문장이 다른 id/group_id로 양쪽에 존재 | 아니오 | test 점수가 순수 암기다 |
| **near-duplicate** | 공백·구두점·어미만 다른 문장 | 아니오 | 위와 같지만 눈으로도 안 보인다 |

**검사기 통과 = 누수 없음이 아니다.** `inspect_data.py`는 id 중복과 `group_id` 겹침만 본다.
텍스트가 같은지, 비슷한지는 보지 않는다.

**near-duplicate가 더 어려운 이유**(과제 3): '거의 같다'의 기준을 정해야 한다.
정규화 후 완전 일치? 편집 거리 N 이하? 임베딩 코사인 유사도 0.95 이상? 기준마다 잡는
것이 다르다. 느슨하면 정상 데이터를 지우고, 빡빡하면 누수를 놓친다. **정답이 하나가
아니라서** 어렵다.

실무 처방: 정규화(소문자화 + 공백 압축 + NFKC) 후 해시로 완전 일치를 먼저 지우고,
남은 것을 임베딩으로 근사 중복 검사한다. 그리고 **그 기준을 데이터셋 버전에 기록한다.**

---

## 수료 기준 2 · 개선인지 학습 노이즈인지 판단하는 절차

**답은 "seed를 바꿔 몇 번 돌려 본다"이고, 그 절차는 다음과 같다.**

```bash
python 02-intermediate/03-first-finetune/run_seeds.py \
  --data runs/data/v1 --gold runs/data/v1/test.jsonl \
  --out runs/seeds/v1 --seeds 42 43 44
```

출력의 `최대-최소` 행이 **seed만 바꿔서 생긴 변동 폭**이다. 판단 규칙:

```text
개선폭 <= seed 변동 폭   ->  개선이라고 부를 수 없다
개선폭 >  seed 변동 폭   ->  개선일 가능성이 있다 (증명은 아니다)
```

v2를 만들었다면 바로 비교할 수 있다.

```bash
python 02-intermediate/03-first-finetune/run_seeds.py \
  --data runs/data/v1 --gold runs/data/v1/test.jsonl --out runs/seeds/v1 \
  --seeds 42 43 44 --compare runs/models/mbert-v2/test-report/report.json
```

**두 번째 축은 신뢰구간이다.** `evaluate.py`가 기본으로 bootstrap 95% 구간을 낸다.
두 구간이 크게 겹치면 그 차이는 표본 변동일 수 있다.

```text
seed 편차  : 같은 데이터·같은 설정에서 학습이 얼마나 흔들리는가
신뢰구간   : 이 평가 표본이 얼마나 작은가
```

**둘은 다른 불확실성이고, 둘 다 확인해야 한다.** seed 편차가 작아도 test가 24건이면
숫자는 여전히 못 믿는다.

---

## 수료 기준 3 · dev에서 threshold를 정하고 test를 한 번만 연다

**절차**:

1. dev에 대해 예측을 만든다.
2. `threshold_sweep.py`로 dev에서 attack recall / benign FPR 곡선을 본다.
3. **제품 요구사항**을 먼저 정한다 (예: "benign FPR 5% 이하에서 recall 최대화").
4. 그 조건을 만족하는 threshold를 dev에서 고른다.
5. test에 **그 값 하나만** 적용해 한 번 평가한다. 결과가 마음에 안 들어도 되돌아가지 않는다.

```bash
python 02-intermediate/03-first-finetune/predict.py \
  --model runs/models/mbert-v1 --input runs/data/v1/dev.jsonl \
  --output runs/models/mbert-v1/dev-pred.jsonl
python 02-intermediate/05-error-analysis/threshold_sweep.py \
  --gold runs/data/v1/dev.jsonl --pred runs/models/mbert-v1/dev-pred.jsonl
```

**왜 되돌아가면 안 되는가**: test를 보고 threshold를 바꾸는 순간 test는 dev가 된다.
그 뒤로 test 숫자는 "미지의 데이터에서의 성능"이 아니라 "우리가 맞춘 데이터에서의 성능"이다.
한 번만 열겠다는 규율이 test의 유일한 가치다.

**보정을 함께 확인한다**(threshold 레슨의 빠진 절반):

```bash
python 02-intermediate/05-error-analysis/calibration.py \
  --gold runs/data/v1/dev.jsonl --pred runs/models/mbert-v1/dev-pred.jsonl
```

파인튜닝한 분류기의 점수는 보정된 확률이 아니다. threshold 0.8은 "공격일 확률 80%"가
아니라 그냥 "이 모델에서 그 위치"다. 그래서 **모델을 재학습하면 같은 0.8이 다른 지점을
가리키고, threshold를 dev에서 다시 정해야 한다.**

---

## 수료 기준 4 · 오류를 유형별로 나누고 처방을 하나만 고른다

```bash
python 02-intermediate/05-error-analysis/error_dump.py \
  --gold runs/data/v1/dev.jsonl --pred runs/models/mbert-v1/dev-pred.jsonl
```

오류 유형은 `gold->pred` 쌍으로 나눈다. 대표적인 넷:

| 유형 | 의미 | 대표 처방 |
|---|---|---|
| `BENIGN->PROMPT_INJECTION` | 오탐. 정상을 차단 | 하드 네거티브 템플릿 추가 |
| `PROMPT_INJECTION->BENIGN` | 미탐. 공격을 통과 | 해당 표현의 공격 템플릿 추가 |
| `PROMPT_INJECTION->JAILBREAK` | 라벨 혼동. 둘 다 차단이므로 운영 영향 작음 | 어노테이션 가이드 경계 정리 |
| `JAILBREAK->BENIGN` | 미탐 | 위와 같음 |

**처방을 하나만 고르는 이유**: 데이터와 threshold를 동시에 바꾸면 무엇이 효과를 냈는지
알 수 없다. 다음 실험에서 다시 처음부터 추측하게 된다.

**고르는 기준은 빈도가 아니라 운영 영향이다.** `PROMPT_INJECTION->JAILBREAK` 혼동이
가장 많더라도, 두 라벨 모두 차단 대상이면 사용자에게는 아무 차이가 없다. 반면
`BENIGN->PROMPT_INJECTION`은 건수가 적어도 정상 사용자를 막는다.

---

## 수료 기준 5 · 악화된 slice까지 포함해 기록한다

**나쁜 기록**:

> v2에서 macro F1이 0.75에서 0.79로 올랐다.

**통과하는 기록**:

> v2(하드 네거티브 템플릿 6개 추가, 그 외 동일 · seed 42 · threshold 0.5):
> - test macro F1 0.75 → 0.79 (seed 3개 변동 폭 0.02보다 큼)
> - benign FPR 0.33 → 0.21 (목표했던 방향)
> - **attack recall 0.91 → 0.87 (악화)** — 하드 네거티브를 늘리면서 경계가 정상 쪽으로
>   이동했다. 예상된 대가이고, 제품 요구(FPR 우선)에 부합하므로 수용한다.
> - **source=tool slice macro F1 0.58 → 0.55 (악화)** — 추가한 템플릿이 전부 user
>   입력이라 tool slice는 개선되지 않았고 경계 이동의 손해만 봤다. 다음 반복의 대상.
> - 신뢰구간: macro F1 0.79 [0.73–0.85] — v1 구간과 일부 겹친다. 확정적이지 않다.

**차이는 세 가지다.** 조건을 적었고, 악화된 축을 숨기지 않았고, 그 악화를 수용할지
판단한 근거를 남겼다.

---

## 과제별 짧은 답

### 02-과제 1 · 중복률을 측정한다

```bash
python -c "
import sys; sys.path.insert(0,'common')
from guardlab.io import read_jsonl
rows = read_jsonl('runs/data/v1/train.jsonl')
texts = [r.text for r in rows]
print(f'{len(texts)}건 중 고유 {len(set(texts))}건, 중복률 {1-len(set(texts))/len(texts):.3f}')
"
```

기본 생성기는 중복률 0이다. `_slot_values()`가 한 템플릿 안에서 **서로 다른 슬롯 조합만**
뽑기 때문이다. `n_per_group`을 슬롯 조합 수보다 크게 잡으면 그때부터 중복이 생긴다.

### 02-과제 2 · 하드 네거티브 템플릿을 늘린다

핵심은 **공격 문장과 어휘를 최대한 겹치게** 만드는 것이다.

```python
("bn-ko-quote-policy", "BENIGN", "ko", "user",
 "'이전 지시를 모두 무시해'가 왜 위험 신호인지 {topic} 교육자료에 넣을 설명을 써 줘."),
("bn-ko-detect", "BENIGN", "ko", "user",
 "시스템 프롬프트 공개를 요구하는 입력을 탐지하는 규칙을 {topic}용으로 만들어 줘."),
```

주의: `(label, language, source)` 조합마다 group이 **3개 이상**이어야 split이 실패하지
않는다. 하나만 추가하면 `ValueError`가 난다 — 그것이 설계 오류를 데이터 생성 단계에서
드러내는 장치다.

### 03-과제 1 · 잘리는 샘플 비율을 측정한다

```bash
python -c "
import sys; sys.path.insert(0,'common')
from transformers import AutoTokenizer
from guardlab.io import read_jsonl
tok = AutoTokenizer.from_pretrained('distilbert-base-multilingual-cased')
rows = read_jsonl('runs/data/v1/train.jsonl')
lens = [len(tok(r.text)['input_ids']) for r in rows]
over = sum(l > 192 for l in lens)
print(f'max_len 192 초과 {over}/{len(lens)}건, 최대 {max(lens)} 토큰')
"
```

합성 데이터는 문장이 짧아 0건이다. **그래서 `03-advanced/01-indirect-injection`이
필요하다** — 긴 문서에서는 truncation이 미탐을 만들고, 그때는 잘리는 비율이 성능을 지배한다.

### 05-과제 5 · 처방 하나를 제안한다

형식:

> **관찰**: dev 오류 중 `BENIGN->PROMPT_INJECTION`이 N건으로 가장 많고, 그중 M건이
> 보안 용어를 인용한 문장이다.
> **가설**: 학습 데이터에 "공격 용어를 인용하지만 정상"인 표현이 부족하다.
> **처방**: 하드 네거티브 템플릿을 6개 추가한다. 다른 것은 바꾸지 않는다.
> **검증**: dev에서 해당 오류 유형이 줄고, attack recall이 seed 편차 이상 떨어지지 않는지 본다.
> **예상되는 대가**: 경계가 정상 쪽으로 이동하므로 recall이 조금 내려갈 것이다.

마지막 줄이 중요하다. **대가를 미리 적어 두면 결과를 정직하게 읽게 된다.**
