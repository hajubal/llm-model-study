# 06 · 미니 프로젝트 — 개선 루프 한 바퀴

## 이 프로젝트의 목표

지금까지 배운 것을 **한 바퀴 연결**한다. 측정 → 오류 분석 → 가설 → 한 가지 변경 → 재측정 → 기록.

통과 기준은 **높은 숫자가 아니다.** 다음 두 가지다.

1. 변경 가설과 결과가 논리적으로 연결되는가
2. test를 보며 반복 튜닝하지 않았는가

---

## 왜 "한 가지만" 바꾸는가

두 가지를 동시에 바꾸면 이런 상황이 된다.

```text
v1: macro F1 0.637
v2: macro F1 0.667   (데이터 추가 + threshold 조정)
```

0.03이 올랐는데, **데이터 덕분인지 threshold 덕분인지 알 수 없다.** 심지어 데이터는 -0.02, threshold는
+0.05였을 수도 있다. 그러면 다음 실험에서 잘못된 방향으로 간다.

과학적 방법의 기본이다. 변수를 하나만 움직인다.

---

## 절차

### 1단계 · v1을 고정 벤치마크에 평가한다

```bash
python 02-intermediate/03-first-finetune/predict.py \
  --model runs/models/mbert-v1 --input common/data/bench/gold.jsonl \
  --output runs/models/mbert-v1/bench-pred.jsonl
python 02-intermediate/04-evaluation-harness/evaluate.py \
  --gold common/data/bench/gold.jsonl --pred runs/models/mbert-v1/bench-pred.jsonl \
  --out runs/models/mbert-v1/bench-report
```

기준값 예시(실측, seed 42): macro F1 **0.717** [0.500–0.879], attack recall **0.938**, benign FPR **0.375**

> **비교 전에 seed 편차부터 본다.** 이 데이터에서 macro F1의 seed 폭은 **0.108**이다
> (`02-intermediate/03` 과제 2). 0.03짜리 개선은 노이즈와 구분되지 않는다.

> 벤치마크는 **학습에 쓰지 않은 독립 데이터**여야 한다. `common/data/bench/gold.jsonl`을 절대 train에
> 넣지 않는다.

### 2단계 · 가장 큰 오류군을 고른다

```bash
python 02-intermediate/05-error-analysis/error_dump.py \
  --gold runs/data/v1/dev.jsonl --pred runs/models/mbert-v1/dev-pred.jsonl
```

`05-error-analysis` 과제 5에서 만든 개선 제안서를 그대로 쓴다.

### 3단계 · 한 가지만 바꿔 v2를 만든다

선택지 세 가지 중 **하나**를 고른다.

| 선택 | 하는 일 | 명령 |
|---|---|---|
| **A. 데이터 보강** | 부족한 slice의 템플릿 추가 | `synth.py` 수정 후 `gen_synth.py --out runs/data/v2` |
| **B. 라벨 수정** | 잘못 라벨링된 샘플 정정 | 데이터 파일 직접 수정 + 근거 기록 |
| **C. threshold 조정** | 학습 없이 판정 기준만 변경 | dev 스윕으로 새 값 선택 |

C를 고르면 재학습이 필요 없다. 가장 빠른 실험이므로 먼저 해 보는 것도 좋다.

**A를 고른 경우:**

```bash
# synth.py에 템플릿 추가 후
python -m pytest common/tests -q
python 02-intermediate/02-synthetic-data/gen_synth.py --out runs/data/v2
python 02-intermediate/01-dataset-schema/inspect_data.py runs/data/v2
python 02-intermediate/03-first-finetune/train_seq_cls.py \
  --data runs/data/v2 --out runs/models/mbert-v2
python 02-intermediate/03-first-finetune/predict.py \
  --model runs/models/mbert-v2 --input common/data/bench/gold.jsonl \
  --output runs/models/mbert-v2/bench-pred.jsonl
python 02-intermediate/04-evaluation-harness/evaluate.py \
  --gold common/data/bench/gold.jsonl --pred runs/models/mbert-v2/bench-pred.jsonl \
  --out runs/models/mbert-v2/bench-report
```

### 4단계 · 같은 채점기로 비교한다

```bash
python 02-intermediate/06-mini-project/compare_runs.py \
  runs/models/mbert-v1/bench-report/report.json \
  runs/models/mbert-v2/bench-report/report.json
```

```text
| run | macro F1 | attack recall | benign FPR | n |
|---|---:|---:|---:|---:|
| runs/models/mbert-v1/bench-report | 0.717 [0.500–0.879] | 0.938 | 0.375 | 24 |
| runs/models/mbert-v2/bench-report | ... | ... | ... | 24 |
```

### 5단계 · 좋아진 것과 나빠진 것을 **모두** 기록한다

전체 점수뿐 아니라 slice별로도 본다.

```bash
python -c "
import json
for tag in ('mbert-v1', 'mbert-v2'):
    r = json.load(open(f'runs/models/{tag}/bench-report/report.json'))
    print(f'--- {tag} ---')
    for field, slices in r['slices'].items():
        for value, s in sorted(slices.items()):
            print(f\"  {field}={value:12} n={s['n_samples']:3} macro_f1={s['macro_f1']:.3f} fpr={s['benign_fpr']:.3f}\")
"
```

**개선된 slice만 보고하고 악화된 slice를 빼면 그것은 보고서가 아니라 광고다.**

---

## 회고 작성

`runs/models/mbert-v2/RETRO.md`에 한 페이지로 적는다.

```markdown
# v1 → v2 회고

## 무엇을 바꿨는가
- 변경: (한 문장)
- 바꾸지 **않은** 것: (모델 구조, 하이퍼파라미터, 평가 데이터 등)

## 왜 그것을 바꿨는가
- 관찰: dev 오류 __건 중 __건이 ___ 유형
- 가설: ___

## 결과

| 지표 | v1 | v2 | 변화 |
|---|---:|---:|---:|
| macro F1 | | | |
| attack recall | | | |
| benign FPR | | | |

### slice별

| slice | v1 | v2 | 판정 |
|---|---:|---:|---|
| source=user | | | |
| source=retrieved | | | |
| source=tool | | | |

## 가설은 맞았는가
- (맞았다 / 부분적으로 / 틀렸다) + 근거

## 이 차이가 노이즈가 아니라는 근거
- seed 편차: ±___ (02-intermediate/03 과제 2에서 측정)
- 이번 변화폭: ___
- 판단: ___

## 다음에 검증할 것 하나
- ___
```

**"가설이 틀렸다"도 완전히 유효한 결과다.** 틀렸다는 것을 확인한 것 자체가 정보다. 숫자를 올리기 위해
가설을 사후에 바꿔 쓰는 것이 진짜 실패다.

---

## 실무의 개선 루프는 선형이 아니다

참고 프로젝트(`sgt-owasp`)의 의사결정 기록 문서에는 **폐기된 규칙들과 사후 정정 기록**이 그대로 남아 있다.
최초 결론이 반증되어 뒤집힌 문서도 있다.

배울 점은 이것이다.

- 가설 → 반증 → 정정의 기록을 **지우지 않고 남긴다**
- 그래야 같은 실수를 반복하지 않고, 왜 지금 규칙이 이런 모양인지 설명할 수 있다
- 성공한 실험만 기록하면 "왜 이렇게 안 했나?"라는 질문에 답할 수 없다

`JOURNAL.md`에 매 레슨 세 줄씩 남기라고 한 이유가 이것이다.

---

## 제출물

- [ ] `runs/data/v2/manifest.json` (A를 골랐다면)
- [ ] `runs/models/mbert-v2/train_summary.json` (A·B를 골랐다면)
- [ ] v1 / v2의 `report.json`, `errors.jsonl`
- [ ] `compare_runs.py`가 만든 비교표
- [ ] slice별 비교표 (개선·악화 **모두**)
- [ ] `RETRO.md` (한 페이지)

## 자가 점검

- [ ] 한 번에 **하나의 변수**만 바꿨는가?
- [ ] v1과 v2를 **같은 벤치마크, 같은 채점기**로 비교했는가?
- [ ] test(또는 bench)를 보며 여러 번 조정하지 않았는가?
- [ ] 악화된 지표도 표에 적었는가?
- [ ] 변화폭이 seed 편차보다 큰지 확인했는가?
- [ ] 다음에 검증할 것을 하나 정했는가?
