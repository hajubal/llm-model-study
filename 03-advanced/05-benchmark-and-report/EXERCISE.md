# 과제 · 숫자에 조건을 붙여 보고한다

## 목표

같은 모델의 성능을 **세 축으로 나눠** 재고, 표본 수와 불확실성을 포함해 보고서와 모델 카드를 완성한다.

## 선행 조건

`02-intermediate/04`와 `03-advanced/04`를 마쳐 아래 파일들이 있어야 한다.

```text
runs/models/mbert-v1/test-report/report.json     # 합성 test
runs/models/mbert-v1/bench-report/report.json    # 독립 bench
runs/latency.json                                # 지연시간
```

---

## 과제 1 · 세 축 비교표

```bash
# 하드 네거티브 전용 평가 추가
python 02-intermediate/03-first-finetune/predict.py \
  --model runs/models/mbert-v1 --input common/data/bench/negatives.jsonl \
  --output runs/models/mbert-v1/neg-pred.jsonl
python 02-intermediate/04-evaluation-harness/evaluate.py \
  --gold common/data/bench/negatives.jsonl --pred runs/models/mbert-v1/neg-pred.jsonl \
  --out runs/models/mbert-v1/neg-report

python 02-intermediate/06-mini-project/compare_runs.py \
  runs/models/mbert-v1/test-report/report.json \
  runs/models/mbert-v1/bench-report/report.json \
  runs/models/mbert-v1/neg-report/report.json
```

| 축 | 데이터 | n | macro F1 | attack recall | benign FPR | 이 축이 말해주는 것 |
|---|---|---:|---:|---:|---:|---|
| 학습 분포 안 | 합성 test | 192 | | | | |
| 독립 holdout | bench gold | 24 | | | | |
| 하드 네거티브 | bench negatives | 8 | — | — | | |
| 운영 분포 | (없음) | 0 | — | — | — | 측정하지 않음 |

**답할 것**:

1. 세 축의 benign FPR이 다르다. 어느 것을 "우리 모델의 오탐률"로 보고하겠는가? 왜?
2. 하드 네거티브 세트의 macro F1과 attack recall에 `—`를 쓴 이유는?
3. 마지막 행을 지우지 않고 "측정하지 않음"으로 남기는 이유는?

---

## 과제 2 · 라벨/source/language별 표본 수를 붙인다

지표 옆에 항상 표본 수를 적는다.

```bash
python - <<'PY'
import json
for tag, path in [("합성 test", "test-report"), ("독립 bench", "bench-report")]:
    r = json.load(open(f"runs/models/mbert-v1/{path}/report.json"))
    print(f"=== {tag} (n={r['n_samples']}) ===")
    print("  라벨별:")
    for label, c in r["per_label"].items():
        print(f"    {label:17} support={c['support']:4} P={c['precision']:.3f} R={c['recall']:.3f} F1={c['f1']:.3f}")
    for field, slices in r["slices"].items():
        print(f"  {field}별:")
        for value, s in sorted(slices.items()):
            print(f"    {value:12} n={s['n_samples']:4} macro_f1={s['macro_f1']:.3f} fpr={s['benign_fpr']:.3f}")
PY
```

**답할 것**: 표본이 10건 미만인 slice가 있는가? 그 slice의 수치를 리포트에 쓸 때 어떤 단서를 달겠는가?

---

## 과제 3 · 신뢰구간을 계산한다

```bash
python - <<'PY'
import json, math

def ci95(p, n):
    if n == 0:
        return None
    half = 1.96 * math.sqrt(max(p * (1 - p), 1e-12) / n)
    return max(0.0, p - half), min(1.0, p + half)

for tag, path in [("합성 test", "test-report"), ("독립 bench", "bench-report"),
                  ("하드 네거티브", "neg-report")]:
    r = json.load(open(f"runs/models/mbert-v1/{path}/report.json"))
    benign_n = r["per_label"]["BENIGN"]["support"]
    attack_n = sum(r["per_label"][k]["support"] for k in ("PROMPT_INJECTION", "JAILBREAK"))
    lo, hi = ci95(r["benign_fpr"], benign_n)
    print(f"{tag:14} benign_fpr={r['benign_fpr']:.3f} (n={benign_n:3})  95% CI [{lo:.3f}, {hi:.3f}]")
    if attack_n:
        lo, hi = ci95(r["attack_recall"], attack_n)
        print(f"{'':14} attack_recall={r['attack_recall']:.3f} (n={attack_n:3})  95% CI [{lo:.3f}, {hi:.3f}]")
PY
```

| 축 | benign FPR | 95% CI | 구간 폭 |
|---|---:|---|---:|
| 합성 test | | | |
| 독립 bench | | | |
| 하드 네거티브 | | | |

**답할 것**:

1. 가장 넓은 구간은 어느 축인가? 폭이 몇 %p인가?
2. "독립 bench에서 FPR을 0.250 → 0.125로 개선했다"는 주장이 통계적으로 의미가 있는가?
   두 구간이 겹치는지 확인한다.
3. 구간을 절반으로 줄이려면 표본을 몇 배로 늘려야 하는가? (힌트: 구간 폭은 `1/√n`에 비례)

---

## 과제 4 · 리포트 생성 후 손으로 채우기

```bash
python 03-advanced/05-benchmark-and-report/make_report.py \
  --reports runs/models/mbert-v1/test-report/report.json \
            runs/models/mbert-v1/bench-report/report.json \
            runs/models/mbert-v1/neg-report/report.json \
  --latency runs/latency.json \
  --out runs/report/model-report.md
```

생성된 파일을 열어 **비어 있는 항목을 전부 채운다.**

특히 "알려진 한계"는 일반 문구를 지우고 **이 모델의 구체적 실패**로 바꾼다.

```markdown
## 알려진 한계

### 측정된 실패
- `source=tool` slice: macro F1 ___, benign FPR ___ (n=___).
  도구 출력 형태의 정상 문장을 ___% 오탐한다.
- 대소문자 변형: attack recall ___ → ___ (___%p 하락).
  ASCII 알파벳을 섞어 쓰면 우회 가능하다.
- 긴 문서(단일 truncation): 공격 __건 중 __건 미탐.

### 측정하지 못한 것
- 실제 운영 트래픽 분포에서의 성능
- 한국어/영어 외 언어
- ___

### 구조적 한계
- 탐지 통과는 안전의 증명이 아니다
- 학습 데이터가 합성 템플릿이라 실제 표현 분포와 다르다
```

---

## 과제 5 · 모델 카드 완성

`runs/models/mbert-v1/MODEL_CARD.md`를 완성한다. `02-intermediate/03` 과제 5에서 만든 초안을 확장한다.

**Out-of-scope use를 최소 5개** 적는다. 이 항목이 모델 카드의 핵심이다.

```markdown
## Out-of-scope use

이 모델을 다음 용도로 사용하지 않는다.

1. 판정 결과를 권한 상승·인증의 근거로 사용
2. 도구 allowlist 없이 이 모델만으로 도구 호출 허용 여부 결정
3. `source=tool` 입력의 단독 판정 (측정된 FPR ___)
4. 한국어/영어 외 언어 입력
5. ___
6. ___
```

---

## 과제 6 · 이해관계자용 한 페이지 요약

기술 배경이 없는 사람에게 보고한다고 가정하고 한 페이지를 쓴다. **"탐지율 XX%"만 쓰면 실패다.**

```markdown
# 탐지 모델 v1 — 요약

## 한 줄 결론
이 모델은 (배포 가능 / 조건부 가능 / 불가)하다. 이유: ___

## 무엇을 잡는가
- 측정한 공격 ___건 중 ___건을 탐지 (n=___, 신뢰구간 ___)
- 어떤 데이터에서 잰 것인지: ___

## 무엇을 놓치는가
- ___
- ___

## 정상 사용자에게 미치는 영향
- 정상 요청의 ___%가 차단되거나 검토 대기 상태가 된다
- 예상 검토 건수: 일 ___건

## 이 숫자가 보증하지 않는 것
- ___

## 배포한다면 함께 있어야 하는 것
- ___ (권한 분리, 도구 allowlist 등)

## 다음에 개선할 것 하나
- ___
```

---

## 정답 확인

- [ ] 세 축을 나눠 재고, 측정하지 않은 축을 "없음"으로 남겼는가?
- [ ] 모든 지표에 **표본 수**를 붙였는가?
- [ ] 신뢰구간을 계산하고, 개선 주장이 통계적으로 유효한지 확인했는가?
- [ ] "알려진 한계"를 **구체적 수치와 slice로** 썼는가?
- [ ] 모델 카드의 Out-of-scope use를 5개 이상 적었는가?
- [ ] 이해관계자 요약에 **정상 사용자 영향**과 **보증하지 않는 것**을 넣었는가?

## 막혔을 때

- **`compare_runs.py`가 같은 이름을 보여준다** → run 이름으로 `report.json`의 **부모 폴더명**을 쓴다.
  리포트를 서로 다른 폴더에 저장한다.
- **negatives 리포트의 macro F1이 이상하다** → support 0인 라벨 때문이다. 채점기 출력의 주의 문구를
  확인하고, 이 세트에서는 benign FPR만 읽는다.
- **신뢰구간이 [0, 1]을 벗어난다** → 정규 근사의 한계다. 작은 표본에서는 Wilson 구간이 더 정확하지만,
  이 과제에서는 "구간이 매우 넓다"는 결론만 얻으면 충분하다.

## 제출물

- 과제 1의 세 축 비교표
- 과제 3의 신뢰구간 표 + "개선 주장이 유효한가"에 대한 답
- 완성된 `runs/report/model-report.md`
- 완성된 `MODEL_CARD.md` (Out-of-scope 5개 이상)
- 이해관계자용 한 페이지 요약
