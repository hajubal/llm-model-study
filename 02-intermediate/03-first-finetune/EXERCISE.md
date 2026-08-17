# 과제 · 학습을 흔들어 보고 무엇이 노이즈인지 가른다

## 목표

하이퍼파라미터를 하나씩 바꿔가며 **성능 변화가 진짜인지 학습 노이즈인지** 구분하는 방법을 익힌다.

## 선행 조건

```bash
python 02-intermediate/03-first-finetune/train_seq_cls.py \
  --data runs/data/v1 --out runs/models/mbert-v1
python 02-intermediate/03-first-finetune/predict.py \
  --model runs/models/mbert-v1 --input runs/data/v1/test.jsonl \
  --output runs/models/mbert-v1/test-pred.jsonl
```

기준값: test macro F1 **0.750**, attack recall **0.914**, benign FPR **0.328** (8 epoch, lr 5e-5, seed 42)

> 학습 1회에 MPS 기준 약 95초, CPU에서는 훨씬 오래 걸린다. 시간이 부족하면 과제 1·2만 하고 나머지는
> `--max-steps`로 축소해서 흐름만 확인한다.

---

## 과제 1 · 잘리는 샘플 비율을 측정한다

`max_len=192`를 넘는 샘플이 몇 %인지 잰다. 잘리면 뒷부분을 모델이 아예 못 본다.

```bash
python - <<'PY'
from transformers import AutoTokenizer
from guardlab.io import read_jsonl

tok = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
for split in ("train", "dev", "test"):
    rows = read_jsonl(f"runs/data/v1/{split}.jsonl")
    lengths = [len(tok(r.text)["input_ids"]) for r in rows]
    over = sum(1 for n in lengths if n > 192)
    print(f"{split:5} n={len(rows):4} 최대 {max(lengths):4}토큰, 평균 {sum(lengths)/len(lengths):5.1f}, "
          f"192 초과 {over}건 ({over/len(rows):.1%})")
PY
```

**답할 것**:

1. 합성 데이터에서 잘리는 샘플이 몇 %인가?
2. `03-advanced/01`에서 만들 긴 문서(정상 문단 80회 반복)는 몇 토큰쯤 될 것 같은가?
3. `max_len`을 512로 키우면 무엇이 좋아지고 무엇이 나빠지는가?

---

## 과제 2 · seed만 바꿔 편차를 잰다

**아무것도 안 바꾸고 seed만** 3번 바꿔 학습한다. 여기서 나오는 폭이 "노이즈의 크기"다.

```bash
for s in 42 7 2026; do
  python 02-intermediate/03-first-finetune/train_seq_cls.py \
    --data runs/data/v1 --out runs/models/seed-$s --seed $s 2>&1 | tail -1
  python 02-intermediate/03-first-finetune/predict.py \
    --model runs/models/seed-$s --input runs/data/v1/test.jsonl \
    --output runs/models/seed-$s/test-pred.jsonl 2>&1 | tail -1
  echo -n "seed=$s: "
  python 01-beginner/03-evaluation-basics/evaluate.py \
    --gold runs/data/v1/test.jsonl --pred runs/models/seed-$s/test-pred.jsonl 2>&1 | sed -n '3,4p' | tr '\n' ' '
  echo
done
```

| seed | macro F1 | attack recall | benign FPR |
|---:|---:|---:|---:|
| 42 | | | |
| 7 | | | |
| 2026 | | | |
| **최대 − 최소** | | | |

**답할 것**: macro F1의 폭이 얼마인가? 앞으로 어떤 개선을 했을 때 **이 폭보다 작은 차이**가 나면
어떻게 해석해야 하는가?

---

## 과제 3 · 변수 하나만 바꿔 비교

아래 중 **하나만** 골라 바꾼다. 두 개를 동시에 바꾸면 원인을 알 수 없다.

| 후보 | 명령 |
|---|---|
| batch | `--batch 16` (또는 `--batch 4`) |
| learning rate | `--lr 2e-5` (또는 `--lr 1e-4`) |
| max_len | `--max-len 128` |

```bash
python 02-intermediate/03-first-finetune/train_seq_cls.py \
  --data runs/data/v1 --out runs/models/tune-A --batch 16
python 02-intermediate/03-first-finetune/predict.py \
  --model runs/models/tune-A --input runs/data/v1/test.jsonl \
  --output runs/models/tune-A/test-pred.jsonl
python 01-beginner/03-evaluation-basics/evaluate.py \
  --gold runs/data/v1/test.jsonl --pred runs/models/tune-A/test-pred.jsonl
```

| 설정 | macro F1 | attack recall | benign FPR | 학습 시간(초) |
|---|---:|---:|---:|---:|
| 기준 (batch 8) | 0.750 | 0.914 | 0.328 | 95.5 |
| 내가 바꾼 것 | | | | |

학습 시간은 `train_summary.json`의 `train_seconds`에 있다.

**답할 것**: 성능 차이가 과제 2에서 잰 **seed 편차보다 큰가?** 크지 않다면 "개선됐다"고 말할 수 없다.

---

## 과제 4 · 과적합 지점을 찾는다

epoch를 늘려가며 dev/test 성능이 언제 꺾이는지 본다.

```bash
for ep in 4 8 16; do
  python 02-intermediate/03-first-finetune/train_seq_cls.py \
    --data runs/data/v1 --out runs/models/ep-$ep --epochs $ep 2>&1 | tail -1
  python 02-intermediate/03-first-finetune/predict.py \
    --model runs/models/ep-$ep --input runs/data/v1/test.jsonl \
    --output runs/models/ep-$ep/test-pred.jsonl > /dev/null
  echo -n "epochs=$ep  dev="
  python -c "
import json; s=json.load(open('runs/models/ep-$ep/train_summary.json'))
print(f\"{s['dev_metrics'].get('eval_macro_f1',0):.3f}\", end='  test=')"
  python 01-beginner/03-evaluation-basics/evaluate.py \
    --gold runs/data/v1/test.jsonl --pred runs/models/ep-$ep/test-pred.jsonl 2>&1 | sed -n '4p'
done
```

| epochs | dev macro F1 | test macro F1 | test benign FPR |
|---:|---:|---:|---:|
| 4 | | | |
| 8 | | | |
| 16 | | | |

**답할 것**:

1. dev와 test가 같은 지점에서 꺾이는가?
2. `load_best_model_at_end=True` 때문에 저장되는 모델은 마지막 epoch가 아니다. 그런데도 epoch를 늘리면
   결과가 나빠지는 이유는? (힌트: dev도 유한하다)
3. 학습 로그에서 `loss`와 `eval_loss`가 갈라지는 epoch를 찾아 적는다

---

## 과제 5 · 모델 카드 초안

`runs/models/mbert-v1/MODEL_CARD.md`를 만든다. 아래 항목을 반드시 채운다.

```markdown
# 모델 카드 — mbert-v1

## Intended use (의도한 용도)
- ...

## Out-of-scope use (범위 밖 용도)
- 이 모델의 판정을 권한 상승 근거로 사용하는 것
- ...

## Training data
- 출처: guardlab.synth (템플릿 90개, 합성)
- 규모: train 384 / dev 144 / test 192
- seed: 42, n_per_group: 8
- **한계: 합성 템플릿이며 실제 사용자 표현 분포를 대표하지 않는다**

## Metrics
| 데이터 | n | macro F1 | attack recall | benign FPR |
|---|---:|---:|---:|---:|
| 합성 test | 192 | | | |
| 독립 bench | 24 | | | |

## Limitations
- 합성 test 점수는 독립 벤치마크 성능을 보장하지 않는다
- source별 성능 차이: user ___ / retrieved ___ / tool ___
- ...

## 재현 명령
```bash
python 02-intermediate/02-synthetic-data/gen_synth.py --out runs/data/v1
python 02-intermediate/03-first-finetune/train_seq_cls.py --data runs/data/v1 --out runs/models/mbert-v1
```
```

source별 성능은 이렇게 얻는다.

```bash
python 01-beginner/03-evaluation-basics/evaluate.py \
  --gold runs/data/v1/test.jsonl --pred runs/models/mbert-v1/test-pred.jsonl | tail -6
```

---

## 정답 확인

- [ ] 잘리는 샘플 비율을 실측했는가?
- [ ] seed 3개로 **노이즈의 크기**를 먼저 쟀는가?
- [ ] 변수를 **한 번에 하나만** 바꿨는가?
- [ ] 성능 차이를 seed 편차와 비교해 판단했는가?
- [ ] 모델 카드에 "합성 데이터"라는 한계를 명시했는가?
- [ ] 실험이 끝난 뒤 불필요한 모델 폴더를 지웠는가? (하나에 약 520MB)

## 막혔을 때

- **MPS에서 연산 미지원 오류** → `export PYTORCH_ENABLE_MPS_FALLBACK=1` 후 재실행
- **메모리 부족** → `--batch 4 --max-len 128`
- **학습이 너무 느리다** → `--max-steps 30`으로 파이프라인만 확인. 단 성능 비교에는 쓰지 않는다
- **`pin_memory` 경고가 뜬다** → MPS에서는 무시해도 되는 경고다
- **디스크가 부족하다** → `du -sh runs/models/*`로 확인하고 실험용 모델을 지운다

## 제출물

- 과제 1의 토큰 길이 통계
- 과제 2의 seed 편차 표 (**이 표가 이후 모든 비교의 기준선**)
- 과제 3의 단일 변수 비교표 + "개선인가 노이즈인가" 판단
- 과제 4의 epoch별 표와 과적합 지점
- `MODEL_CARD.md`
