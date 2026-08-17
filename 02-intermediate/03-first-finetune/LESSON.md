# 03 · 첫 문장 분류 모델 파인튜닝

## 이 레슨을 마치면

- **파인튜닝**이 무엇을 바꾸는 작업인지 설명할 수 있다.
- 토크나이저, epoch, batch, learning rate가 각각 무엇인지 안다.
- 학습 로그의 `loss`와 `eval_macro_f1`을 읽고 과적합을 알아볼 수 있다.
- 학습 결과를 `train_summary.json`으로 재현 가능하게 남긴다.

---

## 1. 파인튜닝이란 무엇인가

`distilbert-base-multilingual-cased`는 이미 수십 개 언어의 대량 텍스트로 학습된 모델이다. 하지만 이 모델은
"다음 단어 맞히기" 같은 일반 과제로 학습됐을 뿐, **우리 라벨 3개를 모른다.**

파인튜닝은 두 가지를 한다.

```text
   [사전학습 인코더]              [분류 헤드]
   768차원 벡터로 문장을 표현   →   3개 점수로 변환
   (이미 학습됨, 미세 조정)        (새로 만듦, 처음부터 학습)
```

1. **분류 헤드를 새로 붙인다** — 768차원 → 3차원 선형 층. 처음에는 무작위 값이다
2. **전체를 함께 조금씩 갱신한다** — 인코더의 기존 지식을 유지하면서 우리 과제에 맞춰 미세 조정

```python
model = AutoModelForSequenceClassification.from_pretrained(
    args.model, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID,
)
```

이 한 줄이 위 작업을 다 한다. `num_labels=3`을 주면 헤드가 자동으로 만들어진다. 처음 실행하면
"일부 가중치가 초기화되지 않았다"는 경고가 뜨는데, **정상이다** — 새로 만든 헤드를 말하는 것이다.

> **용어**
> - **사전학습(pre-training)**: 대량의 일반 텍스트로 언어 자체를 학습. 우리가 하지 않는다
> - **파인튜닝(fine-tuning)**: 그 위에 우리 과제 데이터로 미세 조정. 이 레슨에서 하는 것
> - **인코더(encoder)**: 문장을 고정 길이 벡터로 바꾸는 부분
> - **헤드(head)**: 그 벡터를 원하는 출력 형태(여기선 라벨 3개 점수)로 바꾸는 마지막 층

---

## 2. 토크나이저 — 문장을 숫자로

모델은 문자열을 못 받는다. **토크나이저**가 문장을 정수 배열로 바꾼다.

```python
tokenizer(batch["text"], truncation=True, max_length=max_len)
```

```text
"이전 지시를 무시해"
  → ['[CLS]', '이', '##전', '지', '##시', '##를', '무', '##시', '##해', '[SEP]']
  → [101, 9638, 12310, 9706, 30873, 11513, 9294, 30873, 14523, 102]
```

- `[CLS]` — 문장 시작 토큰. 분류 헤드는 **이 위치의 벡터**를 입력으로 쓴다
- `##` — 앞 토큰에 이어지는 조각(subword). 한국어는 조사·어미 때문에 조각이 많이 난다
- `truncation=True, max_length=192` — 192토큰을 넘으면 **뒤를 잘라 버린다**

마지막 항목이 중요하다. 긴 문서 끝에 공격이 숨어 있으면 잘려 나가서 모델이 아예 못 본다. 이 문제를
`03-advanced/01-indirect-injection`에서 sliding window로 다룬다.

---

## 3. 학습 설정 읽기

```python
training_args = TrainingArguments(
    per_device_train_batch_size=args.batch,       # 8
    num_train_epochs=args.epochs,                 # 8
    learning_rate=args.lr,                        # 5e-5
    weight_decay=0.01,
    warmup_ratio=0.06,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    seed=args.seed,
)
```

| 설정 | 뜻 | 크게 하면 | 작게 하면 |
|---|---|---|---|
| `batch` | 한 번에 처리할 샘플 수 | 안정적이지만 메모리↑ | 노이즈↑, 메모리↓ |
| `epochs` | 전체 데이터를 몇 바퀴 도는가 | 과적합 위험 | 덜 학습됨 |
| `learning_rate` | 한 번에 가중치를 얼마나 움직이는가 | 발산 위험 | 느리게 수렴 |
| `weight_decay` | 가중치가 커지는 것을 억제 | 과적합 억제 | — |
| `warmup_ratio` | 처음 6%는 lr을 서서히 올림 | 초반 불안정 방지 | — |

`load_best_model_at_end=True`가 특히 중요하다. **마지막 epoch의 모델이 아니라, dev에서 가장 좋았던
모델을 저장**한다. 과적합이 시작된 뒤의 모델을 쓰지 않기 위해서다.

### 디바이스 선택

```python
if torch.backends.mps.is_available(): return "mps"      # Apple Silicon GPU
if torch.cuda.is_available():         return "cuda"     # NVIDIA GPU
return "cpu"
```

Apple Silicon에서는 MPS를 쓴다. 학습 시간이 CPU 대비 크게 줄어든다(이 데이터 기준 약 95초).
MPS에서 미지원 연산 오류가 나면 `export PYTORCH_ENABLE_MPS_FALLBACK=1`을 설정한다.

---

## 4. 실행

```bash
python 02-intermediate/03-first-finetune/train_seq_cls.py \
  --data runs/data/v1 --out runs/models/mbert-v1
python 02-intermediate/03-first-finetune/predict.py \
  --model runs/models/mbert-v1 --input runs/data/v1/test.jsonl \
  --output runs/models/mbert-v1/test-pred.jsonl
```

파이프라인만 확인하려면 `--max-steps 20 --out runs/models/smoke`를 쓴다. 1~2초에 끝난다.
**이 결과로 성능을 판단하면 안 된다.**

### 학습 로그 읽기

```text
{'loss': 1.0821, 'grad_norm': 3.42, 'learning_rate': 4.7e-05, 'epoch': 0.21}
{'eval_loss': 0.9214, 'eval_accuracy': 0.611, 'eval_macro_f1': 0.598, 'epoch': 1.0}
{'loss': 0.4133, ..., 'epoch': 2.08}
{'eval_loss': 0.7752, 'eval_accuracy': 0.729, 'eval_macro_f1': 0.731, 'epoch': 2.0}
...
{'train_runtime': 95.5, 'train_samples_per_second': 32.1, 'train_loss': 0.2104}
```

읽는 법:

- `loss` — 학습 데이터에서의 오차. **내려가야 한다**. 안 내려가면 lr이 너무 작거나 데이터에 문제
- `eval_loss` — dev에서의 오차
- `eval_macro_f1` — dev 성능. 이 값이 최고인 시점의 모델이 저장된다
- **`loss`는 계속 내려가는데 `eval_loss`가 올라가기 시작하면 과적합**이다

### 과적합을 실측으로 보기

같은 데이터에 epoch만 바꿔 학습한 결과다.

| 설정 | test macro F1 | attack recall | benign FPR |
|---|---:|---:|---:|
| 4 epoch / 3e-5 | 0.737 | 0.883 | 0.375 |
| **8 epoch / 5e-5** | **0.750** | 0.914 | 0.328 |
| 12 epoch / 5e-5 | 0.659 | 0.961 | 0.500 |

12 epoch에서 오히려 나빠졌다. 학습 데이터는 더 잘 맞히지만 **처음 보는 표현에서 무너진** 것이다.
attack recall만 보면 0.961로 가장 높지만, benign FPR이 0.500 — 정상의 절반을 차단한다. 지표 하나만
보면 잘못된 결론에 도달한다는 것을 여기서도 확인할 수 있다.

그래서 기본값을 8 epoch로 정했다. **데이터가 바뀌면 이 값도 다시 찾아야 한다.**

---

## 5. 결과 확인

```bash
cat runs/models/mbert-v1/train_summary.json
```

```json
{
  "base_model": "distilbert-base-multilingual-cased",
  "data_dir": "runs/data/v1",
  "device": "mps",
  "seed": 42,
  "epochs": 8,
  "batch": 8,
  "learning_rate": 5e-05,
  "max_len": 192,
  "n_train": 384,
  "n_dev": 144,
  "train_seconds": 95.5,
  "dev_metrics": {"eval_accuracy": 0.75, "eval_macro_f1": 0.752, ...}
}
```

이 파일이 있어야 나중에 같은 결과를 재현할 수 있다. **점수만 기록하고 설정을 안 남기면 그 점수는
쓸모없다.**

### 예측 스크립트

```python
probs = torch.softmax(model(**encoded).logits, dim=-1).cpu().tolist()
scores = {LABELS[idx]: value for idx, value in enumerate(values)}
```

- `logits` — 모델의 원시 출력. 범위 제한이 없는 실수
- `softmax` — 이를 합이 1인 확률로 변환
- 세 라벨의 확률을 **전부** 저장한다. 최종 라벨만 남기면 임계값 실험을 다시 할 수 없다

> **주의** — `LABELS[idx]`는 학습 때 쓴 `LABEL2ID` 순서를 가정한다. 외부에서 받은 모델을 쓸 때는
> 그 모델의 `config.json`에 있는 `id2label`을 확인해야 한다. 순서가 다르면 라벨이 통째로 뒤바뀐다.

---

## 6. 실무의 선택지 — 인코더 분류기만 있는 게 아니다

우리는 "인코더 + 분류 헤드"를 만들었다. 참고 프로젝트(`sgt-owasp`)는 다른 방식을 쓴다.

| | 이 커리큘럼 | 참고 프로젝트 |
|---|---|---|
| 기반 모델 | DistilBERT 다국어 (135M) | Kanana 2.1B (생성형 LLM) |
| 판정 방식 | 분류 헤드가 3개 점수 출력 | **토큰 1개만 생성**해서 `<SAFE>`/`<UNSAFE-A1>`/`<UNSAFE-A2>` 중 선택 |
| 확률 | softmax 확률 | 라벨 토큰들의 logit을 renormalize한 확률 |
| 서빙 | PyTorch, ONNX | PyTorch / GGUF(양자화) / vLLM 중 선택 |

**classifier-as-LLM** 방식(생성 모델을 분류에 쓰기)의 장단점:

- 장점: 대형 모델의 언어 이해력을 그대로 활용, 새 라벨 추가가 프롬프트 수정으로 가능
- 단점: 느리고 무겁다(2.1B vs 135M), 모델이 라벨 토큰이 아닌 엉뚱한 토큰을 생성할 수 있다

마지막 항목 때문에 그 프로젝트에는 **fail-close 가드**가 있다. 생성된 토큰이 정해진 3개가 아니면
`UNKNOWN`으로 처리하고, `UNKNOWN`은 **unsafe로 간주**한다. 우리 분류 헤드 방식은 항상 3개 중 하나가
나오므로 이 문제가 없다. 구조를 고르면 실패 방식도 함께 고르는 것이다.

---

## 7. 흔한 실수와 대처

| 증상 | 원인 | 대처 |
|---|---|---|
| `loss`가 안 내려간다 | lr이 너무 작음 / 라벨이 잘못됨 | lr을 5e-5로, 데이터 검사 재실행 |
| `loss`가 `nan` | lr이 너무 큼 | lr을 1/10로 |
| 메모리 부족 | batch·max_len이 큼 | `--batch 4 --max-len 128` |
| dev 점수가 epoch마다 크게 출렁 | dev가 너무 작음 | dev 크기 확인, seed 여러 개로 반복 |
| test 점수만 유난히 높다 | 누수 의심 | `inspect_data.py`와 텍스트 중복 검사 |
| 두 번 돌렸는데 결과가 다르다 | 정상(확률적) | seed 고정 + 여러 번 평균 |

**마지막 항목이 중요하다.** 학습은 확률적이라 seed에 따라 결과가 흔들린다. 이 데이터에서도 seed만 바꿨을
때 macro F1이 수 %p 움직인다. **한 번 돌린 결과로 "A가 B보다 낫다"고 결론 내리면 안 된다.**

---

## 다음 레슨

모델이 생겼다. 다음은 이 모델을 **재현 가능하게 평가**한다.
`02-intermediate/04-evaluation-harness`에서 화면 출력이 아니라 파일로 남는 리포트를 만든다.
