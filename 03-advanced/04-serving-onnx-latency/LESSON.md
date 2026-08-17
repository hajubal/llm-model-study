# 04 · ONNX와 지연시간

## 이 레슨을 마치면

- **ONNX**가 무엇이고 왜 서빙에 쓰는지 안다.
- 변환 전후 **판정이 같은지 먼저 검증**해야 하는 이유를 실무 사례로 안다.
- p50/p95 지연시간을 측정하고 올바르게 보고할 수 있다.
- 벤치마크 하나로 운영 처리량을 주장하면 안 되는 이유를 안다.

---

## 1. 왜 변환하는가

학습에 쓴 PyTorch 모델을 그대로 서빙하면 다음이 따라온다.

- Python 런타임과 PyTorch 전체(수 GB)가 필요하다
- 학습용 기능(자동 미분, 옵티마이저)이 추론에는 불필요하다
- 추론 최적화(연산 융합, 상수 폴딩)가 자동으로 되지 않는다

**ONNX(Open Neural Network Exchange)**는 모델을 프레임워크 중립 형식으로 저장하는 표준이다. 저장된
그래프를 ONNX Runtime 같은 전용 엔진이 최적화해서 실행한다.

```text
[PyTorch 모델]  ──export──▶  [model.onnx]  ──▶  [ONNX Runtime]
 학습용, 무겁다              연산 그래프          추론 전용, 최적화됨
```

실무 선택지는 ONNX만 있는 것은 아니다. 참고 프로젝트(`sgt-owasp`)는 모델 종류에 따라 다르게 쓴다.

| 서빙 방식 | 쓰는 곳 | 특징 |
|---|---|---|
| PyTorch(safetensors) | 기본 | 가장 단순, 무겁다 |
| **GGUF(양자화)** | CPU 환경 | 가중치를 8bit 등으로 줄여 메모리·속도 개선 |
| **vLLM API** | GPU 서버 | Continuous Batching으로 다중 요청 동시 처리 |
| ONNX | (PII·독성 탐지 모델에 사용) | 경량 인코더 모델에 적합 |

흥미롭게도 그 프로젝트의 **jailbreak 모델은 ONNX를 쓰지 않는다.** 2.1B 생성 모델이라 GGUF/vLLM이 더
적합하기 때문이다. **모델 구조에 따라 서빙 방식이 달라진다.** 우리처럼 작은 인코더 분류기는 ONNX가 잘 맞는다.

---

## 2. 변환 코드 읽기

```python
class LogitsOnly(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits
```

왜 래퍼가 필요한가? HuggingFace 모델은 `SequenceClassifierOutput`이라는 **객체**를 반환한다. ONNX는
텐서만 다룰 수 있으므로, `logits` 텐서만 꺼내는 얇은 래퍼를 씌운다.

```python
torch.onnx.export(
    LogitsOnly(model), (example["input_ids"], example["attention_mask"]), str(target),
    input_names=["input_ids", "attention_mask"], output_names=["logits"],
    dynamic_axes={
        "input_ids":      {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "logits":         {0: "batch"},
    },
    opset_version=17,
)
```

- **`example`** — 변환은 예시 입력을 한 번 흘려보내며 그래프를 기록하는 방식(tracing)이다
- **`dynamic_axes`** — 이 축은 실행 시 크기가 바뀔 수 있다고 선언한다. 이걸 빼면 **예시와 똑같은
  길이의 입력만** 처리하는 모델이 된다
- **`opset_version`** — ONNX 연산자 집합 버전. 런타임이 지원하는 범위여야 한다

토크나이저는 ONNX에 들어가지 않는다. **전처리는 여전히 Python에서** 해야 하므로 별도로 저장한다.

```python
tokenizer_dir = target.parent / f"{target.stem}-tokenizer"
tokenizer.save_pretrained(tokenizer_dir)
```

> 학습된 모델 폴더에 그대로 덮어쓰지 않도록 별도 폴더에 둔다. 서빙용 산출물과 학습 산출물을 섞으면
> 어느 것이 원본인지 알 수 없게 된다.

---

## 3. 순서가 중요하다 — 먼저 동등성, 그 다음 속도

**속도부터 재면 안 된다.** 변환된 모델이 원본과 같은 판정을 하는지 먼저 확인한다.

참고 프로젝트에 실제로 있었던 일이다.

> fp32(모델 레벨)에서는 SAFE로 판정되던 문장이, fp16으로 서빙하니 부동소수점 반올림 때문에 argmax가
> 뒤집혀 UNSAFE로 나왔다.

그래서 그 프로젝트는 `flip_vs_model_level`이라는 지표를 상시 측정한다. 모델 레벨 결과와 서빙 레벨
결과를 비교해 **뒤집힌 비율**과 **뒤집힌 방향**을 나눠 본다.

| 방향 | 의미 | 위험도 |
|---|---|---|
| `to_block` | SAFE였는데 서빙에서 차단됨 | 오탐 증가 — 사용자 불편 |
| `to_safe` | UNSAFE였는데 서빙에서 통과됨 | **미탐 증가 — 보안 구멍** |

같은 "뒤집힘"이라도 방향에 따라 심각도가 다르다. 그래서 하나의 숫자로 뭉치지 않는다.

우리 커리큘럼도 같은 검증을 한다. 과제 1이 그것이다.

---

## 4. 실행

```bash
python 03-advanced/04-serving-onnx-latency/export_onnx.py \
  --model runs/models/mbert-v1 --out runs/models/mbert-v1/model.onnx
python 03-advanced/04-serving-onnx-latency/bench_latency.py \
  --model runs/models/mbert-v1 --onnx runs/models/mbert-v1/model.onnx \
  --runs 30 --out runs/latency.json
```

실측 결과:

```json
{
  "runs": 30,
  "batch": 1,
  "torch_cpu": {"p50_ms": 17.315, "p95_ms": 18.267},
  "onnx_cpu":  {"p50_ms": 3.344,  "p95_ms": 3.468}
}
```

**약 5배 빨라졌다.** 다만 이 숫자를 그대로 인용하면 안 되는 이유가 여럿 있다(6절 참고).

### 벤치마크 코드에서 볼 것

```python
for _ in range(5):                      # 워밍업
    with torch.inference_mode():
        model(**encoded)
    session.run(None, {...})
```

**워밍업(warm-up)**을 먼저 돌린다. 첫 호출은 메모리 할당, 커널 컴파일, 캐시 미스 때문에 유난히 느리다.
이를 측정에 포함하면 정상 상태 성능을 과소평가한다.

```python
def stats(values):
    return {"p50_ms": round(float(np.percentile(values, 50)), 3),
            "p95_ms": round(float(np.percentile(values, 95)), 3)}
```

**평균이 아니라 백분위수**를 쓴다.

| 지표 | 의미 | 왜 |
|---|---|---|
| p50 (중앙값) | 절반의 요청이 이보다 빠르다 | 전형적 경험 |
| **p95** | 95%의 요청이 이보다 빠르다 | **꼬리 지연** — 실제 사용자 불만의 원인 |
| 평균 | — | 이상치 하나에 크게 흔들려 오해를 부른다 |

SLA는 보통 p95나 p99로 정한다. "평균 10ms"인 시스템이 "p99 2초"일 수 있다.

---

## 5. 전처리 시간도 지연시간이다

`bench_latency.py`는 토크나이징을 **루프 밖에서** 한 번만 한다. 즉 **모델 추론 시간만** 잰다.

실제 서비스의 지연시간은 이렇다.

```text
전체 = 토크나이징 + 추론 + 후처리(softmax, 임계값) + 정책 결정 + 로깅
```

문서가 길어 sliding window를 쓰면 추론이 window 수만큼 반복된다. **문서 단위 지연시간**은 문장 단위와
전혀 다르다.

```text
문장 1건:     3.3 ms
문서 20 window: 3.3 × 20 = 66 ms + 토크나이징
```

`03-advanced/01`의 sliding window를 실제로 쓴다면 이 계산을 반드시 해야 한다.

---

## 6. 이 숫자로 주장하면 안 되는 것

측정한 것은 **"문장 1개, batch 1, CPU, 워밍업 후, 30회 반복"**이다. 이것으로 다음을 주장할 수 없다.

| 주장 | 왜 안 되는가 |
|---|---|
| "우리 서비스는 3ms에 응답한다" | 전처리·네트워크·정책이 빠져 있다 |
| "초당 300건 처리 가능" | 동시 요청 시 CPU 경합이 생긴다 |
| "긴 문서도 3ms" | window 수만큼 곱해진다 |
| "GPU면 더 빠르다" | batch 1에서는 GPU가 더 느릴 수도 있다 |

올바른 보고 방식은 조건을 명시하는 것이다.

```text
p50 3.3ms / p95 3.5ms
(입력: 한국어 단문 1건, batch 1, ONNX Runtime CPU, Apple Silicon,
 워밍업 5회 후 30회 측정, 토크나이징 제외)
```

참고 프로젝트도 지연시간에 대한 명시적 SLA 수치를 코드에 두지 않는 대신, **동시 요청 상한**(백프레셔
미들웨어, 기본 100)과 **Circuit Breaker**로 시스템을 보호한다. 지연시간을 낙관적으로 약속하기보다
과부하 시 동작을 정의해 두는 편이 안전하다.

---

## 7. 흔한 실수

| 실수 | 결과 | 대신 |
|---|---|---|
| 동등성 검증 없이 속도만 잰다 | 판정이 달라진 채 배포 | 라벨 일치율 먼저 |
| 워밍업 없이 측정 | 첫 호출 때문에 느리게 나옴 | 워밍업 후 측정 |
| 평균만 보고 | 꼬리 지연이 숨는다 | p50·p95 함께 |
| `dynamic_axes` 누락 | 특정 길이만 처리 가능 | batch·sequence를 동적으로 |
| 문장 벤치로 문서 성능 주장 | window 수만큼 차이 | 문서 단위로 재측정 |
| 토크나이저를 모델 폴더에 덮어씀 | 학습 산출물 오염 | 별도 폴더 |

---

## 다음 레슨

성능·정책·지연시간을 모두 쟀다. 마지막은 이것을 **하나의 보고서로 묶는 일**이다.
`03-advanced/05-benchmark-and-report`에서 무엇을 쓰고 무엇을 쓰면 안 되는지 다룬다.
