# 04 · ONNX와 지연시간

정확도뿐 아니라 전처리 포함 p50/p95 지연시간, 배치 크기, 입력 길이, 실행 장치를 함께 기록한다. ONNX 변환 뒤에는
원본 모델과 logits/최종 라벨이 허용 오차 안에서 같은지 먼저 검증해야 한다.

```bash
python 03-advanced/04-serving-onnx-latency/export_onnx.py \
  --model runs/models/mbert-v1 --out runs/models/mbert-v1/model.onnx
python 03-advanced/04-serving-onnx-latency/bench_latency.py \
  --model runs/models/mbert-v1 --onnx runs/models/mbert-v1/model.onnx
```

샘플 하나의 CPU 벤치만으로 운영 처리량을 주장하지 않는다. 짧은/긴 입력, batch 1/8/32, 동시성 조건을 분리한다.

