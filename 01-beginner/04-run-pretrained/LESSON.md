# 04 · 사전학습 제로샷 모델

NLI 기반 제로샷 분류기는 별도 학습 없이 자연어 라벨 설명과 입력의 관계를 비교한다. 빠른 비교 기준으로는 좋지만
보안 데이터로 학습된 운영 탐지기가 아니며, 라벨 문구와 언어에 따라 점수가 크게 변한다.

```bash
python 01-beginner/04-run-pretrained/run_zero_shot.py \
  --input common/data/bench/gold.jsonl --output runs/zero-shot-pred.jsonl
python 01-beginner/03-evaluation-basics/evaluate.py \
  --gold common/data/bench/gold.jsonl --pred runs/zero-shot-pred.jsonl
```

가중치는 처음 실행할 때 내려받는다. 모델 카드와 라이선스를 확인한다. 이 결과를 보안 성능 보증으로 사용하지 않는다.

