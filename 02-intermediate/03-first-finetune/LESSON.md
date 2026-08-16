# 03 · 첫 문장 분류 모델 파인튜닝

다국어 DistilBERT 인코더 위에 3-class 분류층을 붙인다. 입력 전체를 하나의 라벨로 분류하는 sequence
classification이다. 긴 문서가 `max_len`에서 잘리면 뒤쪽의 간접 인젝션을 보지 못하므로 고급에서 chunk 정책을 다룬다.

```bash
python 02-intermediate/03-first-finetune/train_seq_cls.py \
  --data runs/data/v1 --out runs/models/mbert-v1
python 02-intermediate/03-first-finetune/predict.py \
  --model runs/models/mbert-v1 --input runs/data/v1/test.jsonl \
  --output runs/models/mbert-v1/test-pred.jsonl
```

파이프라인만 확인할 때는 `--max-steps 5 --out runs/models/smoke`를 쓴다. 합성 test 점수가 높아도 독립
벤치마크 일반화를 의미하지 않는다.

