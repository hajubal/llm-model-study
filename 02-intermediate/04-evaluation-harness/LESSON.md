# 04 · 재현 가능한 평가 하네스

평가 스크립트는 화면에 숫자만 찍지 않고 `report.json`, `report.md`, `errors.jsonl`을 함께 남긴다. JSON은 이후
비교 자동화에, Markdown은 사람 검토에, 오류 원문은 데이터 개선에 쓴다.

```bash
python 02-intermediate/04-evaluation-harness/evaluate.py \
  --gold runs/data/v1/test.jsonl \
  --pred runs/models/mbert-v1/test-pred.jsonl \
  --out runs/models/mbert-v1/test-report
```

같은 모델을 `common/data/bench/gold.jsonl`에도 적용한다. 합성 test와 독립 bench 사이의 차이가 일반화 격차다.

