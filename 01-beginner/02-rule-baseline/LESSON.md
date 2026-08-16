# 02 · 규칙 기반선

규칙 탐지기는 최종 해답이 아니라 비교 기준이다. 빠르고 설명 가능하지만 동의어, 철자 변형, 긴 문맥에 약하고
“공격을 설명하는 정상 문장”을 잘못 잡는다. `common/guardlab/rules.py`의 패턴과 교육 문맥 discount를 읽은 뒤
결과를 예측해 본다.

```bash
python 01-beginner/02-rule-baseline/rule_detector.py \
  --input common/data/bench/gold.jsonl --output runs/rule-pred.jsonl
```

점수가 잘 나오더라도 데이터와 규칙이 같은 표현을 공유하기 때문일 수 있다. 고급에서는 표현 변형과 독립
holdout으로 이 착시를 확인한다.

