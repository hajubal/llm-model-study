# 03 · 평가 기초

Accuracy는 클래스가 불균형할 때 쉽게 속인다. 이 과정은 세 라벨의 macro F1과 함께 운영상의 두 질문을 따로 잰다.

- attack recall: 실제 공격 100개 중 몇 개를 공격으로 잡았는가?
- benign FPR: 정상 100개 중 몇 개를 잘못 차단했는가?

Prompt Injection을 Jailbreak로 예측하면 3-class 평가에서는 오답이지만 binary 방어 판단에서는 공격을 잡은 것이다.
그래서 confusion matrix와 attack recall을 함께 본다. 예측 파일은 행 순서가 아니라 `id`로 정렬하며 누락/추가 ID를
에러로 처리한다.

```bash
python 01-beginner/03-evaluation-basics/evaluate.py \
  --gold common/data/bench/gold.jsonl --pred runs/rule-pred.jsonl \
  --json-out runs/rule-report.json
```

