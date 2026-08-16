# 06 · 미니 프로젝트 — 개선 루프 한 바퀴

1. v1 데이터/모델을 고정 벤치마크에 평가한다.
2. `errors.jsonl`에서 가장 큰 오류군을 고른다.
3. 데이터 보강, 라벨 수정, threshold 중 하나만 바꿔 v2를 만든다.
4. 같은 벤치마크와 같은 채점기로 v1/v2를 비교한다.
5. 개선된 slice와 악화된 slice를 모두 기록한다.

제출물:

- `runs/data/v2/manifest.json`
- `runs/models/<name>-v2/train_summary.json`
- v1/v2 `report.json`, `errors.jsonl`
- `compare_runs.py`가 만든 비교표
- 무엇을 왜 바꿨고, 다음에 무엇을 검증할지 적은 1페이지 회고

통과 기준은 무조건 높은 숫자가 아니라 변경 가설과 결과가 연결되고, test를 보며 반복 튜닝하지 않은 것이다.

