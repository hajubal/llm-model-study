# 중급 · 데이터를 만들고, 첫 모델을 학습하고, 오류로 개선한다

중급의 중심은 학습 명령 자체가 아니라 데이터 계약과 개선 루프다.

| # | 레슨 | 산출물 |
|---|---|---|
| 01 | 데이터 스키마와 group split | 누수 검사 결과 |
| 02 | 교육용 합성 데이터 | `runs/data/v1/{train,dev,test}.jsonl` |
| 03 | 첫 sequence classifier 파인튜닝 | `runs/models/mbert-v1/` |
| 04 | 평가 하네스 | JSON/Markdown 보고서 |
| 05 | 에러 분석과 threshold | 오류표·threshold sweep |
| 06 | 미니 프로젝트 | v1→v2 비교와 회고 |

데이터 버전과 모델 버전을 분리한다. 같은 파일을 덮어쓰지 말고, 실행 폴더마다 설정과 seed를 함께 남긴다.

