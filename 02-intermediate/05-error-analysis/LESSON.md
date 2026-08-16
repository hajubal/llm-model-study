# 05 · 에러 분석과 threshold

오류를 먼저 `BENIGN→attack`, `attack→BENIGN`, 공격 유형 간 confusion으로 나눈다. 그다음 source/language/길이와
함께 보며 가장 큰 오류군 하나에만 처방한다.

- 정상 오탐: hard negative 추가, 어노테이션 경계 재검토, threshold 상향
- 공격 미탐: 표현 다양성 추가, 긴 입력 처리, threshold 하향
- 유형 혼동: 정의와 라벨 일치도 점검. 운영이 binary 차단이면 우선순위가 낮을 수 있음

threshold는 test가 아니라 dev에서 정하고 test에는 한 번만 적용한다. 같은 test를 보며 반복 조정하면 test가 사실상
학습 데이터가 된다.

