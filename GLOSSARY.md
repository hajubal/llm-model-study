# 용어집

- **Prompt Injection**: 사용자 입력 또는 외부 문서가 애플리케이션의 원래 지시를 바꾸려는 공격.
- **Direct Injection**: 사용자가 모델 입력에 직접 넣는 인젝션.
- **Indirect Injection**: 검색 문서, 이메일, 웹페이지, 도구 출력처럼 신뢰하지 않는 컨텍스트에 섞인 인젝션.
- **Jailbreak**: 모델의 안전 정책이나 제한을 해제·우회하도록 유도하는 입력.
- **Hard Negative**: 공격과 비슷한 단어를 포함하지만 실제로는 정상인 문장. 오탐을 줄이는 데 중요하다.
- **Data Leakage**: 같은 템플릿이나 변형 문장이 train/test 양쪽에 있어 실제보다 높은 점수가 나오는 현상.
- **Group Split**: 같은 `group_id`를 반드시 한 split에만 배치하는 분할법.
- **Macro F1**: 라벨별 F1을 동일한 비중으로 평균한 값.
- **Attack Recall**: `PROMPT_INJECTION` 또는 `JAILBREAK` 샘플 중 공격으로 탐지된 비율.
- **Benign FPR**: `BENIGN` 샘플 중 공격으로 잘못 탐지된 비율.
- **Threshold**: 공격 점수가 이 값 이상일 때 차단/검토로 보내는 경계.
- **Calibration**: 모델 점수가 실제 정답 확률과 비슷하도록 맞추는 과정.
- **Slice**: 언어, 입력 출처, 길이 등 특정 조건으로 나눈 평가 부분집합.
- **ASR (Attack Success Rate)**: 실제 대상 시스템에서 공격 목표가 성공한 비율. 탐지 recall과 다르며 이 과정은 공격 실행을 자동화하지 않는다.
- **Defense in Depth**: 탐지 모델, 권한 통제, 도구 제한, 출력 검증 등 여러 방어층을 겹치는 설계.

