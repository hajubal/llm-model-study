# LLM 보안 탐지 모델 개발 커리큘럼 — Jailbreak · Prompt Injection

> **대상**: 백엔드/LLM 애플리케이션 개발 경험은 있지만 ML 모델 개발은 처음인 개발자  
> **목표**: Jailbreak와 Prompt Injection 탐지기를 직접 만들고, 재현 가능한 벤치마크로 평가하며,
> 우회·오탐 사례를 분석해 모델과 운영 정책을 함께 개선한다.  
> **장비**: Apple Silicon 16GB에서 전 과정 실행 가능. CPU도 가능하지만 파인튜닝은 느리다.

이 저장소는 공격 기법을 실행하거나 다른 시스템을 우회하기 위한 자료가 아니다. 모든 공격 문장은
로컬 분류 실습용 텍스트이며 외부 서비스로 자동 전송하지 않는다. 실제 사용자 대화나 비밀이 포함된
운영 로그를 커밋하지 말고, 수집·보존 정책과 개인정보 규정을 먼저 확인한다.

## 0. 왜 학습보다 측정부터 시작하는가

좋은 탐지기는 공격 문장만 잘 잡는 모델이 아니다. 정상적인 보안 질문, 역할극, 인용문, 시스템 프롬프트를
설명하는 문장을 공격으로 오인하지 않아야 한다. 따라서 정확도 하나가 아니라 다음을 함께 본다.

- `macro F1`: 세 라벨을 균등하게 본 분류 성능
- `attack recall`: 실제 공격을 놓치지 않는 비율
- `benign FPR`: 정상 문장을 차단한 비율
- 라벨별 confusion: Prompt Injection과 Jailbreak를 서로 헷갈리는 정도
- slice 성능: 한국어/영어, 직접 입력/검색 문서, 짧은/긴 문장별 성능

기본 라벨은 아래 셋이다.

| 라벨 | 의미 | 예 |
|---|---|---|
| `BENIGN` | 정상 요청 또는 공격을 설명·인용하는 안전한 텍스트 | "프롬프트 인젝션의 정의를 설명해 줘" |
| `PROMPT_INJECTION` | 모델의 지시 계층이나 외부 컨텍스트를 바꾸려는 입력 | 이전 지시 무시 요구, 검색 문서 속 숨은 명령 |
| `JAILBREAK` | 안전 정책이나 제한을 해제·우회하도록 요구하는 입력 | 무제한 역할을 부여하거나 정책 무시를 요구 |

라벨은 완전한 현실을 표현하지 않는다. 두 공격은 겹칠 수 있으므로 애매한 사례는 `meta.secondary_labels`에
보존하고, 데이터셋마다 어노테이션 가이드를 함께 버전 관리한다.

## 1. 로드맵

| 단계 | 폴더 | 분량 | 끝나면 할 수 있는 것 |
|---|---|---:|---|
| 환경 | `00-setup/` | 30분 | Python 3.12, torch/MPS, 공용 패키지 설치·검증 |
| 초급 | `01-beginner/` | 1일 | 위협 모델, 규칙 기반선, 채점기, 제로샷 모델 비교 |
| 중급 | `02-intermediate/` | 1~2주 | 누수 없는 데이터 생성, 첫 분류 모델 파인튜닝, 에러 분석과 개선 |
| 고급 | `03-advanced/` | 2~3주 | 간접 인젝션, 적대적 변형, 하이브리드 게이트, ONNX, 벤치마크·캡스톤 |
| 공용 | `common/` | — | `guardlab` 스키마·평가·규칙·합성 데이터와 고정 벤치마크 |

각 레슨은 가능한 한 `LESSON.md`(개념), 짧은 실행 스크립트, `EXERCISE.md`(과제)로 구성한다.

## 2. 데이터 계약

한 줄에 한 샘플인 JSONL을 사용한다.

```json
{"id":"ko-pi-001","text":"...","label":"PROMPT_INJECTION","source":"user","language":"ko","group_id":"pi-ignore-previous","meta":{"synthetic":true}}
```

- `id`: 분할 전체에서 유일한 ID
- `text`: 분류할 문자열. 비밀·개인정보를 넣지 않는다.
- `label`: `BENIGN`, `PROMPT_INJECTION`, `JAILBREAK`
- `source`: `user`, `retrieved`, `tool`, `system` 중 입력이 유입된 경로
- `group_id`: 같은 템플릿/원문의 파생 샘플을 묶는 키. 같은 그룹은 한 split에만 둔다.
- `meta`: 생성 방식, 보조 라벨, 검수 상태 등. 모델 정답의 지름길로 사용하지 않는다.

## 3. 학습 방법

1. 스크립트를 먼저 읽고 출력과 실패 조건을 예측한다.
2. 모든 실험 결과는 `runs/` 아래 새 폴더에 남긴다. 기존 결과를 덮어쓰지 않는다.
3. 데이터 변경과 임계값 변경을 동시에 하지 않는다. 한 번에 한 변수만 바꾼다.
4. 랜덤 행 분할 대신 `group_id` 분할을 사용해 템플릿 누수를 막는다.
5. 정상 하드 네거티브를 공격 데이터만큼 중요하게 다룬다.

## 4. 완료 기준

| 단계 | 통과 기준 |
|---|---|
| 초급 | 규칙 기반선을 고정 mini set에서 평가하고 macro F1, attack recall, benign FPR의 차이를 설명한다 |
| 중급 | group split 데이터로 모델을 학습하고 test macro F1 ≥ 0.80, benign FPR ≤ 5%를 목표로 개선 루프를 1회 수행한다 |
| 고급 | 간접 인젝션·변형·하드 네거티브 벤치 결과, 하이브리드 정책, 지연시간을 한 리포트로 묶고 한계를 기록한다 |

수치는 학습 목표이지 보안 보증이 아니다. 탐지 모델 하나로 입력을 안전하다고 증명할 수 없으며, 권한 분리,
도구 allowlist, 컨텍스트 경계, 출력 검증, 감사 로그와 함께 방어층으로 사용해야 한다.

## 5. 빠른 시작

```bash
cd ~/project/llm-model-study
bash 00-setup/setup.sh
source .venv/bin/activate
python -m pytest common/tests -q
python 01-beginner/02-rule-baseline/rule_detector.py \
  --input common/data/bench/gold.jsonl --output runs/rule-pred.jsonl
python 01-beginner/03-evaluation-basics/evaluate.py \
  --gold common/data/bench/gold.jsonl --pred runs/rule-pred.jsonl
```

## 6. 파일 지도

```text
llm-model-study/
├── README.md
├── GLOSSARY.md
├── RESULTS.md
├── JOURNAL.md
├── 00-setup/
├── common/
│   ├── guardlab/       # 스키마·IO·평가·규칙·합성 데이터
│   ├── data/bench/     # 작고 고정된 교육용 벤치마크
│   └── tests/
├── 01-beginner/
├── 02-intermediate/
├── 03-advanced/
└── runs/               # 생성 데이터·모델·예측·리포트 (git 제외)
```

