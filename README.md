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
{"id":"ko-pi-001","text":"...","label":"PROMPT_INJECTION","source":"user","language":"ko","group_id":"pi-ko-ignore","meta":{"synthetic":true}}
```

- `id`: 분할 전체에서 유일한 ID
- `text`: 분류할 문자열. 비밀·개인정보를 넣지 않는다.
- `label`: `BENIGN`, `PROMPT_INJECTION`, `JAILBREAK`
- `source`: `user`, `retrieved`, `tool`, `system` 중 입력이 유입된 경로.
  **서버가 신뢰 경계에서 부여한다.** 클라이언트가 이 값을 정할 수 있으면 아무 의미가 없다.
- `group_id`: 같은 템플릿/원문의 파생 샘플을 묶는 키. 같은 그룹은 한 split에만 둔다.
- `meta`: 생성 방식, 보조 라벨, 검수 상태 등. 모델 정답의 지름길로 사용하지 않는다.

`source`에 `system`이 있는 이유: system 채널이라고 항상 우리가 쓴 문장인 것은 아니다.
프롬프트 템플릿은 설정 저장소에서 오고, 멀티테넌트 제품에서는 테넌트가 넣은 문장이 system으로
들어가기도 한다. **"system이니까 신뢰한다"는 가정 자체가 위협 모델의 대상이다.**

## 3. 학습 방법

1. 스크립트를 먼저 읽고 출력과 실패 조건을 예측한다.
2. 모든 실험 결과는 `runs/` 아래 새 폴더에 남긴다. 기존 결과를 덮어쓰지 않는다.
3. 데이터 변경과 임계값 변경을 동시에 하지 않는다. 한 번에 한 변수만 바꾼다.
4. 랜덤 행 분할 대신 `group_id` 분할을 사용해 템플릿 누수를 막는다. 분할은 `label`·`language`·`source`를
   함께 층화하므로, dev/test에도 검색 문서와 도구 출력이 남아 간접 인젝션 slice를 볼 수 있다.
5. 정상 하드 네거티브를 공격 데이터만큼 중요하게 다룬다.
6. 샘플 수보다 표현(템플릿) 수를 먼저 늘린다. 같은 문장을 여러 번 넣어도 학습 신호는 늘지 않는다.
7. **숫자에는 불확실성을 함께 적는다.** 두 가지가 있고 둘 다 확인해야 한다.
   - **신뢰구간**: 평가 표본이 작아서 생긴다. `evaluate.py`가 bootstrap 95% 구간을 함께 낸다.
   - **seed 편차**: 학습이 불안정해서 생긴다. `run_seeds.py`로 잰다.
   개선폭이 둘 중 어느 것보다도 작으면 그것은 개선이 아니다.

## 4. 완료 기준

| 단계 | 통과 기준 |
|---|---|
| 초급 | 여러 기준선을 고정 mini set에서 평가하고 macro F1, attack recall, benign FPR의 차이를 설명한다 |
| 중급 | group split 데이터로 모델을 학습하고, **seed 편차와 신뢰구간을 먼저 잰 뒤** 개선 여부를 판단한다. dev에서 정한 threshold를 test에 한 번만 적용하고, 오류군 하나를 골라 개선 루프를 1회 수행한다 |
| 고급 | 간접 인젝션·변형·하드 네거티브 벤치 결과, 하이브리드 정책, 지연시간과 비용을 한 리포트로 묶고 한계를 기록한다 |

### 기본 설정으로 나오는 숫자를 미리 알아 둔다

기본 제공 합성 데이터(템플릿 108개)로 그대로 학습하면 seed에 따라 이렇게 나온다.

| 지표 | 평균 | 최대−최소 |
|---|---:|---:|
| 합성 test macro F1 | 0.674 | **0.108** |
| attack recall | 0.904 | **0.256** |
| benign FPR | 0.487 | **0.287** |

**가장 중요한 열은 오른쪽이다.** seed만 바꿔도 macro F1이 0.11, benign FPR이 0.29 폭으로
흔들린다. 즉 **단일 학습 결과를 소수점 셋째 자리까지 인용하는 것은 근거가 없다.**

`macro F1 ≥ 0.80`, `benign FPR ≤ 5%`는 **개선 루프의 목표치**이지 기본 설정의 결과가 아니다.
여기에 도달하려면 약한 slice(`retrieved` / `system`, 조합당 train 템플릿 1개)를 보강하고
(`02-intermediate/02` 과제), threshold를 dev에서 조정해야 한다.

**숫자가 바로 나오지 않는 것이 실패가 아니다.** 무엇을 바꿔야 오르는지 찾는 것, 그리고
그 변화가 노이즈보다 큰지 판단하는 것이 이 단계의 과제다. 전체 실측은 [RESULTS.md](RESULTS.md).

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
├── .github/workflows/ci.yml   # 회귀 게이트 (03-advanced/03의 산출물)
├── 00-setup/
├── common/
│   ├── guardlab/       # 스키마·IO·평가·규칙·통계·합성 데이터
│   ├── data/bench/     # 작고 고정된 교육용 벤치마크
│   └── tests/
├── 01-beginner/        # + SOLUTIONS.md
├── 02-intermediate/    # + SOLUTIONS.md
├── 03-advanced/        # + SOLUTIONS.md, APPENDIX-production-checklist.md
└── runs/               # 생성 데이터·모델·예측·리포트 (git 제외)
```

각 단계 폴더의 `SOLUTIONS.md`에 수료 기준과 주요 과제의 **예시 답안**이 있다.
먼저 스스로 답한 뒤에 열 것 — 유일한 정답이 아니라 "이 정도면 통과"의 기준선이다.

