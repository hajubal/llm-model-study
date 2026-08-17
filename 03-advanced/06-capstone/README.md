# 06 · 캡스톤 — 데이터에서 배포 결정까지

## 목표

처음 보는 데이터 묶음으로 **탐지기 v1을 만들고, 측정하고, 개선해 v2를 만들고, 운영 정책과 리포트까지**
완성한다. 지금까지 레슨별로 나눠 배운 것을 한 줄로 연결하는 작업이다.

**공격 성공률을 높이기 위한 외부 시스템 실험은 범위 밖이다.** 모든 작업은 로컬 분류 실습이다.

---

## 데이터를 어디서 가져오는가

세 가지 경로가 있다. 하나를 고르거나 섞는다.

| 경로 | 방법 | 주의 |
|---|---|---|
| **A. 직접 작성** | 자기 도메인(사내 도구, 고객 상담 등)의 문장을 손으로 작성 | 가장 안전. 최소 라벨당 40건 |
| **B. 합성 확장** | `guardlab.synth`의 템플릿을 자기 도메인으로 교체 | 빠르지만 표현 다양성 한계 |
| **C. 공개 데이터** | 공개 jailbreak/injection 데이터셋 사용 | **라이선스·재배포 조건을 반드시 확인** |

C를 고른다면 다음을 먼저 문서화한다.

- 데이터셋 이름, 출처 URL, 라이선스, 수집 시점
- 상업적 사용 가능 여부, 재배포 가능 여부
- 개인정보 포함 여부와 처리 방법
- 기존 벤치마크(`common/data/bench/`)와의 중복 여부

**운영 로그를 쓰려면** 수집 근거, PII 제거, 보존 기간, 접근 통제를 먼저 설계한다. 이 설계 없이 로그를
저장소에 넣지 않는다.

---

## 진행 순서

### 1단계 · 위협 모델과 어노테이션 가이드 (`01-beginner/01`)

- 보호 대상 한 문장
- 라벨 정의와 판정 순서
- dual-use 판정 기준 ("분석 대상이면 통과, 답변 목표이면 차단")
- 보류 규칙과 검수 절차
- **애매한 사례 5건**과 각각의 결정 기록

### 2단계 · 데이터 manifest와 누수 검사 (`02-intermediate/01`, `02`)

```bash
python 02-intermediate/01-dataset-schema/inspect_data.py runs/capstone/data/v1
```

- 라이선스·PII 검토 결과를 `manifest.json`에 기록
- exact duplicate 검사 (텍스트 해시)
- **near-duplicate 검사** (정규화 후 비교 또는 유사도 기반) — 방법과 결과를 기록
- `group_id` 설계 근거: 무엇을 같은 그룹으로 묶었는가
- 층화 결과: 각 split의 label / language / source 분포

### 3단계 · 세 가지 기준선 (`01-beginner/02`, `04`, `02-intermediate/03`)

```bash
# 규칙
python 01-beginner/02-rule-baseline/rule_detector.py \
  --input runs/capstone/data/v1/test.jsonl --output runs/capstone/rule-pred.jsonl

# 모델
python 02-intermediate/03-first-finetune/train_seq_cls.py \
  --data runs/capstone/data/v1 --out runs/capstone/models/v1

# 하이브리드
python 03-advanced/03-hybrid-gates/apply_policy.py \
  --input runs/capstone/data/v1/test.jsonl \
  --pred runs/capstone/models/v1/test-pred.jsonl \
  --output runs/capstone/decisions-v1.jsonl
```

세 기준선을 **같은 채점기, 같은 데이터**로 비교한 표를 만든다.

### 4단계 · 전체 및 slice 보고서 (`02-intermediate/04`, `03-advanced/05`)

- 전체 지표 + label / language / source / **길이** / **변형** slice
- 각 slice의 표본 수와 신뢰구간
- 가장 나쁜 slice와 그 원인 가설

길이 slice는 직접 만든다.

```bash
python - <<'PY'
import collections
from transformers import AutoTokenizer
from guardlab.io import read_jsonl, read_predictions
from guardlab.eval import evaluate

tok = AutoTokenizer.from_pretrained("runs/capstone/models/v1")
gold = read_jsonl("runs/capstone/data/v1/test.jsonl")
pred = {p.id: p for p in read_predictions("runs/capstone/models/v1/test-pred.jsonl")}

def bucket(n):
    return "<20" if n < 20 else "20-35" if n < 35 else "35-55" if n < 55 else "55+"

b = collections.defaultdict(list)
for s in gold:
    b[bucket(len(tok(s.text)["input_ids"]))].append(s)
for key in ("<20", "20-35", "35-55", "55+"):
    if not b[key]:
        continue
    r = evaluate(b[key], [pred[s.id] for s in b[key]], collect_errors=False)
    print(f"{key:6} n={r.n_samples:4} macro_f1={r.macro_f1:.3f} recall={r.attack_recall:.3f} fpr={r.benign_fpr:.3f}")
PY
```

### 5단계 · dev에서 threshold 결정, test 1회 (`02-intermediate/05`)

- dev 스윕 결과 표
- 운영 제약(FPR 상한 또는 recall 하한)을 **먼저** 명시
- 고른 값과 근거
- test 적용 결과 (**한 번만**)

### 6단계 · 개선 루프 v1 → v2 (`02-intermediate/06`)

- 실패 사례 **20건**을 분류(오탐 / 미탐 / 유형 혼동, slice별)
- 가장 큰 오류군 하나 선택
- **변경 하나**만 적용
- 같은 채점기로 v1/v2 비교 (개선·악화 모두)

### 7단계 · 서빙 동등성과 지연시간 (`03-advanced/04`)

- torch vs ONNX **라벨 일치율** (n≥100)
- 뒤집힘 방향별 건수 (`to_block` / `to_safe`)
- p50/p95: 입력 길이별, batch별, **문서 단위**
- 측정 조건 명시

### 8단계 · 운영 정책 (`03-advanced/03`)

- allow / review / block 임계값과 근거
- **review 비율과 그 운영 비용**
- fail 정책: 추론 실패 / 길이 초과 / 백엔드 다운 시 각각의 동작
- 롤백 조건과 모니터링 지표

### 9단계 · 방어 심층화 설명

**탐지기가 완전히 실패해도 권한이 보호되는 구조**를 데이터 흐름으로 설명한다.

```text
사용자 입력 → [탐지] → LLM → [출력 검증] → 도구 호출 → [allowlist] → 실행
                ↓ 실패해도              ↓ 여기서 막힘        ↓ 여기서도 막힘
```

각 층에서 무엇을 막는지, 탐지가 뚫렸을 때 다음 층이 무엇을 하는지 적는다.

### 10단계 · 재현 명령, 모델 카드, 리포트

- 처음부터 끝까지 실행하는 명령 목록 (복사해서 그대로 실행 가능해야 함)
- 모델 카드 (intended / out-of-scope / training data / metrics / limitations)
- 최종 리포트 (`make_report.py` 생성 후 손으로 완성)

---

## 필수 산출물 체크리스트

```text
runs/capstone/
├── GUIDE.md                    # 위협 모델 + 어노테이션 가이드
├── data/
│   ├── v1/{train,dev,test}.jsonl + manifest.json
│   └── v2/... (개선 후)
├── models/
│   ├── v1/  train_summary.json, test-pred.jsonl, test-report/, model.onnx
│   └── v2/  ...
├── reports/
│   ├── baselines.md            # 규칙 / 모델 / 하이브리드 비교
│   ├── slices.md               # label/language/source/길이/변형
│   ├── threshold.md            # dev 스윕 + 선택 근거 + test 1회 결과
│   ├── errors-20.md            # 실패 사례 20건 분류
│   ├── serving.md              # 동등성 + 지연시간
│   └── model-report.md         # 최종 리포트
├── POLICY.md                   # 운영 정책 + fail 정책 + 롤백
├── MODEL_CARD.md
├── DEFENSE_IN_DEPTH.md         # 9단계
└── REPRODUCE.md                # 재현 명령
```

---

## 통과 기준

`RUBRIC.md`의 10개 항목에서 **16점 이상**, 그리고 **★ 표시된 3개 항목이 모두 2점**이어야 한다.

★ 항목은 타협할 수 없는 것들이다.

| ★ 항목 | 왜 필수인가 |
|---|---|
| 데이터 안전 | 라이선스·PII 위반은 기술 문제가 아니라 법적 문제다 |
| 누수 방지 | 누수가 있으면 다른 모든 숫자가 무의미하다 |
| test 규율 | test를 반복해서 보면 그 숫자는 성능이 아니다 |

**높은 점수는 통과 기준이 아니다.** macro F1 0.6짜리 모델이라도 한계를 정확히 알고 문서화했다면
통과이고, 0.95인데 누수가 있으면 불합격이다.

---

## 자주 하는 실수

| 실수 | 결과 |
|---|---|
| 데이터를 먼저 만들고 가이드를 나중에 씀 | 라벨 기준이 사후 정당화가 된다 |
| test를 보며 여러 번 조정 | ★ test 규율 0점 |
| 개선된 slice만 보고 | 보고서 항목 감점 |
| ONNX 속도만 재고 동등성 생략 | 서빙 항목 감점 |
| review 비율을 안 잼 | 운영 불가능한 정책 |
| "모델이 잡으니 안전하다"고 서술 | 방어 심층화 0점 |
