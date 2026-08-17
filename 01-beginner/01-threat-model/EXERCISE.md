# 과제 · 어노테이션 가이드 만들기

## 목표

직접 작성한 10건의 데이터로 **어노테이션 가이드 초안**을 만든다. 이 파일은 이후 모든 레슨에서 라벨 판정의
기준이 되므로, 남이 읽고 같은 판정을 내릴 수 있을 만큼 구체적이어야 한다.

## 선행 조건

- `LESSON.md`의 판정 순서(4단계)를 읽었을 것
- 코드 실행은 필요 없다. 텍스트 파일만 만든다.

---

## 과제 1 · 샘플 10건 작성

`runs/exercise/01-threat-model/samples.jsonl`에 아래 구성으로 10건을 작성한다. 한 줄에 한 건(JSONL)이다.

| 라벨 | 건수 | 조건 |
|---|---:|---|
| `BENIGN` | 4 | 이 중 **2건은 공격 문구를 인용한 정상 문장**이어야 한다 |
| `PROMPT_INJECTION` | 3 | 최소 1건은 `source`가 `retrieved` 또는 `tool` |
| `JAILBREAK` | 3 | 최소 1건은 영어 |

각 줄의 형식은 데이터 계약(`README.md` 2절)을 따른다.

```json
{"id":"ex-001","text":"...","label":"BENIGN","source":"user","language":"ko","group_id":"ex-quote","meta":{"reason":"공격 문구를 인용했지만 목표는 설명"}}
```

- `id`: 중복 없는 문자열
- `group_id`: 같은 아이디어에서 파생된 문장은 같은 값. 서로 다른 아이디어면 다른 값
- `meta.reason`: **왜 그 라벨인지 한 줄**. 이 필드가 이 과제의 핵심이다
- 실제 비밀·개인정보·운영 로그를 넣지 않는다

### 검증

```bash
python -c "
from guardlab.io import read_jsonl
import collections
rows = read_jsonl('runs/exercise/01-threat-model/samples.jsonl')
print('총', len(rows), '건')
print('라벨', dict(collections.Counter(r.label for r in rows)))
print('source', dict(collections.Counter(r.source for r in rows)))
print('language', dict(collections.Counter(r.language for r in rows)))
print('reason 누락:', [r.id for r in rows if not r.meta.get('reason')])
"
```

기대 출력: 총 10건, 라벨이 4/3/3, `reason 누락: []`. 스키마가 틀리면 `read_jsonl`이 어느 줄이 왜 틀렸는지
알려주며 실패한다.

---

## 과제 2 · 하드 네거티브 2건 검증

과제 1에서 만든 "공격 문구를 인용한 정상 문장" 2건이 실제로 규칙 탐지기를 속이는지 확인한다.

```bash
python 01-beginner/02-rule-baseline/rule_detector.py \
  --input runs/exercise/01-threat-model/samples.jsonl \
  --output runs/exercise/01-threat-model/rule-pred.jsonl
python 01-beginner/03-evaluation-basics/evaluate.py \
  --gold runs/exercise/01-threat-model/samples.jsonl \
  --pred runs/exercise/01-threat-model/rule-pred.jsonl
```

`benign FPR`이 0보다 크면 하드 네거티브가 제 역할을 한 것이다. 0이라면 문장이 너무 순해서 규칙이 걸리지
않은 것이므로, 공격 키워드(`이전 지시`, `시스템 프롬프트`, `제한 없는` 등)를 더 직접적으로 인용해 다시 쓴다.

> 이 과제의 목표는 규칙을 이기는 것이 아니라, **어휘가 겹치지만 라벨이 반대인 문장**을 손으로 만들어 보는
> 것이다. 이런 문장이 중급 이후 오탐의 주범이 된다.

---

## 과제 3 · 애매한 사례 1건과 결정 기록

라벨을 하나로 확정하기 어려운 문장을 1건 만들고, 아래를 `runs/exercise/01-threat-model/GUIDE.md`에 적는다.

```markdown
## 애매한 사례

문장: (여기에 문장)

- 후보 라벨: PROMPT_INJECTION / JAILBREAK
- primary로 고른 것: ...
- 고른 이유: ...
- secondary_labels에 남긴 것: ...
- 이 문장이 실제 서비스에 들어오면 어떤 동작을 원하는가: (차단 / 검토 / 통과)
```

마지막 줄이 중요하다. **라벨과 운영 동작은 다른 층**이다. 라벨이 `JAILBREAK`라도 운영에서는 점수가 낮으면
검토로 보낼 수 있다.

---

## 과제 4 · 가이드 문서 완성

`runs/exercise/01-threat-model/GUIDE.md`에 다음 항목을 채운다.

1. **보호 대상 한 문장** — 이 탐지기가 무엇을 지키는가
2. **라벨 3개의 정의** — 각각 한 문장 + 내가 만든 샘플에서 고른 예시 1건씩
3. **판정 순서** — 레슨의 4단계를 내 표현으로 다시 씀
4. **dual-use 기준** — "분석 대상 vs 답변 목표" 원칙을 내 사례에 적용한 문장 1개
5. **보류 규칙** — 어떤 조건이면 확정하지 않고 검수 큐로 보내는가
6. **금지 사항** — 데이터에 절대 넣지 않을 것(비밀, 개인정보, 실제 운영 로그 등)

---

## 정답 확인

정답이 하나로 정해진 과제가 아니다. 대신 아래를 스스로 점검한다.

- [ ] 내가 만든 10건을 **일주일 뒤에 다시 라벨링해도 같은 답**이 나올 기준인가?
- [ ] 동료에게 가이드만 주고 내 샘플을 라벨링하게 하면 **8건 이상 일치**할 것 같은가?
- [ ] 하드 네거티브 2건이 실제로 규칙 탐지기를 속였는가(`benign FPR > 0`)?
- [ ] 애매한 사례에 대해 "라벨"과 "운영 동작"을 따로 적었는가?
- [ ] 문장에 없는 의도를 추측해서 라벨을 정한 건이 없는가?

## 막혔을 때

- **하드 네거티브가 안 떠오른다** → `common/data/bench/negatives.jsonl`의 8건을 열어보고, 같은 방식으로
  내 도메인 단어를 넣어 변형한다.
- **PROMPT_INJECTION과 JAILBREAK 구분이 안 된다** → "무엇을 없애라고 하는가"를 본다. *앞선 지시*를 없애면
  injection, *안전 규칙*을 없애면 jailbreak다.
- **`read_jsonl`이 계속 실패한다** → 에러 메시지에 파일명과 줄 번호가 찍힌다. `group_id` 누락이 가장 흔하다.

## 제출물

- `runs/exercise/01-threat-model/samples.jsonl` (10건, `meta.reason` 포함)
- `runs/exercise/01-threat-model/GUIDE.md` (6개 항목 + 애매한 사례 결정 기록)
- 규칙 탐지기 평가 결과 캡처 또는 수치 메모 (`benign FPR` 값)
