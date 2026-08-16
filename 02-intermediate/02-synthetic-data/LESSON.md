# 02 · 합성 데이터 만들기

합성 데이터는 스키마와 학습 파이프라인을 익히는 데 좋지만 현실 분포를 대신하지 못한다. 생성 템플릿의 단어를
모델이 외우기 쉽고, 실제 공격의 길이·언어·오타·외부 컨텍스트를 충분히 표현하지 못한다.

```bash
python 02-intermediate/02-synthetic-data/gen_synth.py --out runs/data/v1
python 02-intermediate/01-dataset-schema/inspect_data.py runs/data/v1
```

생성기는 템플릿을 `group_id`로 사용해 파생 문장이 split을 넘지 않게 한다. `manifest.json`은 재현에 필요한 seed와
한계를 남긴다. 운영 로그를 추가할 때는 비밀/개인정보 제거, 수집 근거, 보존 기간, 검수자 간 일치도를 먼저 설계한다.

