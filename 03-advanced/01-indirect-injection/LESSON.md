# 01 · 간접 Prompt Injection과 긴 문서

검색 문서나 도구 출력은 데이터이지 명령이 아니다. 하지만 애플리케이션이 같은 컨텍스트에 이어 붙이면 모델은 둘의
경계를 완벽히 지키지 못한다. 탐지기도 `max_len` 뒤쪽 텍스트를 보지 못하면 문서 끝의 공격을 놓친다.

```bash
python 03-advanced/01-indirect-injection/make_indirect_eval.py
python 03-advanced/01-indirect-injection/chunk_predict.py \
  --model runs/models/mbert-v1 --input runs/data/indirect-eval.jsonl \
  --output runs/models/mbert-v1/indirect-pred.jsonl
```

chunk의 공격 확률을 max로 모으면 recall은 좋아지지만, 문서가 길어 window가 많아질수록 우연한 오탐 기회도 늘어난다.
문서 길이별 FPR을 따로 잰다. 탐지와 별개로 컨텍스트 출처 표시, 도구 최소 권한, 민감 작업 확인을 적용한다.

