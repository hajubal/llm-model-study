# 00 · 환경 설정

## 목표

- Python 3.12 가상환경과 PyTorch/Transformers를 설치한다.
- Apple Silicon에서는 MPS, NVIDIA 환경에서는 CUDA, 그 외에는 CPU를 자동 선택한다.
- 공용 `guardlab` 패키지와 오프라인 단위 테스트가 동작하는지 확인한다.

## 실행

```bash
cd ~/project/llm-model-study
bash 00-setup/setup.sh
source .venv/bin/activate
python -m pytest common/tests -q
```

모델을 처음 실행하면 Hugging Face에서 가중치를 내려받는다. 외부 모델은 라이선스와 모델 카드를 확인하고,
운영 데이터가 외부 API로 전송되지 않도록 한다. 이 저장소의 기본 스크립트는 로컬 추론만 수행한다.

MPS에서 미지원 연산 오류가 나면 다음을 설정한다.

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

16GB 장비에서 메모리가 부족하면 `--batch 4`, `--max-len 128`부터 시작하고 gradient accumulation을 사용한다.

