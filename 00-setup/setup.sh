#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "[error] uv가 없습니다. brew install uv 후 다시 실행하세요." >&2
  exit 1
fi

if [ ! -d .venv ]; then
  echo "[1/4] Python 3.12 가상환경 생성"
  uv venv --python 3.12 .venv
else
  echo "[1/4] .venv가 이미 있어 재사용"
fi

# shellcheck disable=SC1091
source .venv/bin/activate
echo "[2/4] 학습 의존성 설치"
uv pip install -q -r 00-setup/requirements.txt
echo "[3/4] guardlab editable 설치"
uv pip install -q -e common
echo "[4/4] 환경 검증"
python 00-setup/check_env.py
mkdir -p runs

echo
echo "완료. 다음 명령으로 시작하세요:"
echo "  cd ~/project/llm-model-study && source .venv/bin/activate"
echo "  open 01-beginner/README.md"

