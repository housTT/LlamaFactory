#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"
export PATH="$ROOT/.venv/bin:$PATH"
export PJRT_DEVICE=TT
: "${TT_VISIBLE_DEVICES:?set TT_VISIBLE_DEVICES to the chip index that training jobs should use, see tt-smi -ls}"
export TT_VISIBLE_DEVICES
export GRADIO_SERVER_NAME="${GRADIO_SERVER_NAME:-0.0.0.0}"
export GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT:-7860}"

cd "$ROOT"
exec llamafactory-cli webui
