#!/usr/bin/env bash
set -euo pipefail

export PJRT_DEVICE=TT
: "${TT_VISIBLE_DEVICES:?set TT_VISIBLE_DEVICES to the chip index to train on, see tt-smi -ls}"
export TT_VISIBLE_DEVICES

CONFIG="${1:-$(dirname "$(readlink -f "$0")")/llama3_lora_sft.yaml}"
llamafactory-cli train "$CONFIG"
