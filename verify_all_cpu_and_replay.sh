#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
bash verify_all.sh
python3 reproducibility/qwen_cache_pinned_replay/verify_replay_cpu.py
echo "FULL CPU RELEASE VERIFICATION PASS"
