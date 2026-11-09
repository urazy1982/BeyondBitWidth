#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
: "${HF_TOKEN:?Set HF_TOKEN in your terminal before launching; never paste it into the script}"
mkdir -p five_results
# Each dataset is independently frozen and evaluated. Failures are logged and do not overwrite others.
fail=0
for dataset in qwen_thinking wildguard_prompt toxicchat_prompt xstest_prompt beavertails_response; do
  echo "========== DATASET: $dataset =========="
  if python -u run_five.py --dataset "$dataset" --max-items 5000 --out five_results 2>&1 | tee "five_results/${dataset}.log"; then
    echo "PASS: $dataset"
  else
    echo "FAILED: $dataset (inspect five_results/${dataset}.log)" >&2
    fail=$((fail+1))
  fi
done
python run_five.py --report-only --out five_results || true
if (( fail > 0 )); then echo "$fail dataset(s) failed; examine logs and rerun each independently" >&2; exit 1; fi
