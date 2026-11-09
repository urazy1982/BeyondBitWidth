#!/usr/bin/env bash
# Recompute every reported number from the raw prediction records (CPU only, ~1 minute).
# Usage: bash verify_all.sh
set -euo pipefail
cd "$(dirname "$0")"
REPO="$PWD"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# The verifiers expect the deposit layout; build it from symlinks (no copies).
mkdir -p "$WORK/root"/{qwen_main,llama,quant_rep,gpu_rep,llama_rep}
ln -s "$REPO/results/five_results"                          "$WORK/root/qwen_main/five_results"
ln -s "$REPO/results/llama_guard_replication_results"       "$WORK/root/llama/llama_guard_replication_results"
ln -s "$REPO/results/guard_quantized_repeatability_results" "$WORK/root/quant_rep/guard_quantized_repeatability_results"
ln -s "$REPO/results/guard_repeatability_results"           "$WORK/root/gpu_rep/guard_repeatability_results"
ln -s "$REPO/results/llama_guard_reload_repeatability"      "$WORK/root/llama_rep/llama_guard_reload_repeatability"

status=0
check() {  # name, command output file, reference file
  if diff -q "$2" "$3" >/dev/null; then echo "PASS  $1 (identical to reference output)"
  else echo "FAIL  $1 (differs from reference; see diff below)"; diff "$2" "$3" | head -20; status=1; fi
}

echo "== 1/4 Checksums"
if sha256sum -c --quiet SHA256SUMS.txt; then echo "PASS  all files match SHA256SUMS.txt"; else status=1; fi

echo "== 2/4 Primary results (51,970 predictions)"
python3 -I verification/verify_main.py "$WORK/root" > "$WORK/main.txt"
check "verify_main.py" "$WORK/main.txt" verification/reference_outputs/verify_main_output.txt

echo "== 3/4 Repeatability controls (13,944 predictions)"
python3 -I verification/verify_repeat.py "$WORK/root" > "$WORK/repeat.txt"
check "verify_repeat.py" "$WORK/repeat.txt" verification/reference_outputs/verify_repeat_output.txt
python3 -I verification/verify_llama_reload.py "$WORK/root" > "$WORK/reload.txt"
check "verify_llama_reload.py" "$WORK/reload.txt" verification/reference_outputs/verify_llama_reload_output.txt

echo "== 4/4 Paper statistics and tables (requires scipy)"
python3 -I analysis/scripts/paper_stats.py "$WORK/root" "$WORK/out" > /dev/null
for f in stats.json hazard.json tab_operating_points.tex tab_equal_bit.tex tab_bf16_ref.tex tab_hazard.tex; do
  check "analysis/output/$f" "$WORK/out/$f" "analysis/output/$f"
done

[ "$status" -eq 0 ] && echo "ALL CHECKS PASSED" || echo "SOME CHECKS FAILED"
exit "$status"
