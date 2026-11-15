# NF4 / FP4 GPU repeatability — independent reloads

This study does **not** modify existing `five_results/` data. It reuses exactly the same frozen-selection algorithm as the preceding BF16 repeatability kit (seed 20261108; up to 128 safe + 128 unsafe records per dataset). It performs four independent GPU model loads: NF4 A, NF4 B, FP4 A, FP4 B. The environment and (if available) model/tokenizer revision metadata are logged. Same-format A/B disagreement estimates repeatability; cross-format disagreement is compared between independent fresh trials and the archived NF4/FP4 results. No quantization format is swapped silently.

## Install
Save `Guard_Quantized_Repeatability_Kit.zip` in `C:\Guard`, then inside Ubuntu-Guard:

```bash
cd ~/guard-study/guard_five_datasets
source ~/guard-study/.venv/bin/activate
unzip -o /mnt/c/Guard/Guard_Quantized_Repeatability_Kit.zip -d .
python3 -m py_compile guard_quantized_repeat.py
```

## Run

```bash
mkdir -p /mnt/c/Guard/result
set -o pipefail
timeout --signal=TERM --kill-after=60s 4h \
  bash run_quantized_repeat_and_export.sh \
  2>&1 | tee /mnt/c/Guard/result/quantized_repeatability.log
```

If your Hugging Face account needs authentication, set `HF_TOKEN` in the current shell beforehand. The script checkpoints every 20 examples and resumes partial files. It does not edit existing BF16/NF4/FP4/FP8/INT8 predictions or selections. It exports `Guard_Quantized_Repeatability.zip` and `quantized_repeatability_cross_dataset.json` to `C:\Guard\result`.

## Interpretation limits
Zero decision disagreement in fresh A/B runs supports stability for this selected subset and implementation but cannot establish universal deterministic behavior. Historical agreement also depends on unresolved checkpoint revision provenance. Neither a result nor a null result alone certifies novelty or SCIE Q1 acceptance.
