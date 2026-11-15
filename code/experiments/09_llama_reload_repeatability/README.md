# Llama Guard 3 — Independent-reload repeatability control

**Purpose.** Certify that the NF4-versus-FP4 decisions reported for Llama Guard 3 8B on XSTest and BeaverTails recur after independently reloading the same *pinned* model commit. This control should be conducted **before** journal submission because the Llama results are the study's largest observed format-dependent differences.

## Primary prespecified design

- Freeze **all 450 XSTest and all 1,500 BeaverTails inputs** already used for the original Llama results. **No resampling.**
- Re-evaluate both NF4 and FP4 using **two independent loads per format**, giving four new model loads and **7,800 fresh predictions**.
- Preserve the original selection files, model revision, task-specific label definitions and deterministic decoding. Each arm/repeat checkpoint every 20 examples.
- For each format, compare fresh A versus B; compare both to the original corresponding arm. Compare fresh NF4 versus fresh FP4 for A and B, as well as the original NF4-versus-FP4 pairs.
- Report exact paired McNemar p values for both accuracy and unsafe-class false negatives; treat the two reload runs as repeatability observations, **not independent scientific replicates** of 1,950 new samples.
- The original run's frozen model revision is loaded exactly. Tokenizer output lengths are checked on the frozen messages before GPU inference.

## Install and run (Ubuntu-Guard)

Save this ZIP as `C:\Guard\Guard_Llama_Reload_Repeatability_Kit.zip`.

```bash
cd ~/guard-study/guard_five_datasets
source ~/guard-study/.venv/bin/activate
unzip -o /mnt/c/Guard/Guard_Llama_Reload_Repeatability_Kit.zip -d .
python3 -m unittest -v test_llama_reload_repeat.py
read -rs -p 'HF token: ' HF_TOKEN; echo; export HF_TOKEN
mkdir -p /mnt/c/Guard/result
set -o pipefail
timeout --signal=TERM --kill-after=60s 10h \
  bash run_llama_reload_and_export.sh \
  2>&1 | tee /mnt/c/Guard/result/llama_reload_repeatability.log
```

**Resume after outage:** rerun exactly the `timeout ... bash run_llama_reload_and_export.sh` command. Existing valid checkpoint records are checked and skipped; a previously completed arm/repeat is not rerun. If a run was interrupted mid-batch, at most the latest 19 unsaved predictions are recomputed.

**Expected outputs:** `C:\Guard\result\Guard_Llama_Reload_Repeatability.zip`, `C:\Guard\result\llama_reload_cross_dataset_summary.json`, plus the logfile. Original archive `Guard_Llama3_Replication.zip` is not modified.

**Important:** On each reload the script verifies the original selection hashes, IDs, gold labels, and original prediction completeness, and requires the original pinned revision. It aborts on invalid new classifications. It will reject any change in key run software versions when resumed. The SHA-256 manifest of original inputs is recorded.

**Interpretation:** If fresh NF4/FP4 within-format disagreements remain zero and the original NF4–FP4 flip IDs recur in both reloads, the observed quantization-format differences are stable under independent reloads. This does **not** control for different back-end implementations, guarantee cross-hardware reproducibility, or make an untested model-family-wide causal claim. There is no need for a new selection or a Qwen rerun here.

**Dependencies:** Requires `replicate.py` supplied in this kit; the existing original experiment data at `~/guard-study/llama_guard_replication_results`; Python env containing PyTorch, Transformers, bitsandbytes, huggingface_hub; gated model permission on Hugging Face. Run on your RTX 5090. The script does not infer that a 10-hour timeout is the expected runtime.
