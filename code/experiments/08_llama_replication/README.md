# Independent guard-family replication: Llama Guard 3 8B

Uses frozen Qwen study selections from `~/guard-study/guard_five_datasets/five_results`, without overwriting them. Evaluates XSTest prompt-safety contrasts and BeaverTails response safety **separately**, with BF16, bitsandbytes NF4, and bitsandbytes FP4. The outcome is **within-Llama** format sensitivity; do not compare raw model accuracies as if Qwen and Llama used the same hazard policy. The XSTest source is a `train` split in the Paul mirror, not independently held-out.

**Model access:** `meta-llama/Llama-Guard-3-8B` is gated. Accept terms on Hugging Face before running. The preflight fails rather than silently substituting another checkpoint.

Commands inside Ubuntu-Guard:

```bash
cd ~/guard-study/guard_five_datasets
source ~/guard-study/.venv/bin/activate
unzip -o /mnt/c/Guard/Guard_Llama3_Replication_Kit.zip -d .
python3 -m unittest -v test_replication.py
read -rs -p 'HF token: ' HF_TOKEN; echo; export HF_TOKEN
python3 replicate.py --smoke --arms bf16 # separate smoke output, 2 items per dataset
# Only after smoke succeeds:
mkdir -p /mnt/c/Guard/result
set -o pipefail
timeout --signal=TERM --kill-after=60s 8h bash run_and_export.sh 2>&1 | tee /mnt/c/Guard/result/llama_replication.log
```

Results: `~/guard-study/llama_guard_replication_results/`. Export: `C:\Guard\result\Guard_Llama3_Replication.zip`, `llama_cross_dataset_summary.json`, and `llama_replication.log`. Each arm checkpoints after 20 predictions and resumes. Source selection/hash and model revision are frozen. No exact-equality inference guarantee from `do_sample=False`; inspect repeatability separately.

The script does not perform a direct independent model-reload repeatability run for Llama; it covers replication of the **research hypothesis** across model families, not repeatability within the new family. Model weights are not exported.

Literature: https://huggingface.co/meta-llama/Llama-Guard-3-8B and https://dev.meta.ai/llama/docs/model-cards-and-prompt-formats/llama-guard-3
