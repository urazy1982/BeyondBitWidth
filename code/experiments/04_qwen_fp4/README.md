# FP4 extension for the frozen five-dataset Qwen3Guard campaign

This is a **fourth arm**, not a replacement for the original runner. It performs bitsandbytes FP4 weight quantization (W4A16; BF16 compute) and compares it with the original bitsandbytes NF4 on exactly the same frozen selections. It **does not use or imply NVIDIA NVFP4**.

Prerequisites: `~/guard-study/guard_five_datasets` contains `run_five.py` and its existing `five_results/{dataset}/selection.json`, `manifest.json`, and complete `nf4.json` for four datasets. The earlier BF16/NF4/FP8 arm outputs are never changed. WildGuard is skipped if still gated.

Download the ZIP to `C:\Guard`, then in Ubuntu-Guard:

```bash
cd ~/guard-study/guard_five_datasets
source ~/guard-study/.venv/bin/activate
unzip -o /mnt/c/Guard/guard_fp4_extension.zip -d .
python3 -m py_compile run_fp4.py
bash run_fp4_and_export.sh
```

Your preexisting `HF_TOKEN` environment variable can be used, but do not put the token in code or the exported files. If you need to set it: `read -rs -p 'HF token: ' HF_TOKEN; echo; export HF_TOKEN` (does not echo/store the token in shell history).

Outputs: new `fp4.json` and `fp4_comparison.json` in each completed dataset folder; new `five_results/fp4_cross_dataset_summary.json` and `fp4_summary_table.csv`. All results copied and verified into `C:\Guard\result\Guard_Results_FP4.zip` on success **or error**. Incomplete FP4 records are checkpointed every 20 rows, with stable IDs, and reruns resume them.

**Scientific limits:** Comparing NF4/FP4 holds nominal weight precision, bnb 4-bit compute dtype, double quantization and model checkpoint fixed; actual codebook, numerical format, and possibly kernel paths differ. This is more controlled than NF4 versus FP8 but still does not isolate every hardware implementation variable. Predictions are deterministic-decoding observations, not a runtime noise floor. Statistical significance, task mapping and independent model generalization remain to be evaluated. No claims of novelty or publication guarantees follow automatically.
