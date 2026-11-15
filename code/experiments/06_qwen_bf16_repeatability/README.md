# Guard GPU repeatability control

Fresh BF16 A/B on an exact, deterministic class-stratified subset of the four existing frozen selections. Both repeats use the same loaded model, single-item batch size and decoding parameters. Original results and frozen selections remain untouched. This is GPU-dependent, unlike the completed CPU reproducibility checks.

**Preparation:** Place `guard_gpu_repeat.py` and `run_gpu_repeat_and_export.sh` in `~/guard-study/guard_five_datasets/`. Ensure the original `run_five.py`, `five_results` and working venv exist. Model downloads may need `HF_TOKEN` exported locally; never store it in an artifact.

**Execute:** `bash run_gpu_repeat_and_export.sh` (or `timeout 3h bash run_gpu_repeat_and_export.sh`).

**Outputs:** `~/guard-study/guard_repeatability_results/` and Windows `C:\Guard\result\Guard_GPU_Repeatability.zip`.

**Interpretation:** `bf16_A_vs_B` is a pure same-session determinism check. `original_vs_repeat_A` and `original_vs_repeat_B` detect historical baseline drift but cannot attribute drift uniquely because the original exact model revision was not recorded. The model commit retrieved for the new run does **not** retrospectively establish the original commit. A complete absence of observed flips on the subset does not prove zero nondeterminism across the full benchmark or all batch sizes.
