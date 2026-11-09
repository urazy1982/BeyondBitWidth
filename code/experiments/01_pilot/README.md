# Fast novelty-signal pilot — safety guard precision

This is an **exploratory feasibility and effect-detection experiment, NOT a proof of academic novelty**.

## Design

- GPU: one NVIDIA RTX 5090 (CUDA).
- Model: official `Qwen/Qwen3Guard-Gen-4B` checkpoint, unchanged.
- Arms: BF16 weights/compute and bitsandbytes NF4 4-bit weights with BF16 compute. **NF4 is not NVFP4**.
- Dataset: `Qwen/Qwen3GuardTest`, `thinking` split; balanced safe/unsafe subset selected with fixed seed from examples fitting the specified token budget.
- Unit: original prompt+assistant message pair. Official Qwen safety chat template retained.
- Deterministic generation. Structured `Safety:` output parser; invalid responses retained.
- `Controversial` is mapped to requiring moderation (`unsafe`) for binary policy, with raw three-class decisions preserved.
- Paired outputs are saved per arm so analysis can run without GPU afterward.

## Setup (Ubuntu / WSL CUDA environment)

Use a **working CUDA-enabled PyTorch installation** appropriate for the local NVIDIA driver. Then:

```bash
python3 -m venv ~/guard-pilot-venv
source ~/guard-pilot-venv/bin/activate
pip install --upgrade pip
pip install 'transformers>=4.51.0' accelerate bitsandbytes datasets safetensors
python -c 'import torch; print(torch.cuda.is_available(),torch.cuda.get_device_name(0))'
```

If PyTorch cannot see the GPU, fix CUDA/WSL installation before proceeding. Don't install a second PyTorch wheel over a working installation unnecessarily.

## Fast dry run

```bash
python quick_pilot.py --n 20 --out quick20
```

## Actual pilot

```bash
python quick_pilot.py --n 200 --out pilot200
```

Optional if first pilot works and effects merit more precision:

```bash
python quick_pilot.py --n 400 --out pilot400
```

## Outputs

`manifest.json`, `selection.json`, `bf16.json`, `nf4.json`, `summary.json`.

## Stop/go criteria

- **Technical STOP:** any invalid-output or model-loading issue that prevents accurate comparison; debug before expanding.
- **Novelty hypothesis weakened:** <1% flips, no directional effect, and intervals consistent with negligible effects. This is *not* evidence the question has been previously published; it is evidence that this particular result may be too small to sustain a paper.
- **Worth deeper investigation:** substantial valid paired flips, especially BF16 correctly detecting an unsafe example that NF4 misses, **even if headline accuracy barely changes**. Inspect examples and compare closest prior studies before interpreting.
- **Do not treat 200 samples as adequate for final inference** or as proof of safety harm.

## Caveats

Qwen3GuardTest is an in-family test set from the guard authors, not independent external validation. Its `thinking` split is specific to long reasoning responses; the length filter changes the target population. A full paper needs separate externally sourced validation, additional models/precisions, uncertainty intervals, hardware provenance, and independent novelty comparison.

Dataset license is CC BY-NC 4.0; respect its usage terms. Model license is Apache-2.0.
