# Beyond Bit-Width: Equal-Bit-Width Quantization and the Decisions of LLM Guard Models

U. Yavanoglu, *"Beyond Bit-Width: Model-Dependent Safety Risk under Fixed-Precision Quantization of LLM Guardrails"* Submitted to REDACTED, 2026

**Question.** Do two deployments of the same guard model with the same nominal bit width make the same safe/unsafe decisions?

**Answer, on the tested models.** No.
- **Disagreement:** bitsandbytes NF4 and FP4 (identical 4-bit storage) disagree on 1.6–2.8% of decisions in every model–dataset cell.
- **Llama Guard 3 8B:** FP4 newly misses 8 of 200 unsafe XSTest prompts (0 the other way) and 32 of 750 unsafe BeaverTails responses (1 the other way). The false-negative rate rises by 4.0 and 4.1 points while accuracy falls by less than 2 points.
- **Qwen3Guard-Gen-4B:** the changes go both ways.
- **Repeatability:** independent model reloads reproduce every changed decision exactly.

| | Qwen3Guard-Gen-4B | Llama Guard 3 8B |
|---|---|---|
| Deployments | BF16, NF4, FP4, FP8, INT8 | BF16, NF4, FP4 |
| Datasets | Qwen3GuardTest-Think, ToxicChat, XSTest, BeaverTails | XSTest, BeaverTails (balanced) |
| Inputs / decisions | 9,224 / 46,120 | 1,950 / 5,850 |
| Repeat predictions | 6,144 | 7,800 |

## Repository layout
```
results/                                   raw per-record predictions (CC BY-NC 4.0, see below)
  five_results/                            Qwen3Guard: 4 datasets x 5 deployments, frozen selections, manifests, logs
  llama_guard_replication_results/         Llama Guard 3: 2 datasets x 3 deployments
  guard_repeatability_results/             Qwen BF16 repeat runs (same loaded model)
  guard_quantized_repeatability_results/   Qwen NF4/FP4 independent-reload runs
  llama_guard_reload_repeatability/        Llama NF4/FP4 independent-reload runs (+ input hashes)
code/experiments/                          GPU runners, in the order they were run (01 pilot ... 09 Llama reload)
verification/                              raw-record verifiers and their reference outputs
  earlier_cpu_audit/                       first independent audit (2026-11-08), superseded by verification/
analysis/scripts/                          statistics, tables and figures used in the paper
analysis/output/                           generated statistics (stats.json), LaTeX tables, figures
verify_all.sh                              one-command reproduction of every reported number (CPU)
SHA256SUMS.txt                             checksums of every file
```
`results/README.md` documents the fields of every file.

## Reproduce the reported numbers (CPU, about one minute)
```bash
python3 -m pip install -r requirements.txt   # scipy and matplotlib only
bash verify_all.sh
```
`verify_all.sh` checks the checksums, re-derives every label from the generated text, recomputes the selection hashes, and recomputes all metrics, exact McNemar tests, Holm corrections, repeatability comparisons and paper tables. It then compares everything with the stored reference outputs, and the last line should read `ALL CHECKS PASSED`.

## Re-run the GPU experiments
Each folder in `code/experiments/` has its own README with exact commands. Environment used:

- one NVIDIA GeForce RTX 5090, CUDA 12.8
- PyTorch 2.11.0+cu128, Transformers 5.22.2, bitsandbytes 0.50.2
- greedy decoding, batch size 1

The Llama Guard 3 runs pin Hugging Face revision `7327bd9f6efbbe6101dc6cc4736302b3cbb6e425` and require accepting Meta's license on Hugging Face (`HF_TOKEN`). The Qwen3Guard revision was not recorded in the original runs. Model weights are not redistributed. Fresh runs on other GPUs or library versions may flip different individual inputs; the repeatability claims apply to the stack above.

## Licenses
- **Code** (`code/`, `verification/`, `analysis/scripts/`, `verify_all.sh`): MIT, see `LICENSE`.
- **Data** (`results/`, `analysis/output/`, `verification/reference_outputs/`): CC BY-NC 4.0, see `LICENSE-DATA`. The selection files contain benchmark text from ToxicChat, BeaverTails and Qwen3GuardTest (each CC BY-NC 4.0) and XSTest (CC BY 4.0); attribution to those datasets is required, and non-commercial terms apply. Model outputs were produced with Llama Guard 3 (Llama 3.1 Community License) and Qwen3Guard.

**Content warning.** The benchmark inputs include harmful, offensive and toxic text by design.

## Citation
See `CITATION.cff`. Please also cite the benchmark datasets and guard models listed there.

## Funding and disclosure
This work was financially supported by UAYTECH Corporation, which also provided the GPU workstation. The author is the owner of UAYTECH Corporation.

## AI Assistance Disclosure

OpenAI GPT and Anthropic Claude models assisted with code review, debugging, and validation. All research findings and conclusions remain the responsibility of the author.
