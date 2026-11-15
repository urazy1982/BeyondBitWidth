# Beyond Bit-Width: Equal-Bit-Width Quantization and the Decisions of LLM Guard Models

Author: Uraz Yavanoğlu, Department of Computer Engineering, Gazi University, Ankara, Türkiye
([ORCID 0000-0001-8358-8150](https://orcid.org/0000-0001-8358-8150)).

> U. Yavanoglu, *"Beyond Bit-Width: Model-Dependent Safety Risk under Fixed-Precision Quantization of LLM Guardrails"* submitted to *IEEE Access* (IEEE), 2026.

This is release **v1.2.0**, the version cited in the manuscript.

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
                                           (paper_stats.py: RQ1-RQ4; extra_stats.py: RQ5-RQ6; supp_stats.py: supplement;
                                            make_figure*.py: figures)
analysis/output/                           generated statistics, LaTeX tables, figures, nf4_fp4_flips.csv
reproducibility/qwen_cache_pinned_replay/  retrospective Qwen3Guard checkpoint-provenance replay (560 predictions)
verify_all.sh                              one-command reproduction of every reported number (CPU)
verify_all_cpu_and_replay.sh               verify_all.sh plus verification of the provenance replay
SHA256SUMS.txt                             checksums of every file
CHANGELOG.md                               release history
```
`results/README.md` documents the fields of every file.

## Reproduce the reported numbers (CPU, about one minute)
```bash
python3 -m pip install -r requirements.txt   # scipy and matplotlib only
bash verify_all_cpu_and_replay.sh
```
`verify_all.sh` checks the checksums, re-derives every label from the generated text, recomputes the selection hashes, and recomputes all metrics, exact McNemar tests, Holm corrections, repeatability comparisons, the RQ5–RQ6 analyses and the supplementary tables. It compares everything with the stored reference outputs and ends with `ALL CHECKS PASSED`. `verify_all_cpu_and_replay.sh` runs it and then checks the provenance replay against the original records; its last line is `FULL CPU RELEASE VERIFICATION PASS`.

## Where each result in the paper comes from
| Result in the manuscript | Script |
|---|---|
| Tables 5, 6, 11, 12; RQ1–RQ4 statistics | `analysis/scripts/paper_stats.py` |
| Table 7 (repeatability) | `verification/verify_repeat.py`, `verification/verify_llama_reload.py` |
| Tables 8, 9, 10; RQ5–RQ6 statistics | `analysis/scripts/extra_stats.py` |
| Figs. 4, 6, 7 | `analysis/scripts/make_figure.py`, `analysis/scripts/make_figures_extra.py` |
| Figs. 2, 5, 8 | `analysis/scripts/make_figures_ext.py` |
| Record-level integrity checks | `verification/verify_main.py` |
| Qwen3Guard checkpoint replay (Sections IV-G, V-D) | `reproducibility/qwen_cache_pinned_replay/verify_replay_cpu.py` |
| Supplementary Tables S1–S9 and the list of flipped inputs | `analysis/scripts/supp_stats.py` |

## Re-run the GPU experiments
Each folder in `code/experiments/` has its own README with exact commands. Environment used:

- one NVIDIA GeForce RTX 5090, CUDA 12.8
- PyTorch 2.11.0+cu128, Transformers 5.22.2, bitsandbytes 0.50.2
- greedy decoding, batch size 1

The Llama Guard 3 runs pin Hugging Face revision `7327bd9f6efbbe6101dc6cc4736302b3cbb6e425` and require accepting Meta's license on Hugging Face (`HF_TOKEN`). The original Qwen3Guard runs did not record a checkpoint revision. A single local Hugging Face cache snapshot, `6ec42827da0c1ff11e7a49dc269d2e810d27e108` (dated November 7, 2026), was recovered. A subsequent **pinned targeted replay** of 152 original NF4/FP4 disagreements and 128 frozen controls (280 inputs × two formats = 560 predictions) exactly matched both verdicts and generated text in archived records. This strongly corroborates, but cannot conclusively prove, the historical revision. The added replay runner explicitly pins that full commit; the original archived experiment runners are left intact for provenance. See `reproducibility/qwen_cache_pinned_replay/`. Model weights are not redistributed. Fresh runs on other GPUs or library versions may flip different individual inputs; the repeatability claims apply to the stack above.

## Licenses
- **Code** (`code/`, `verification/`, `analysis/scripts/`, `verify_all.sh`): MIT, see `LICENSE`.
- **Data** (`results/`, `analysis/output/`, `verification/reference_outputs/`): CC BY-NC 4.0, see `LICENSE-DATA`. The selection files contain benchmark text from ToxicChat, BeaverTails and Qwen3GuardTest (each CC BY-NC 4.0) and XSTest (CC BY 4.0); attribution to those datasets is required, and non-commercial terms apply. Model outputs were produced with Llama Guard 3 (Llama 3.1 Community License) and Qwen3Guard.

**Content warning.** The benchmark inputs include harmful, offensive and toxic text by design.

## Citation
See `CITATION.cff`. Please also cite the benchmark datasets and guard models listed there.

## Funding and disclosure
This work was financially supported by UAYTECH Corporation, which also provided the GPU workstation. The author is the owner of UAYTECH Corporation.

## Data availability and pinned checkpoint
The archived records, verification scripts, full commit ID, frozen replay subset, replay-generated text and per-file SHA-256 checksums are distributed in this repository. The recovered Qwen checkpoint is **corroborated retrospectively**; it is not recorded in the historical manifests. The Llama Guard 3 checkpoint is pinned in its original experiment code. The CPU verification works without GPUs, benchmark downloads, or model weights.

## Development note
The author designed and ran all experiments. AI assistants (Anthropic Claude,
OpenAI ChatGPT) were used under the author's direction only for partial code validation and 
audit checks.
