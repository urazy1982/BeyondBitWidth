# Publication-Level Validation Attempt — Guard Quantization Study

**Audit date:** 2026-11-08  
**Status:** **PARTIAL — NOT PUBLICATION-CERTIFIED**  
**Scope:** Qwen/Qwen3Guard-Gen-4B, 4 datasets, BF16/NF4/FP4/FP8/INT8, original RTX 5090 inference; CPU-only independent validation.

## Executive assessment

Three requested workstreams were performed as far as the available evidence allows:

| Workstream | Status | Precise interpretation |
|---|---|---|
| Raw-record certification | **BLOCKED** | The original ZIP is registered in `/UrazResearch/Guard_Results_FP4_INT8_2026-11-08.zip` and the file service reports a materialized file path, but its bytes are not visible in the current CPU runtime. Therefore CRC, source selection hashes, all 46,120 individual predictions, and stored label parsing **could not be independently checked**. Summary-level checks passed in the previous audit. |
| CPU analysis repeatability | **PASS** | Recalculated the full statistical CSV twice; bytewise SHA-256 identical on both runs. Re-ran the original five unit tests and the eight verifier unit tests: all 13 passed. |
| GPU inference repeatability | **NOT ESTABLISHED** | No independent same-model, same-input, same-setting BF16 rerun is present. CPU repeatability does **not** imply GPU determinism. Different runs over 636 vs 753 selected items cannot substitute for this control. |
| Closest-prior-art investigation | **COMPLETED as a targeted search, not an exhaustive systematic review** | Substantial adjacent literature exists, including same-bit-width format selection and quantized-safety evaluations. No exact duplicate of the multi-dataset generative guard NF4-vs-FP4 paired protocol was identified in the targeted search, but **a never-studied claim cannot be certified**. |

The user requested publication-level validation, but completion of the full certification **cannot honestly be claimed**. The missing raw-record check and same-setting GPU repeat remain open.

## CPU code test results

- `guard_five_datasets/test_logic.py`: **5/5 PASS**. Includes source labels and paired logic.
- `test_cpu_audit.py`: **8/8 PASS**. Includes synthetically complete five-arm archive, intentional corruption rejection, ID and alignment tests, parser and rates.
- Statistical analysis script (`compute_guard_stats.py`) run twice with identical output hash: `991d70ff8747ae395098dd5f8756fa37b69e558b49e2856ef3427a942823d265`.
- Earlier summary-level invariant checks of three uploaded JSON reports: **2,362 assertions PASS** (n, metrics, directional paired counts, mutually consistent flipped-ID sets, baseline-relative pair parity). This is not equivalent to raw per-example verification.

## Primary empirical findings

**Do not pool all four tasks into one safety FNR.** The studies use distinct outcome constructs: reasoning-response safety, ToxicChat prompt toxicity proxy, XSTest prompt safety contrasts, and BeaverTails response safety.

### Fixed-nominal-four-bit NF4 to bitsandbytes FP4

| Dataset | N | Flips | Δaccuracy (percentage points) | McNemar exact p | ΔFNR (percentage points) |
|---|---:|---:|---:|---:|---:|
| Qwen3GuardTest | 753 | 14 | −0.797 | 0.17957 | +1.572 |
| ToxicChat | 5,000 | 81 | +0.500 | 0.00728 | +1.130 |
| XSTest | 450 | 9 | +0.222 | 1.00000 | −1.000 |
| BeaverTails | 3,021 | 48 | −0.066 | 0.88543 | +0.462 |

ToxicChat is significant for accuracy using Holm correction across the four pre-specified four-bit dataset tests (adjusted p=0.0291). It is **not** significant when the eight four- and eight-bit overall-accuracy comparisons are treated as one exploratory family (adjusted p=0.0582). The ToxicChat change primarily reflects false-positive corrections, with missed unsafe cases rising from six to ten. **None of the four FNR changes is significant at 0.05 under the exact paired test.**

### FP8 to bitsandbytes LLM.int8

| Dataset | N | Flips | Δaccuracy (percentage points) | McNemar exact p | ΔFNR (percentage points) |
|---|---:|---:|---:|---:|---:|
| Qwen3GuardTest | 753 | 11 | +0.664 | 0.22656 | −2.201 |
| ToxicChat | 5,000 | 31 | −0.100 | 0.47313 | −0.282 |
| XSTest | 450 | 7 | −0.222 | 1.00000 | −1.000 |
| BeaverTails | 3,021 | 16 | +0.000 | 1.00000 | −0.346 |

All accuracy comparisons are nonsignificant. Qwen's unadjusted FNR p=0.0391, rising to 0.1563 after Holm correction over four FNR comparisons. **LLM.int8 mixed-precision outlier handling and activation casting differ materially from FineGrainedFP8; this is an algorithm/implementation comparison, not an isolated FP-vs-INT representation test.**

**No equivalence claim is justified:** nonsignificance and low flip rates are not proof of equivalence, and a practically acceptable margin was not prespecified.

## Repeatability assessment

**What was executed:** CPU unit tests and statistical recomputation using the prior reported paired counts (both reruns produced byte-identical CSVs). **What was not executed:** GPU BF16 inference again, same-setting repetition for quantized inference, or comparisons of independent duplicate raw model outputs. Those require checkpoint weights and local GPU environment. A change in batch size would test deployment robustness, not pure repeatability.

The earlier 636-example and 753-example Qwen thinking assessments had the same count of 318 eligible unsafe items and similar error patterns, but selection hashes differ and per-example pair equivalence is not verified, so treating them as a controlled repeat would be invalid.

## Novelty / closest-prior-art evaluation (web checked Nov 8, 2026)

| Prior work | What it establishes | Relation to this project |
|---|---|---|
| Zhang et al., **Integer or Floating Point? New Outlooks for Low-Bit Quantization on Large Language Models** (2023), https://arxiv.org/abs/2305.12356 | Direct same-bit-width integer/FP quantization comparisons and MoFQ | Precludes generic “first same-bit-width format comparison” novelty claim; not dedicated guard moderation. |
| Kharinaev et al., **Investigating the Impact of Quantization Methods on the Safety and Reliability of Large Language Models**, IEEE Access (published 15 June 2026), DOI:10.1109/ACCESS.2026.3703899 | 66 variants, four safety benchmarks, safety performance varies with quantization algorithm | Strong adjacent study, but focuses on safety behavior of general LLMs, not controlled dedicated guard classifier decisions. |
| **Qwen3Guard Technical Report** (Oct 2025), https://arxiv.org/abs/2510.14276 | Three-way guard labels, guard model family, evaluation tasks | Model/methodological baseline; not a systematic quantization-format audit. |
| **Qwen3Guard-Stream-4B-bnb-fp4** artifact, https://huggingface.co/iskiai/Qwen3Guard-Stream-4B-bnb-fp4 | Quantized dedicated Qwen3Guard-Stream FP4 with evaluated BeaverTails performance | Direct counterexample to “no guard quantization studied”; distinct architecture, not NF4/FP4 multi-dataset generative-guard comparison; non-peer-reviewed artifact. |
| **Quantization Breaks the Format**, EMNLP Industry 2026, https://aclanthology.org/2026.emnlp-industry.37/ | Safety vulnerability changes across GGUF quantization levels; differing evaluation metrics | Quantized deployment safety and metric divergence already active; not matched guard detector schemes. |
| Hong et al., **Guard Models Are Overconfident Where Base Models Are Uncertain**, Findings EMNLP 2026, https://aclanthology.org/2026.findings-emnlp.2276/ | Guard confidence calibration can fail under adversarial attack | Adjacent calibration problem; not quantization-controlled. Raw score collection would be needed to claim a different calibration mechanism. |
| **On Jailbreaking Quantized Language Models Through Fault Injection Attacks**, ACM GLSVLSI 2025, DOI:10.1145/3716368.3735302 | Safety evaluator variants FP8/INT8/INT4 appear as part of safety attack evaluation | Relevant methodology but not matched guard-precision investigation. |

**Defensible novelty statement:** “We present a paired, multi-task empirical characterization of the decision changes caused by equal-nominal-weight-bit NF4 and bitsandbytes FP4 quantization of a dedicated generative safety guard.” This is a description of the currently available evidence, **not an independently certified claim of first publication**.

**Nondefensible novelty statements:** “The first comparison of numeric formats at a fixed bit-width”; “the first quantization study of guard models”; “safety degrades consistently as precision decreases”; “Blackwell NVFP4 native-kernel study”; “we proved bit-width-independent format causality.”

## Remaining publication blockers

1. **P0 Raw evidence audit:** on user's Ubuntu run `python3 guard_cpu_verify_raw.py /mnt/c/Guard/result/Guard_Results_FP4_INT8.zip --report /mnt/c/Guard/result/guard_raw_verification.json`. Confirm `status=PASS`, `datasets_verified=4`, `records_verified=46120`. This is **CPU only**.
2. **P0 True inference repeatability:** must rerun a frozen common sample with identical BF16 settings (and preferably quantized settings) on a compatible GPU; impossible to certify from the current summaries alone.
3. **P1 Checkpoint provenance:** record exact HF revision / immutable weight hashes; do not pretend `model_revision: None` is sufficient.
4. **P1 NF4 provenance:** model was explicitly loaded in NF4 by inspected source code, but original `quantization_config` was logged as `None`; runtime codebook was not independently authenticated by the exported metadata.
5. **P1 Label and score instrumentation:** generated labels were parsed by regex; no label logprobs for margin, AUROC, or threshold calibration studies. Verify Safe/Controversial/Unsafe mapping against official protocol.
6. **P1 External validity:** only one 4B generative guard checkpoint; a second model family or very strong scope limitation is necessary for generalized claims.
7. **P2 Dataset definition:** exclude 306 long reasoning samples; ToxicChat is a toxicity proxy; XSTest is from `Paul/XSTest` train split; WildGuardTest access denied.
8. **P2 Novelty coverage:** targeted source search done, not an exhaustive preprint/patent/citation network audit; Q1 status of any proposed venue must be verified in Clarivate JCR directly.

## Final decision

**CPU analysis reproducibility: PASS.**  
**Summary internal-consistency verification: PASS.**  
**Actual research archive raw-record certification: NOT COMPLETED (unavailable bytes in audit runtime).**  
**GPU inference repeatability: NOT COMPLETED (no duplicate inference evidence).**  
**Literature novelty: PARTIALLY DIFFERENTIATED, NOT CERTIFIED FIRST.**  
**Publication-ready as a broad causal/unique Q1 claim: NO.**  

A paper framed as a carefully scoped measurement study may be viable after evidence validation and stronger external replication. Do not claim publication-level certification is complete.
