# Independent CPU Audit — Guard Precision Study

**Audit date:** 2026-11-08  
**Model:** Qwen/Qwen3Guard-Gen-4B  
**Platforms:** RTX 5090 for original inference; CPU only for this independent audit  
**Source materials:** original and extension experiment source code; `cross_dataset_summary.json`, `fp4_cross_dataset_summary.json`, `int8_cross_dataset_summary.json`; saved library ZIP metadata.

## Executive verdict

**SUMMARY-LEVEL AUDIT PASSED; RAW-RECORD/ARCHIVE AUDIT UNAVAILABLE IN THIS RUNTIME.**

Four datasets (Qwen3GuardTest 753, ToxicChat 5,000, XSTest 450, BeaverTails 3,021) and five precision arms yield 46,120 *reported* classifications. The three summary reports are internally consistent; 2,362 numerical, identity and pairwise assertions were checked without an error. These are *not* equivalent to validating every underlying archived prediction. Repeated attempts to materialize the saved ZIP into the analysis container produced an inaccessible path. Raw selection-hash recomputation, individual record validation, and ZIP CRC verification therefore remain **not performed on the actual research archive**.

## Available CPU tests

| Test group | Result | Scope |
|---|---|---|
| Existing study Python unit tests | **5/5 pass** | Selection hash order, dataset label adapters, paired comparisons and rate computation |
| Additional CPU verifier unit tests | **8/8 pass** | Length/ID checks, unsafe regressions/recoveries, invalid-output rate, parsing, synthetic full five-arm archive and deliberate corruption |
| Multi-report numerical consistency | **PASS: 2,362 assertions** | Four datasets, five arms, 9 saved pairwise comparisons per dataset, cross-arm XOR consistency |
| Exact McNemar tests and Holm adjustment | **Completed** | Four-bit and eight-bit comparison families |
| 95% Wilson CI of flip rates | **Completed** | Eight dataset–format comparisons |
| Paired Wald 95% intervals for differences | **Completed, approximate** | Accuracy and false-negative rate differences |
| Actual ZIP CRC, frozen selection hash recomputation | **NOT DONE** | Source archive contents inaccessible to this CPU runtime |
| Regeneration of model predictions | **NOT DONE** | Requires user GPU / checkpoint, outside CPU scope |

## Accuracy by dataset and precision

| Dataset | BF16 | NF4 | FP4 | FP8 | INT8 |
|---|---:|---:|---:|---:|---:|
| Qwen3GuardTest (n=753) | 84.99% | 85.79% | 84.99% | 84.73% | 85.39% |
| ToxicChat (n=5,000) | 92.06% | 91.70% | 92.20% | 92.10% | 92.00% |
| XSTest (n=450) | 90.44% | 89.11% | 89.33% | 90.22% | 90.00% |
| BeaverTails (n=3,021) | 84.57% | 84.61% | 84.54% | 84.67% | 84.67% |

**Do not pool accuracies or false-negative rates**: Qwen/BeaverTails evaluate response safety; ToxicChat is a prompt toxicity proxy; XSTest tests prompt safety contrasts.

## Independently computed paired inference

**Direction:** differences below are second scheme minus first, so positive accuracy delta favors the second scheme; positive FNR delta is worse safety detection.

| Comparison | Dataset | Flips | Δ accuracy (pp) | 95% paired interval (pp) | exact McNemar p | Holm p (within 4) | Δ FNR (pp) | exact FNR p |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| NF4 → FP4 | Qwen3GuardTest | 14 | -0.797 | [-1.769, 0.175] | 0.180 | 0.539 | +1.572 | 0.227 |
| NF4 → FP4 | ToxicChat | 81 | +0.500 | [0.147, 0.853] | 0.007 | 0.029 | +1.130 | 0.125 |
| NF4 → FP4 | XSTest | 9 | +0.222 | [-1.084, 1.529] | 1.000 | 1.000 | -1.000 | 0.500 |
| NF4 → FP4 | BeaverTails | 48 | -0.066 | [-0.516, 0.383] | 0.885 | 1.000 | +0.462 | 0.115 |
| FP8 → INT8 | Qwen3GuardTest | 11 | +0.664 | [-0.198, 1.526] | 0.227 | 0.906 | -2.201 | 0.039 |
| FP8 → INT8 | ToxicChat | 31 | -0.100 | [-0.318, 0.118] | 0.473 | 1.000 | -0.282 | 1.000 |
| FP8 → INT8 | XSTest | 7 | -0.222 | [-1.374, 0.930] | 1.000 | 1.000 | -1.000 | 0.500 |
| FP8 → INT8 | BeaverTails | 16 | +0.000 | [-0.260, 0.260] | 1.000 | 1.000 | -0.346 | 0.070 |

**Multiplicity qualification.** On ToxicChat, NF4→FP4 accuracy change is +0.50 pp with p=0.00728; Holm adjusted within the 4 planned four-bit accuracy tests gives p=0.0291. When all eight reported four-/eight-bit accuracy tests are considered one exploratory family, Holm-adjusted p=0.0582. No per-dataset FNR differences survive Holm adjustment. Qwen FP8→INT8 FNR improvement has uncorrected p=0.0391; within the four eight-bit FNR tests Holm p=0.1563. Therefore, no broad cross-dataset safety benefit is established.

**Confidence interval caution.** Differences were computed from paired counts, with approximate 95% Wald intervals. For sparse unsafe discordances, rely on exact paired p-values and explicitly describe uncertainty; the Wald intervals are descriptive, not exact tests. Equivalence cannot be inferred from non-significant differences and no accepted equivalence margin was pre-registered.

## Flip rates with 95% Wilson confidence intervals

| Comparison | Dataset | Flip rate | Wilson 95% CI |
|---|---|---:|---:|
| NF4 → FP4 | Qwen3GuardTest | 1.86% | [1.11%, 3.10%] |
| NF4 → FP4 | ToxicChat | 1.62% | [1.31%, 2.01%] |
| NF4 → FP4 | XSTest | 2.00% | [1.06%, 3.76%] |
| NF4 → FP4 | BeaverTails | 1.59% | [1.20%, 2.10%] |
| FP8 → INT8 | Qwen3GuardTest | 1.46% | [0.82%, 2.60%] |
| FP8 → INT8 | ToxicChat | 0.62% | [0.44%, 0.88%] |
| FP8 → INT8 | XSTest | 1.56% | [0.76%, 3.18%] |
| FP8 → INT8 | BeaverTails | 0.53% | [0.33%, 0.86%] |

## Summary-level invariant checks performed

For each of four datasets, these tests ran over the three structured summaries: arm sample sizes, valid/invalid counts, class denominators, exact confusion count arithmetic, accuracy recomputation, pair improvement/regression directions, unsafe miss/recovery directions, flipped-ID list size, flipped-ID uniqueness, consistent reference BF16 classifications, baseline-anchored XOR parity among nine pairwise comparisons, no contradictory class gold across lists, and matching BF16/NF4/FP8 metrics reused by both extensions. All checked invariants passed. The tenth possible pair, FP4-versus-INT8, is absent from saved summaries but its disagreement counts can be inferred from the four BF16-anchored flip sets (below).

| Dataset | Inferred FP4–INT8 disagreements | Rate |
|---|---:|---:|
| Qwen3GuardTest | 11 | 1.46% |
| ToxicChat | 76 | 1.52% |
| XSTest | 13 | 2.89% |
| BeaverTails | 44 | 1.46% |

These inferred disagreement counts are a mathematical consistency check from the published flipped-ID sets, not an independent raw-record count.

## Methodological defects and publication risks

**P0 — Source archive could not be opened for raw-record verification.** The Library ZIP is saved, but the CPU audit container lacks readable bytes at the materialized path. Never claim the original 46,120 per-example records were independently audited until `guard_cpu_verify_raw.py` reports PASS on the actual ZIP.

**P1 — Missing source checkpoint revision.** Both FP4 and INT8 metadata report `model_revision: None`, so checkpoint identity cannot be reconstructed unambiguously from summaries alone. Pin model SHA/commit or compute trusted model weight hashes.

**P1 — NF4 provenance gap.** Original study code explicitly uses `BitsAndBytesConfig(... bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16 ...)` but logged model quantization configuration is `None`. The FP4 and INT8 scripts were patched to accept that missing metadata. Script inspection supports intended NF4 quantization; it is not direct proof of the historical active NF4 codebook.

**P1 — Repeatability baseline absent.** No second BF16 deterministic identical-settings run is in the available evidence. Some flips might be numerical/backend nondeterminism rather than solely weight-codebook changes. A batch-size-changed repeat would test robustness, not pure same-settings repeatability.

**P1 — No second model family.** Four datasets using the same 4B Qwen3Guard checkpoint do not establish cross-architecture generalization.

**P1 — Missing score probabilities.** Current script saves generated labels and text only. Cannot verify threshold transfer, AUROC, calibration error, or confidence-margin mechanisms from stored labels.

**P1 — Dataset selection bias.** Qwen3GuardTest excludes 306 of 1,059 samples for exceeding the 1,536-token input limit. Results apply to the eligible 753 sample subset only.

**P2 — Dataset construct heterogeneity.** ToxicChat `toxicity` is not identical to policy-based unsafe content. XSTest uses the accessible mirror `Paul/XSTest` `train` split; do not present it as an official held-out test without verification. WildGuardTest was not completed due to gated access.

**P2 — Eight-bit comparison is confounded.** FP8 FineGrainedFP8 and bitsandbytes LLM.int8 vary weight quantization algorithms, activation precision, outlier policy, and kernels. `MatMul8bitLt` casts BF16 activations to FP16. It is a deployment-scheme comparison, not a pure number-format intervention.

**P2 — FP4 nomenclature.** bitsandbytes `fp4` is W4A16 and not NVIDIA Blackwell native NVFP4. The manuscript must not conflate these formats.

**P2 — Label parser risks.** The source uses `LABEL_RE.search` over generated output and maps Controversial→unsafe. A malformed output or an incidental `Safety:` phrase in generated text could be misinterpreted; label-token extraction should be independently validated. Existing reports say invalid=0, but raw outputs were not accessible here.

## Publication readiness

**CONDITIONAL GO for a measurement paper, NO-GO for a strong causal-format or universal safety claim at present.** The four-bit comparisons show repeated paired decision changes (~1.6–2.0%), but no consistent, statistically validated increase in missed unsafe examples. ToxicChat gives a detectable overall accuracy gain for FP4 against NF4 under the 4-test Holm family, mainly through reduced false positives. The eight-bit schemes do not yield robust overall accuracy differences. Novelty with respect to the peer-reviewed literature and current Clarivate SCIE JIF Q1 classification remains outside the scope of this file-only CPU audit.

**Reasonable working title:** *Beyond Bit-Width: Dataset-Dependent Safety Decisions under Fixed-Bit-Width Guard-Model Quantization*.

## CPU-only raw verification (optional final certification)

Run on Ubuntu without GPU or ML packages:

```bash
python3 guard_cpu_verify_raw.py /mnt/c/Guard/result/Guard_Results_FP4_INT8.zip \
  --report /mnt/c/Guard/result/guard_raw_verification.json
```

It verifies ZIP CRC, frozen selection SHA256, dataset IDs, all five `finished` flags, full per-example alignment, parsed label consistency, raw confusion matrices, and all stored pairwise summary lists; it also computes the previously omitted FP4–INT8 pair. The script passed eight isolated CPU tests including detection of intentionally corrupted records. Do not substitute synthetic test outcomes for an actual archive pass.

## Inputs and chain of custody

- User’s UrazResearch library archive: `/UrazResearch/Guard_Results_FP4_INT8_2026-11-08.zip` (Library upload confirmed; inaccessible bytes in CPU analysis container).
- Supplied `cross_dataset_summary.json` (initial 3 arms), `fp4_cross_dataset_summary.json`, `int8_cross_dataset_summary.json`.
- Source code from previously generated ZIPs (`run_five.py`, `run_fp4.py`, `run_int8.py`, `test_logic.py`) inspected and tested locally; local patched scripts were not available for byte-for-byte comparison.
- CPU environment: Python 3, SciPy 1.17.0, statsmodels 0.14.6, NumPy 2.3.5.
- Statistics file: `Guard_CPU_Statistical_Results.csv`; verification tool: `guard_cpu_verify_raw.py`.
