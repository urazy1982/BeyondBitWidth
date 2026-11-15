# Changelog

## v1.2.0 (2026-11-15) — release cited by the manuscript
- **Qwen3Guard checkpoint provenance.** Added `reproducibility/qwen_cache_pinned_replay/`: the single cached
  snapshot `6ec42827da0c1ff11e7a49dc269d2e810d27e108` was reloaded under NF4 and FP4, and all 152 historical
  NF4–FP4 disagreements plus 128 fixed-seed controls were re-run (560 predictions). All 560 verdicts and generated
  texts match the archived records. This corroborates, but cannot prove, the historical revision, which the original
  manifests did not record. Includes the pinned runner, frozen subset, outputs and a standard-library CPU verifier.
- **Analyses used in the paper are complete in one place.** `extra_stats.py` (RQ5–RQ6), `supp_stats.py`
  (supplementary material, including `nf4_fp4_flips.csv`, the released list of flipped inputs) and
  `make_figures_ext.py` (Figs. 2, 5, 8) are included with their reference outputs.
- `verify_all.sh` now has six steps (checksums, primary records, repeatability, paper tables, RQ5–RQ6, supplement);
  `verify_all_cpu_and_replay.sh` additionally verifies the replay.
- `CITATION.cff` points to this repository; `SHA256SUMS.txt` rebuilt.
- Historical records and original experiment runners are unchanged.

## v1.1.0 (2026-11-09)
- Added the extended analyses (RQ5–RQ6), supplementary tables and the codebook, pairwise and power figures.

## v1.0.0 (2026-11-09)
- First public release: raw prediction records (51,970 primary, 13,944 repeat), GPU runners, raw-record
  verifiers and paper statistics.
