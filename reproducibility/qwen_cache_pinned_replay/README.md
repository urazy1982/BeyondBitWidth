# Qwen3Guard cached-checkpoint targeted replay (2026-11-15)

**Candidate cached snapshot (pinned in new replay code):** `6ec42827da0c1ff11e7a49dc269d2e810d27e108` (`Qwen/Qwen3Guard-Gen-4B`). Historical model revision was **not logged in original run manifests**, so this is a retrospective corroboration, not definitive proof of historical identity.

A single local cache snapshot had filesystem timestamps from 2026-11-07. The archived FP4 output updates are dated 2026-11-07 to 2026-11-08 UTC. These are filesystem/output timestamps, **not a signed model access log**. The snapshot may have been updated; do not elevate it to proof.

All 152 historical NF4–FP4 binary disagreement inputs, plus 128 controls (16 safe and 16 unsafe unchanged decisions from each of four datasets, selected with a fixed seed), were re-run in both formats: `(152+128)*2=560` predictions. All 560 binary decisions, three-tier predictions, gold labels, and full generated output strings matched the archived original records. The 152 cases were **selected on observed binary disagreements**, making them a sensitive regression test for perturbations, though their closeness to an actual logit margin was **not measured** (no logits recorded). The controls are not a random representative estimate of the full datasets.

Replay environment: NVIDIA RTX 5090, PyTorch `2.11.0+cu128`, Transformers `5.22.2`, bitsandbytes `0.50.2`, CUDA `12.8` (run identity records the exact versions); greedy decoding, max 48 new tokens, same archived input templates. `run.sh` contains machine-specific paths and requires adapting the environment path; `replay.py` explicitly pins `REV` and refuses inference without a matching cached snapshot (`local_files_only=True`). Original experiment runners were not altered.

### CPU reproduction

From the repository root run `bash verify_all.sh`, then `python3 reproducibility/qwen_cache_pinned_replay/verify_replay_cpu.py`. The latter recomputes every replay comparison against *original archived primary prediction records* and verifies frozen sample membership. Requires only Python's standard library.

`results/` stores the new GPU replay output (not a second independent benchmark sample). `frozen/` stores the subset and original reference records, including benchmark text subject to upstream licensing. `replay.py` and `run.sh` are the actual replay scripts. No model weights are included.
