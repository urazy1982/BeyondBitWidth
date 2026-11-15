# Five independent guard-quantization dataset experiments

## Purpose
Run one checkpoint, **Qwen/Qwen3Guard-Gen-4B**, separately at BF16, bitsandbytes NF4 (4-bit weights, BF16 compute), and Hugging Face FineGrainedFP8, across five different datasets. All experiment rows are frozen before any model inference. The experiments produce **five distinct result folders**, not one pooled task. This is an empirical generalization study, not a proof of literature novelty.

## Dataset registry (do not equate target labels across tasks)

| Identifier | Hugging Face source | Task | Labels |
|---|---|---|---|
| `qwen_thinking` | `Qwen/Qwen3GuardTest`, `thinking` | whole prompt+reasoning response moderation | `label` |
| `wildguard_prompt` | `allenai/wildguardmix`, config `wildguardtest`, split `test` | **prompt** harmfulness only, never response harm | `prompt_harm_label` |
| `toxicchat_prompt` | `lmsys/toxic-chat`, config `toxicchat0124`, split `test` | **prompt** toxicity proxy (different policy) | `toxicity` |
| `xstest_prompt` | `Paul/XSTest`, split `train` | prompt contrast for overblocking and harmful prompts | `label` |
| `beavertails_response` | `PKU-Alignment/BeaverTails`, split `30k_test` | prompt + **response** moderation | `is_safe` |

* WildGuardMix requires agreement to the provider's Hugging Face access terms; accept them in the browser using your own account before running. If unavailable, the sequential runner logs the error, continues other benchmarks, and exits nonzero.
* Toxicity is not identical to policy harmfulness. ToxicChat is a separate **proxy-task** sensitivity analysis.
* XSTest evaluates safety cues and refusal contrasts; classification and refusal behavior are distinct. The model here is **only a classifier**.
* Qwen3GuardTest `thinking` derives prompts from BeaverTails; those two datasets are **not independent at the prompt-source level**. Never call all five independent source distributions.
* Original Qwen pilot636 has a different balanced, length-filtered selection. `qwen_thinking` here selects all eligible examples (default cap 5000) and reruns all three arms. Do **not** combine the Qwen pilot's results with this one as independent observations.
* Selection has a global per-dataset cap of 5000, random sampling without replacement (fixed seed). It is **not** artificially class-balanced. Rows over 1536 input tokens are excluded and counted. Test sizes vary.
* The original benchmark's Safe/Unsafe labels, category granularity, and annotation policy differ from Qwen3Guard's Controversial class. Primary binary mapping is `Safe=>safe`, `Unsafe/Controversial=>unsafe`. Outputs keep all three original labels for later mapping sensitivity analysis.
* Exact raw prompts are preserved in each `selection.json` for reproducibility. Dataset content may contain disturbing material. Keep output folders private; comply with each dataset's license, particularly CC-BY-NC.

## Installation (Ubuntu-Guard)

1. Download `guard_five_datasets.zip` into `C:\guard`.
2. Inside Ubuntu-Guard:

```bash
cd ~/guard-study
unzip -o /mnt/c/guard/guard_five_datasets.zip
cd ~/guard-study/guard_five_datasets
source ~/guard-study/.venv/bin/activate
python -m pip install -U datasets transformers accelerate bitsandbytes safetensors
python -m pip install 'kernels==0.17.2'
python -m unittest -v test_logic.py
```

**Token:** `HF_TOKEN` must already be set in your terminal. Never put an actual token in source files, shell history, research exports, or this chat.

## Separate dataset-by-dataset execution

For the entire campaign, each dataset is completed independently, in the order shown:

```bash
bash run_sequential.sh
```

Or evaluate them **one at a time**:

```bash
python run_five.py --dataset qwen_thinking --out five_results
python run_five.py --dataset wildguard_prompt --out five_results
python run_five.py --dataset toxicchat_prompt --out five_results
python run_five.py --dataset xstest_prompt --out five_results
python run_five.py --dataset beavertails_response --out five_results
python run_five.py --report-only --out five_results
```

A dataset's default command executes BF16 then NF4 then FP8 before starting the next dataset. Each output is checkpointed every 20 rows and resumes after interruption. A rerun never silently replaces an already completed arm.

To prepare all frozen selections before inference:

```bash
python run_five.py --dataset qwen_thinking --prepare-only
```

To run just one condition after selection:

```bash
python run_five.py --dataset qwen_thinking --arm fp8 --out five_results
```

**Important:** Keep the default `--max-input-tokens 1536`, `--max-new-tokens 48`, and seed when resuming existing frozen selections. This runner does not update the source repo revision automatically once frozen, but saves the Hugging Face dataset fingerprint and prompt hashes, and will stop if the tokenizer/template changes within an in-progress selection.

## Artifact structure

```
five_results/
  qwen_thinking/{manifest.json,selection.json,bf16.json,nf4.json,fp8.json,summary.json}
  wildguard_prompt/...
  toxicchat_prompt/...
  xstest_prompt/...
  beavertails_response/...
  cross_dataset_summary.json
  summary_table.csv
  <dataset>.log
```

`manifest.json` stores the selection SHA256, source, source fingerprint, policies and exclusion counts. Per-arm files store raw classifications and incomplete/complete markers. `summary.json` gives metrics and matched flips; `cross_dataset_summary.json` keeps tasks separate. No automatically pooled accuracy/FNR.

## Scientific interpretation

The user pilot's Qwen3Guard 636-example BF16/NF4/FP8 results are **not** re-used as independent data; they are prior exploratory findings. Do not treat a similar result across five datasets using the same checkpoint as cross-model replication. The broader study also requires a separate closest-prior-art review before claiming novelty.

## Environment validation

This project was syntax checked and tested using five local data-free unit tests. It was **not end-to-end executed on RTX 5090** in the authoring environment. Dataset access, software compatibility and the actual runtime must be validated locally. FineGrainedFP8 may require a compatible `kernels` package (0.17.2 in the user's previous successful setup).
