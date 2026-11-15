# Third precision arm — frozen 636-sample pilot

Place `third_precision.py` in `~/guard-study/guard_novelty_pilot/`. **Keep** the existing `pilot636` directory (which should contain `manifest.json`, `selection.json`, `bf16.json`, `nf4.json`, `summary.json`). The script refuses to run if it cannot verify the exact selection and reference predictions. It will not modify the old arms.

```bash
cd ~/guard-study/guard_novelty_pilot
source ~/guard-study/.venv/bin/activate
# Set HF_TOKEN in environment if required; never save the secret in source code.
python third_precision.py --pilot pilot636 --precision fp8
```

The FP8 path requires `transformers.FineGrainedFP8Config` and a compatible FP8 kernel. A failure means the FP8 arm is unsupported in the installed runtime; **do not label a fallback as FP8**. If FP8 is unsupported, run the documented independent INT8 comparison instead:

```bash
python third_precision.py --pilot pilot636 --precision int8
```

The script writes `pilot636/fp8.json` or `pilot636/int8.json` and a `third_precision_<arm>_summary.json`. It does **not** claim to test native NVFP4. It requires no fresh BF16/NF4 runs.

To view the compact third-arm outcome:

```bash
python - <<'PY'
import json
p='pilot636/third_precision_fp8_summary.json'  # change fp8 to int8 if needed
x=json.load(open(p))
print(json.dumps(x['rates'],indent=2))
for name,item in x['comparisons'].items():
 print(name,{k:v for k,v in item.items() if k!='flipped_ids'})
PY
```

Technical limits: model `FineGrainedFP8Config` support is backend/version dependent, including on RTX 5090. INT8 is weight quantization using bitsandbytes, not FP8. The initial 636 were selected from Qwen3GuardTest `thinking` with original prompt-length filtering; this script checks the selection hash and prompt token counts.
