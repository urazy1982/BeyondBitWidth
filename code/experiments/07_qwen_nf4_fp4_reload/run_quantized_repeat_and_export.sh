#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT="$HOME/guard-study/guard_five_datasets"
OUT="$HOME/guard-study/guard_quantized_repeatability_results"
DEST="/mnt/c/Guard/result"
mkdir -p "$DEST"
cd "$PROJECT"
source "$HOME/guard-study/.venv/bin/activate"
export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
status=0
python3 "$PROJECT/guard_quantized_repeat.py" --project "$PROJECT" --out "$OUT" --per-class 128 --seed 20261108 || status=$?
python3 - "$OUT" "$DEST" <<'PY'
import sys,zipfile
from pathlib import Path
src=Path(sys.argv[1]);dst=Path(sys.argv[2])/'Guard_Quantized_Repeatability.zip'
with zipfile.ZipFile(dst,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in sorted(src.rglob('*')):
        if p.is_file():z.write(p,'guard_quantized_repeatability_results/'+str(p.relative_to(src)))
with zipfile.ZipFile(dst) as z:
    assert z.testzip() is None
print('Saved:',dst,'size:',dst.stat().st_size,flush=True)
PY
if [ -f "$OUT/quantized_repeatability_cross_dataset.json" ]; then
    cp "$OUT/quantized_repeatability_cross_dataset.json" "$DEST/quantized_repeatability_cross_dataset.json"
fi
exit "$status"
