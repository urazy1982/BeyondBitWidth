#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT="$HOME/guard-study/guard_five_datasets"
OUT="$HOME/guard-study/guard_repeatability_results"
DEST="/mnt/c/Guard/result"
mkdir -p "$DEST"
cd "$PROJECT"
source "$HOME/guard-study/.venv/bin/activate"
export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
status=0
python3 "$PROJECT/guard_gpu_repeat.py" --project "$PROJECT" --out "$OUT" --per-class 128 --seed 20261108 || status=$?
python3 - "$OUT" "$DEST" <<'PY'
import sys,zipfile
from pathlib import Path
src=Path(sys.argv[1]);target=Path(sys.argv[2])/'Guard_GPU_Repeatability.zip'
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(src.rglob('*')):
        if p.is_file():z.write(p,'guard_repeatability_results/'+str(p.relative_to(src)))
with zipfile.ZipFile(target) as z:
    assert z.testzip() is None
print('Export:',target,'size:',target.stat().st_size)
PY
exit "$status"
