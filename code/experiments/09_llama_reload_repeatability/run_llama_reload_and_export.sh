#!/usr/bin/env bash
set -Eeuo pipefail
cd "$HOME/guard-study/guard_five_datasets"
source "$HOME/guard-study/.venv/bin/activate"
SRC="$HOME/guard-study/llama_guard_reload_repeatability"
DST="/mnt/c/Guard/result"
mkdir -p "$DST"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export HF_HUB_DISABLE_TELEMETRY=1

# Always export whatever completed, including checkpoints on timeout or SIGINT.
export_evidence(){
 status=$?
 trap - EXIT
 echo "Exporting reload experiment; exit status=$status"
 if [ -d "$SRC" ]; then
   python3 - "$SRC" "$DST/Guard_Llama_Reload_Repeatability.zip" <<'PY'
from pathlib import Path
import sys,zipfile
src=Path(sys.argv[1]);dest=Path(sys.argv[2]);tmp=dest.with_suffix('.zip.tmp')
with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
    for f in sorted(src.rglob('*')):
        if f.is_file():z.write(f,arcname=str(Path('llama_guard_reload_repeatability')/f.relative_to(src)))
with zipfile.ZipFile(tmp) as z:
    assert z.testzip() is None, 'Bad ZIP member'
tmp.replace(dest)
print('Saved',dest,'bytes',dest.stat().st_size)
PY
   cp -f "$SRC/llama_reload_cross_dataset_summary.json" "$DST/llama_reload_cross_dataset_summary.json" 2>/dev/null || true
 fi
 exit "$status"
}
trap export_evidence EXIT
python3 llama_reload_repeat.py --src "$HOME/guard-study/llama_guard_replication_results" --out "$SRC"
