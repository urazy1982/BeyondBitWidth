#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HOME/guard-study/qwen_cache_pinned_targeted_results"
WIN="/mnt/h/GuardReplication"
mkdir -p "$OUT" "$WIN"
if [[ -f "$HOME/guard-study/.venv/bin/activate" ]]; then
  source "$HOME/guard-study/.venv/bin/activate"
else
  echo "ERROR: expected environment ~/guard-study/.venv; edit run.sh if your venv is elsewhere" >&2; exit 2
fi
set +e
python3 -u "$ROOT/replay.py" --root "$ROOT" --out "$OUT" --max-minutes 150 2>&1 | tee "$OUT/replay.log"
STATUS=${PIPESTATUS[0]}
set -e
python3 - "$OUT" "$WIN" <<'PY'
import sys,zipfile
from pathlib import Path
src=Path(sys.argv[1]);win=Path(sys.argv[2]);win.mkdir(parents=True,exist_ok=True)
p=win/'Qwen_Cache_Pinned_Targeted_Results.zip';temp=p.with_name(p.name+'.tmp')
with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
 for f in src.rglob('*'):
  if f.is_file():z.write(f,str(f.relative_to(src)))
temp.replace(p)
with zipfile.ZipFile(p) as z:assert z.testzip() is None
print('WINDOWS RESULT:',p)
for name in ['comparison_summary.json','replay.log']:
 f=src/name
 if f.exists():(win/('targeted_'+name)).write_bytes(f.read_bytes())
PY
exit "$STATUS"
