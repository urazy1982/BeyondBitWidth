#!/usr/bin/env bash
set -Eeuo pipefail
cd "$HOME/guard-study/guard_five_datasets"
source "$HOME/guard-study/.venv/bin/activate"
DEST='/mnt/c/Guard/result'; mkdir -p "$DEST"
export_results() {
  code=$?; trap - EXIT
  echo "Exporting INT8 data, status=$code" 
  if [ -d five_results ]; then
    cp -a five_results/. "$DEST/"
    python3 - <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
src=Path('five_results'); dest=Path('/mnt/c/Guard/result/Guard_Results_FP4_INT8.zip')
tmp=dest.with_suffix('.zip.partial')
with ZipFile(tmp,'w',ZIP_DEFLATED,compresslevel=6) as z:
 for p in sorted(src.rglob('*')):
  if p.is_file():z.write(p,Path('five_results')/p.relative_to(src))
with ZipFile(tmp) as z:
 bad=z.testzip()
 if bad:raise RuntimeError('Corrupt zip: '+bad)
tmp.replace(dest)
print('Verified ZIP:',dest)
PY
  fi
  exit "$code"
}
trap export_results EXIT
python3 -m py_compile run_int8.py
python3 run_int8.py --out five_results --dataset all
