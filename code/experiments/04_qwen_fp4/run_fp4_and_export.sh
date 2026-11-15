#!/usr/bin/env bash
# Run inside ~/guard-study/guard_five_datasets, with FP4 extension copied alongside run_five.py.
set -Eeuo pipefail
cd "$HOME/guard-study/guard_five_datasets"
source "$HOME/guard-study/.venv/bin/activate"
DEST='/mnt/c/Guard/result'
mkdir -p "$DEST"
export_results() {
  local code=$?
  trap - EXIT
  echo "Exporting current results to $DEST (exit status $code)"
  if [ -d five_results ]; then
    cp -a five_results/. "$DEST/"
    python3 - <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from datetime import datetime,timezone
src=Path('five_results')
zip_path=Path('/mnt/c/Guard/result/Guard_Results_FP4.zip')
tmp=zip_path.with_suffix('.zip.partial')
with ZipFile(tmp,'w',ZIP_DEFLATED,compresslevel=6) as z:
    for path in sorted(src.rglob('*')):
        if path.is_file():z.write(path,Path('five_results')/path.relative_to(src))
with ZipFile(tmp) as z:
    bad=z.testzip()
    if bad:raise RuntimeError('Corrupt zip entry: '+bad)
tmp.replace(zip_path)
print('Verified archive:',zip_path,'size_MB:',round(zip_path.stat().st_size/2**20,2))
PY
  fi
  exit "$code"
}
trap export_results EXIT
python3 -m py_compile run_fp4.py
python3 -m unittest -v test_logic.py
python3 run_fp4.py --out five_results --dataset all
