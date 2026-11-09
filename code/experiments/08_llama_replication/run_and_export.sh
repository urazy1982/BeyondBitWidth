#!/usr/bin/env bash
set -Eeuo pipefail
cd "$HOME/guard-study/guard_five_datasets"
source "$HOME/guard-study/.venv/bin/activate"
mkdir -p /mnt/c/Guard/result
EXPORT=/mnt/c/Guard/result
SRC="$HOME/guard-study/llama_guard_replication_results"
backup(){
  code=$?
  echo "Exporting Llama Guard evidence (status=$code)"
  if [[ -d "$SRC" ]]; then
    python3 - "$SRC" "$EXPORT/Guard_Llama3_Replication.zip" <<'PY'
import pathlib,sys,zipfile
src=pathlib.Path(sys.argv[1]);dst=pathlib.Path(sys.argv[2]);tmp=dst.with_suffix('.zip.tmp')
with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
  for p in sorted(src.rglob('*')):
    if p.is_file():z.write(p,str(pathlib.Path('llama_guard_replication_results')/p.relative_to(src)))
with zipfile.ZipFile(tmp) as z:
  if z.testzip() is not None:raise RuntimeError('ZIP integrity failure')
tmp.replace(dst)
print('Verified archive:',dst,'files:',len(zipfile.ZipFile(dst).namelist()))
PY
    cp -f "$SRC/llama_cross_dataset_summary.json" "$EXPORT/llama_cross_dataset_summary.json" 2>/dev/null || true
  fi
  exit "$code"
}
trap backup EXIT
python3 replicate.py --project "$HOME/guard-study/guard_five_datasets" --out "$SRC" --datasets xstest_prompt,beavertails_response --arms bf16,nf4,fp4
