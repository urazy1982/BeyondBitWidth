# INT8 add-on to frozen 4-dataset guard experiment

Run after the existing FP4 process completes; never run two GPU arms concurrently.
Place the files beside `run_five.py` and `run_fp4.py`, in `~/guard-study/guard_five_datasets`.

Commands:

```bash
cd ~/guard-study/guard_five_datasets
source ~/guard-study/.venv/bin/activate
unzip -o /mnt/c/Guard/guard_int8_extension.zip -d .
python3 -m py_compile run_int8.py
bash run_int8_and_export.sh
```

The INT8 implementation uses bitsandbytes LLM.int8 with threshold 6.0, not a pure INT8-vs-FP8 numeric-format control. It reads the original selection and NF4 outputs and refuses to rerun or overwrite them. INT8 predictions are saved to `five_results/<dataset>/int8.json` with periodic checkpoints. The export produces `C:\Guard\result\Guard_Results_FP4_INT8.zip` after success or failure. It may include partial results on failure. Never claim an INT8 arm has completed unless its saved record has `finished: true` and contains every selected example.

No actual GPU run was done when preparing this extension; syntax validated only.
