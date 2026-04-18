# CD38 External Tool Transfer Package

- Zip file: `D:\minimal_ML_test\benchmarks\cd38\external_tool_transfer\cd38_external_tool_inputs_transfer.zip`
- Source package: `D:\minimal_ML_test\benchmarks\cd38\external_tool_inputs`
- Files packaged: `24`
- Total source bytes: `2741778`
- Include existing outputs: `False`
- Next-run runbook refresh: `success`

## Use

Send the zip to a machine with P2Rank/fpocket or unzip it in WSL/Linux.
The generated preflight workflow resolves package-local paths first, so the transfer zip is not tied to absolute paths from the original project.
Before returning outputs, check `cd38_external_tool_return_checklist.md` so every expected PDB/method output is present.
If `cd38_external_tool_next_run.md` is present, it is the shortest current runbook: it only includes missing benchmark actions, not every reproducibility template.

Windows:

```powershell
run_cd38_external_next_benchmark.ps1
```

Linux / WSL:

```bash
chmod +x run_cd38_external_next_benchmark.sh
./run_cd38_external_next_benchmark.sh
```

If you intentionally need every generated template row, use `run_p2rank_templates.*` and `run_fpocket_templates.*` instead.

After running external tools, copy `p2rank_outputs/` and `fpocket_runs/*/*_out/` back into the original package directory, or use `python import_cd38_external_tool_outputs.py --source <returned_zip_or_dir>`.

A shorter path is `python finalize_cd38_external_benchmark.py --import_source <returned_zip_or_dir> --run_discovered`, which imports the returned outputs before readiness checks.
If candidate output files are 0, first check the import report `source_diagnosis`, then inspect `cd38_external_tool_output_import_scan.csv` for ignored files and `cd38_external_tool_output_import_coverage.csv` for missing PDB/method pairs.

If you imported outputs separately, then run `python finalize_cd38_external_benchmark.py --run_discovered`.
