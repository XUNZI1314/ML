# CD38 External Tool Output Import

- Source: `D:\minimal_ML_test\benchmarks\cd38\external_tool_transfer\cd38_external_tool_inputs_transfer.zip`
- Source type: `zip`
- Source diagnosis: `input_package_without_external_outputs`
- Diagnostic message: This looks like the original external-tool input package, not a completed return package. Run P2Rank/fpocket first, then return p2rank_outputs/ and fpocket_runs/*/*_out/.
- Source files scanned: `19`
- Package directory: `D:\minimal_ML_test\benchmarks\cd38\external_tool_inputs`
- Dry run: `True`
- Overwrite: `False`
- Candidate output files: `0`
- Expected outputs ready from return: `0/6`
- Missing expected outputs: `6`
- Repair actions: `6`
- Unexpected returned outputs: `0`
- Ignored files: `19`
- Imported files: `0`
- Skipped existing files: `0`

## Source Top-Level Entries

- `external_tool_inputs`

## Scan Status Counts

- `ignored`: `19`

## Ignored Reason Counts

- `fpocket_runs_without_out_dir`: `3`
- `not_p2rank_or_fpocket_output`: `3`
- `too_shallow`: `13`

## Role Counts

- No importable P2Rank/fpocket output files were found.

## Expected Coverage

- Ready expected outputs: `0/6`
- Missing: `3ROP:p2rank, 3ROP:fpocket, 4OGW:p2rank, 4OGW:fpocket, 3F6Y:p2rank, 3F6Y:fpocket`

## Next Actions

- This appears to be the original transfer/input package. Run P2Rank/fpocket on it first, then import the returned outputs.
- Expected returned outputs: `p2rank_outputs/<PDB>/*_predictions.csv` and `fpocket_runs/<PDB>/<PDB>_out/pockets/pocket*_atm.pdb`.
- Open the repair plan CSV for exact missing paths: `D:\minimal_ML_test\benchmarks\cd38\external_tool_inputs\imported_outputs\cd38_external_tool_output_import_repair_plan.csv`.

## Outputs

- Import manifest: `D:\minimal_ML_test\benchmarks\cd38\external_tool_inputs\imported_outputs\cd38_external_tool_output_import_manifest.csv`
- Scan manifest: `D:\minimal_ML_test\benchmarks\cd38\external_tool_inputs\imported_outputs\cd38_external_tool_output_import_scan.csv`
- Coverage manifest: `D:\minimal_ML_test\benchmarks\cd38\external_tool_inputs\imported_outputs\cd38_external_tool_output_import_coverage.csv`
- Repair plan: `D:\minimal_ML_test\benchmarks\cd38\external_tool_inputs\imported_outputs\cd38_external_tool_output_import_repair_plan.csv`
- Summary JSON: `D:\minimal_ML_test\benchmarks\cd38\external_tool_inputs\imported_outputs\cd38_external_tool_output_import_summary.json`
