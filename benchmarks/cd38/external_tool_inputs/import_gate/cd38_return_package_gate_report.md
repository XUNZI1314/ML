# CD38 Return Package Gate

This gate evaluates whether the latest returned P2Rank/fpocket package is safe to import into the CD38 benchmark workflow.

## Gate Decision

- Gate status: `FAIL_INPUT_PACKAGE`
- Decision: The source looks like the original transfer/input package, not a completed return package.
- Source diagnosis: `input_package_without_external_outputs`
- Candidate files: `0`
- Imported files: `0`
- Dry-run files: `0`
- Expected coverage: `0/6`
- Missing expected outputs: `6`
- Unexpected outputs: `0`
- Synthetic fixture detected: `False`

## Recommended Actions

- Run P2Rank/fpocket on the transfer package first.
- Return `p2rank_outputs/` and `fpocket_runs/*/*_out/`, then dry-run again.

## Action Plan Context

- Action plan rows: `6`
- Benchmark gap rows: `4`
- Benchmark gap rows still missing: `4`

## Files

- Import summary: `D:\minimal_ML_test\benchmarks\cd38\external_tool_inputs\import_gate\preimport_dry_run\cd38_external_tool_output_import_summary.json`
- Import scan CSV: `D:\minimal_ML_test\benchmarks\cd38\external_tool_inputs\import_gate\preimport_dry_run\cd38_external_tool_output_import_scan.csv`
- Gate summary JSON: `D:\minimal_ML_test\benchmarks\cd38\external_tool_inputs\import_gate\cd38_return_package_gate_summary.json`
- Gate report: `D:\minimal_ML_test\benchmarks\cd38\external_tool_inputs\import_gate\cd38_return_package_gate_report.md`
- Gate decision CSV: `D:\minimal_ML_test\benchmarks\cd38\external_tool_inputs\import_gate\cd38_return_package_gate_decision.csv`

## Boundary

This gate checks returned package shape, coverage, and obvious misuse. It does not validate scientific pocket accuracy. Real P2Rank/fpocket outputs still need benchmark scoring and manual review.
