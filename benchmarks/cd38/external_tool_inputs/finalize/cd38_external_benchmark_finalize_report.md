# CD38 External Benchmark Finalize

This report combines external-tool preflight checks, readiness discovery, and optional benchmark import for CD38 P2Rank/fpocket outputs.

## Summary

- Package directory: `D:\minimal_ML_test\benchmarks\cd38\external_tool_inputs`
- Import source: `D:\minimal_ML_test\benchmarks\cd38\external_tool_transfer\cd38_external_tool_inputs_transfer.zip`
- Import source diagnosis: `input_package_without_external_outputs`
- Import diagnostic message: This looks like the original external-tool input package, not a completed return package. Run P2Rank/fpocket first, then return p2rank_outputs/ and fpocket_runs/*/*_out/.
- Imported files: `0`
- Import expected outputs ready: `0/6`
- Import missing expected outputs: `6`
- Import repair actions: `6`
- Import gate enabled: `True`
- Import gate strict: `True`
- Import gate status: `FAIL_INPUT_PACKAGE`
- Import blocked by gate: `True`
- Run discovered benchmark rows: `False`
- Runnable external benchmark rows: `0`
- Benchmark follow-up status: `not_requested`
- Run sensitivity analysis: `False`
- External action plan status: `blocked_external_outputs_missing`
- External action plan rows: `6`
- External benchmark-gap rows: `4`
- P2Rank missing outputs: `3`
- fpocket missing outputs: `3`
- P2Rank discovery status: `needs_external_output`
- fpocket discovery status: `needs_external_output`

## Next Actions

- Import was blocked by return-package gate: FAIL_INPUT_PACKAGE. The source looks like the original transfer/input package, not a completed return package.
- Run a dry-run import and review the gate report before attempting finalize again.
- Import source diagnosis: input_package_without_external_outputs. This looks like the original external-tool input package, not a completed return package. Run P2Rank/fpocket first, then return p2rank_outputs/ and fpocket_runs/*/*_out/.
- Missing expected external outputs in returned source: 3ROP:p2rank, 3ROP:fpocket, 4OGW:p2rank, 4OGW:fpocket, 3F6Y:p2rank, 3F6Y:fpocket.
- Open the repair plan CSV for exact missing paths and run templates: D:\minimal_ML_test\benchmarks\cd38\external_tool_inputs\finalize\import_gate\preimport_dry_run\cd38_external_tool_output_import_repair_plan.csv.
- Open the import scan CSV to inspect ignored files and confirm whether p2rank_outputs/ or fpocket_runs/*/*_out/ are present.
- External outputs are still missing; run or adapt run_p2rank_templates.ps1 / run_fpocket_templates.ps1 on a machine with the tools installed.
- Open the consolidated external benchmark action plan for exact PDB-method priorities: D:\minimal_ML_test\benchmarks\cd38\action_plan\cd38_external_benchmark_action_plan.md.
- After outputs are copied back, rerun this finalize script.
- Action plan still has 4 missing benchmark-completion rows.
- Check P2Rank readiness report; no runnable P2Rank manifest rows were created.
- Check fpocket readiness report; no runnable fpocket manifest rows were created.

## Child Steps

| Step | Status | Return Code |
| --- | --- | ---: |
| preimport_gate_dry_run | success | 0 |
| preimport_return_package_gate | success | 0 |
| import_outputs | skipped |  |
| preflight | success | 0 |
| readiness | success | 0 |
| external_benchmark_action_plan | skipped |  |

## Output Files

- Finalize summary JSON: `D:\minimal_ML_test\benchmarks\cd38\external_tool_inputs\finalize\cd38_external_benchmark_finalize_summary.json`
- Finalize report: `D:\minimal_ML_test\benchmarks\cd38\external_tool_inputs\finalize\cd38_external_benchmark_finalize_report.md`
- Import report: `D:\minimal_ML_test\benchmarks\cd38\external_tool_inputs\imported_outputs\cd38_external_tool_output_import_report.md`
- Import scan CSV: `D:\minimal_ML_test\benchmarks\cd38\external_tool_inputs\imported_outputs\cd38_external_tool_output_import_scan.csv`
- Import coverage CSV: `D:\minimal_ML_test\benchmarks\cd38\external_tool_inputs\imported_outputs\cd38_external_tool_output_import_coverage.csv`
- Import repair plan CSV: `D:\minimal_ML_test\benchmarks\cd38\external_tool_inputs\imported_outputs\cd38_external_tool_output_import_repair_plan.csv`
- Import gate report: `D:\minimal_ML_test\benchmarks\cd38\external_tool_inputs\finalize\import_gate\cd38_return_package_gate_report.md`
- Preflight report: `D:\minimal_ML_test\benchmarks\cd38\external_tool_inputs\preflight\cd38_external_tool_preflight_report.md`
- Readiness report: `D:\minimal_ML_test\benchmarks\cd38\readiness\cd38_benchmark_readiness_report.md`
- External benchmark action plan: `D:\minimal_ML_test\benchmarks\cd38\action_plan\cd38_external_benchmark_action_plan.md`
- External benchmark action plan CSV: `D:\minimal_ML_test\benchmarks\cd38\action_plan\cd38_external_benchmark_action_plan.csv`
- Benchmark panel: `benchmarks/cd38/results/cd38_benchmark_panel.md`
- Parameter sensitivity report: `benchmarks/cd38/parameter_sensitivity/cd38_pocket_parameter_sensitivity_report.md`
- Proxy calibration report: `benchmarks/cd38/proxy_calibration/cd38_proxy_calibration_report.md`
