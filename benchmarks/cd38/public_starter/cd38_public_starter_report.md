# CD38 Public Structure Starter

This report refreshes the locally runnable CD38 benchmark starter. It uses public PDB structures already available in the repository and does not run external pocket finders.

## What This Proves

- Public CD38 PDB inputs are present and reusable for benchmark checks.
- Existing ligand-contact and P2Rank baseline results can be summarized into one panel.
- The current pocket-shape proxy calibration can be refreshed without changing model scores.
- Missing external fpocket/P2Rank outputs are converted into an explicit action plan and next-run script.

## Current Status

- Panel rows: `4`
- Panel methods: `{'ligand_contact': 2, 'p2rank': 2}`
- Readiness missing/pending pairs: `4`
- Action plan status: `blocked_external_outputs_missing`
- Missing benchmark gap actions: `4`

## Next Actions

- Review the P2Rank readiness report; rerun with --run_discovered or run the generated P2Rank manifest when ready.
- Run external fpocket for structures listed as needs_fpocket_output, then rerun with --fpocket_root.

## Key Outputs

- panel_csv: `D:\minimal_ML_test\benchmarks\cd38\results\cd38_benchmark_panel.csv`
- panel_md: `D:\minimal_ML_test\benchmarks\cd38\results\cd38_benchmark_panel.md`
- ligand_candidate_report_md: `D:\minimal_ML_test\benchmarks\cd38\ligand_candidates\cd38_ligand_candidate_report.md`
- parameter_sensitivity_report_md: `D:\minimal_ML_test\benchmarks\cd38\parameter_sensitivity\cd38_pocket_parameter_sensitivity_report.md`
- proxy_calibration_report_md: `D:\minimal_ML_test\benchmarks\cd38\proxy_calibration\cd38_proxy_calibration_report.md`
- readiness_report_md: `D:\minimal_ML_test\benchmarks\cd38\readiness\cd38_benchmark_readiness_report.md`
- external_tool_preflight_report_md: `D:\minimal_ML_test\benchmarks\cd38\external_tool_inputs\preflight\cd38_external_tool_preflight_report.md`
- action_plan_md: `D:\minimal_ML_test\benchmarks\cd38\action_plan\cd38_external_benchmark_action_plan.md`
- external_tool_next_run_md: `D:\minimal_ML_test\benchmarks\cd38\external_tool_inputs\cd38_external_tool_next_run.md`
- external_tool_next_run_ps1: `D:\minimal_ML_test\benchmarks\cd38\external_tool_inputs\run_cd38_external_next_benchmark.ps1`

## Commands Run

| Step | Status | Return Code |
| --- | --- | ---: |
| summarize_cd38_benchmarks | success | 0 |
| inspect_cd38_ligand_candidates | success | 0 |
| analyze_cd38_pocket_parameter_sensitivity | success | 0 |
| build_cd38_proxy_calibration_report | success | 0 |
| prepare_cd38_external_tool_inputs | success | 0 |
| check_cd38_external_tool_environment | success | 0 |
| refresh_cd38_benchmark_readiness | success | 0 |
| build_cd38_external_benchmark_action_plan | success | 0 |
| build_cd38_external_tool_runbook | success | 0 |

## Boundary

This starter is a public-structure benchmark helper. It is not a nanobody screening run and does not replace real experimental validation.
