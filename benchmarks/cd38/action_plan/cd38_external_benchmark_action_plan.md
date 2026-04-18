# CD38 External Benchmark Action Plan

This file turns scattered CD38 external-tool blockers into an executable checklist.

## Summary

- Overall status: `blocked_external_outputs_missing`
- Total action rows: `6`
- Benchmark-completion rows: `4`
- Package reproducibility rows: `2`
- Missing benchmark-completion rows: `4`
- P2Rank on PATH: `False`
- fpocket on PATH: `False`
- Preflight missing P2Rank outputs: `3`
- Preflight missing fpocket outputs: `3`

## Recommended Order

1. Finish priority `1` fpocket rows for `3ROP` and `4OGW`; these close the highest-value ligand-bound benchmark gaps.
2. Finish priority `2` rows for `3F6Y`; this adds an apo-like CD38 stress test for P2Rank/fpocket.
3. If full external-package reproducibility is needed, also finish the package-only P2Rank rows for `3ROP` and `4OGW`.
4. Copy returned files under `benchmarks/cd38/external_tool_inputs/`, then run `python finalize_cd38_external_benchmark.py --run_discovered --run_sensitivity`.

## Action Rows

| Priority | PDB | Method | Tier | Status | Run On | Expected Return |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | 3ROP | fpocket | benchmark_completion | missing_output_dir | external_machine_or_wsl | `fpocket_runs/3ROP/3ROP_out/pockets/pocket*_atm.pdb` |
| 1 | 4OGW | fpocket | benchmark_completion | missing_output_dir | external_machine_or_wsl | `fpocket_runs/4OGW/4OGW_out/pockets/pocket*_atm.pdb` |
| 2 | 3F6Y | fpocket | benchmark_completion | missing_output_dir | external_machine_or_wsl | `fpocket_runs/3F6Y/3F6Y_out/pockets/pocket*_atm.pdb` |
| 2 | 3F6Y | p2rank | benchmark_completion | missing_output | external_machine_or_wsl | `p2rank_outputs/3F6Y/3F6Y.pdb_predictions.csv` |
| 3 | 3ROP | p2rank | package_reproducibility | missing_output | external_machine_or_wsl | `p2rank_outputs/3ROP/3ROP.pdb_predictions.csv` |
| 3 | 4OGW | p2rank | package_reproducibility | missing_output | external_machine_or_wsl | `p2rank_outputs/4OGW/4OGW.pdb_predictions.csv` |

## Commands

- Prepare or refresh input package: `python prepare_cd38_external_tool_inputs.py`
- Check local tools and missing outputs: `python check_cd38_external_tool_environment.py`
- Run P2Rank template where P2Rank is installed: `benchmarks\cd38\external_tool_inputs\run_p2rank_templates.ps1`
- Run fpocket template where fpocket is installed: `benchmarks\cd38\external_tool_inputs\run_fpocket_templates.ps1`
- Finalize after outputs return: `python finalize_cd38_external_benchmark.py --run_discovered --run_sensitivity`

## Detail Outputs

- CSV: `D:\minimal_ML_test\benchmarks\cd38\action_plan\cd38_external_benchmark_action_plan.csv`
- JSON: `D:\minimal_ML_test\benchmarks\cd38\action_plan\cd38_external_benchmark_action_plan.json`
- Markdown: `D:\minimal_ML_test\benchmarks\cd38\action_plan\cd38_external_benchmark_action_plan.md`
