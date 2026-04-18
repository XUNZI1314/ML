# CD38 External Tool Next Runbook

This runbook is generated from the current CD38 external benchmark action plan.
It is meant for the machine or WSL/Linux environment that has P2Rank and/or fpocket installed.

## Summary

- Selected actions: `4`
- Include package reproducibility rows: `False`
- Include already-ready rows: `False`
- Need P2Rank: `True`
- Need fpocket: `True`

## Run

Windows PowerShell:

```powershell
.\run_cd38_external_next_benchmark.ps1
```

Linux / WSL:

```bash
chmod +x run_cd38_external_next_benchmark.sh
./run_cd38_external_next_benchmark.sh
```

## Return And Import

After the external tools finish, return the whole completed package folder or zip it.
In the original ML project, first dry-run/gate the returned folder, then finalize:

```bash
python finalize_cd38_external_benchmark.py --import_source returned_external_tool_inputs.zip --run_discovered --run_sensitivity
```

The finalize command will gate the returned package before importing. Non-PASS gate statuses are blocked by default.

## Included Actions

| Order | Priority | PDB | Method | Tier | Current Status | Expected Return |
| ---: | ---: | --- | --- | --- | --- | --- |
| 1 | 1 | 3ROP | fpocket | benchmark_completion | missing_output_dir | `fpocket_runs/3ROP/3ROP_out/pockets/pocket*_atm.pdb` |
| 2 | 1 | 4OGW | fpocket | benchmark_completion | missing_output_dir | `fpocket_runs/4OGW/4OGW_out/pockets/pocket*_atm.pdb` |
| 3 | 2 | 3F6Y | fpocket | benchmark_completion | missing_output_dir | `fpocket_runs/3F6Y/3F6Y_out/pockets/pocket*_atm.pdb` |
| 4 | 2 | 3F6Y | p2rank | benchmark_completion | missing_output | `p2rank_outputs/3F6Y/3F6Y.pdb_predictions.csv` |

## Outputs

- CSV: `cd38_external_tool_next_run_plan.csv`
- JSON: `cd38_external_tool_next_run_summary.json`
- Markdown: `cd38_external_tool_next_run.md`
- PowerShell runner: `run_cd38_external_next_benchmark.ps1`
- Bash runner: `run_cd38_external_next_benchmark.sh`

## Boundary

This file only describes and runs external pocket-finder tools. It does not create benchmark evidence until the returned outputs pass import gate and are finalized in the original project.
