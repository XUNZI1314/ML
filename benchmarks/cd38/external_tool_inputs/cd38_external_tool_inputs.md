# CD38 External Tool Input Package

This package contains PDB inputs and command templates for running external P2Rank/fpocket tools before importing their outputs into the CD38 benchmark.

## Summary

- Prepared structures: `3`
- P2Rank targets: `3`
- fpocket targets: `3`
- Package directory: `external_tool_inputs/`

## Files

- Input manifest: `cd38_external_tool_input_manifest.csv`
- Expected return manifest: `cd38_external_tool_expected_returns.csv`
- Return checklist: `cd38_external_tool_return_checklist.md`
- P2Rank PowerShell template: `run_p2rank_templates.ps1`
- fpocket PowerShell template: `run_fpocket_templates.ps1`
- P2Rank Bash template: `run_p2rank_templates.sh`
- fpocket Bash template: `run_fpocket_templates.sh`
- Environment/output preflight template: `check_external_tool_environment.ps1`
- Readiness refresh template: `refresh_readiness_after_external_tools.ps1`
- Finalize workflow template: `finalize_external_benchmark.ps1`

## Recommended Flow

1. Run `check_external_tool_environment.ps1` to confirm whether P2Rank/fpocket commands and expected outputs are available.
   The preflight step resolves current package-local paths first, so the package can be moved to WSL/Linux or another machine without being trapped by old absolute paths in the manifest.
2. Run or adapt `run_p2rank_templates.ps1` on a machine where P2Rank is installed.
3. Run or adapt `run_fpocket_templates.ps1` on a machine where fpocket is installed.
4. On Linux/WSL, use `run_p2rank_templates.sh` and `run_fpocket_templates.sh` instead.
5. Before zipping/copying back, open `cd38_external_tool_return_checklist.md` and confirm the expected returned output paths exist.
6. Copy the generated output folders back into this package if they were run elsewhere; if you received a whole returned directory or zip, use `finalize_cd38_external_benchmark.py --import_source <returned_zip_or_dir>`.
7. Run `check_external_tool_environment.ps1` again to confirm expected outputs are now present.
8. Run `finalize_external_benchmark.ps1` for a one-command preflight + readiness summary.
9. If readiness reports look correct, rerun `finalize_cd38_external_benchmark.py --run_discovered` to import discovered rows; if the returned package is already confirmed, use `finalize_cd38_external_benchmark.py --import_source <returned_zip_or_dir> --run_discovered`.

## Targets

| PDB ID | Methods | PDB Input | Expected P2Rank CSV | Expected fpocket pockets dir |
| --- | --- | --- | --- | --- |
| 3ROP | ligand_contact,p2rank,fpocket | pdbs/3ROP.pdb | p2rank_outputs/3ROP/3ROP.pdb_predictions.csv | fpocket_runs/3ROP/3ROP_out/pockets |
| 4OGW | ligand_contact,p2rank,fpocket | pdbs/4OGW.pdb | p2rank_outputs/4OGW/4OGW.pdb_predictions.csv | fpocket_runs/4OGW/4OGW_out/pockets |
| 3F6Y | p2rank,fpocket | pdbs/3F6Y.pdb | p2rank_outputs/3F6Y/3F6Y.pdb_predictions.csv | fpocket_runs/3F6Y/3F6Y_out/pockets |
