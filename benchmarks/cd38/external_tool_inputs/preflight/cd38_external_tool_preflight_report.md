# CD38 External Pocket Tool Preflight

This report checks whether the CD38 external-tool input package has local tool commands and expected P2Rank/fpocket outputs.

## Summary

- Package directory: `D:\minimal_ML_test\benchmarks\cd38\external_tool_inputs`
- Manifest rows: `3`
- Path resolution: `package_portable_first`
- P2Rank command available: `False`
- P2Rank command path: ``
- fpocket command available: `False`
- fpocket command path: ``
- Missing P2Rank outputs: `3`
- Missing fpocket outputs: `3`

## Next Actions

- P2Rank command is not available on PATH; install P2Rank or run run_p2rank_templates.ps1 on another machine, then copy outputs back.
- fpocket command is not available on PATH; install fpocket or run run_fpocket_templates.ps1 on another machine, then copy outputs back.
- After external tools finish, run benchmarks/cd38/external_tool_inputs/refresh_readiness_after_external_tools.ps1.

## Per-Structure Status

| PDB ID | Methods | PDB Input | PDB Source | P2Rank | P2Rank Source | fpocket | fpocket Source | fpocket pockets |
| --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| 3ROP | ligand_contact,p2rank,fpocket | ready | package_portable | missing_output | package_portable | missing_output_dir | package_portable | 0 |
| 4OGW | ligand_contact,p2rank,fpocket | ready | package_portable | missing_output | package_portable | missing_output_dir | package_portable | 0 |
| 3F6Y | p2rank,fpocket | ready | package_portable | missing_output | package_portable | missing_output_dir | package_portable | 0 |
