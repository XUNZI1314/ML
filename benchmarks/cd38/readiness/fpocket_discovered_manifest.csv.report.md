# CD38 fpocket Benchmark Readiness Report

This report checks whether real `fpocket` output folders are ready to be converted into the CD38 benchmark manifest.

## Summary

- Manifest CSV: `D:\minimal_ML_test\benchmarks\cd38\readiness\fpocket_discovered_manifest.csv`
- Summary JSON: `D:\minimal_ML_test\benchmarks\cd38\readiness\fpocket_discovered_manifest.csv.summary.json`
- Report MD: `D:\minimal_ML_test\benchmarks\cd38\readiness\fpocket_discovered_manifest.csv.report.md`
- Discovered pocket files: `0`
- Runnable manifest rows: `0`
- Skipped files: `0`
- Chain filter: `A`
- Near-hit threshold: `4.5`
- Include HETATM: `False`
- Max pockets per structure: `not capped`
- Recommended next action: No fpocket pocket*_atm.pdb files were found; place real fpocket outputs under the scanned root and rerun discovery.

## Scanned Roots

| Input | Resolved Path | Exists | Directory | pocket*_atm.pdb files |
| --- | --- | --- | --- | ---: |
| D:\minimal_ML_test\benchmarks\cd38\external_tool_inputs\fpocket_runs | D:\minimal_ML_test\benchmarks\cd38\external_tool_inputs\fpocket_runs | True | True | 0 |

## Manifest Rows By PDB ID

No runnable fpocket manifest rows were generated.

## Skipped Files

No discovered pocket files were skipped.

## Commands

Regenerate manifest and report only:

```powershell
python prepare_cd38_fpocket_panel.py --fpocket_root D:\minimal_ML_test\benchmarks\cd38\external_tool_inputs\fpocket_runs --manifest_out D:\minimal_ML_test\benchmarks\cd38\readiness\fpocket_discovered_manifest.csv --summary_json D:\minimal_ML_test\benchmarks\cd38\readiness\fpocket_discovered_manifest.csv.summary.json --report_md D:\minimal_ML_test\benchmarks\cd38\readiness\fpocket_discovered_manifest.csv.report.md --results_root benchmarks/cd38/results --truth_file benchmarks/cd38/cd38_active_site_truth.txt --chain_filter A --near_threshold 4.5
```

Regenerate manifest, then run the fpocket benchmark panel:

```powershell
python prepare_cd38_fpocket_panel.py --fpocket_root D:\minimal_ML_test\benchmarks\cd38\external_tool_inputs\fpocket_runs --manifest_out D:\minimal_ML_test\benchmarks\cd38\readiness\fpocket_discovered_manifest.csv --summary_json D:\minimal_ML_test\benchmarks\cd38\readiness\fpocket_discovered_manifest.csv.summary.json --report_md D:\minimal_ML_test\benchmarks\cd38\readiness\fpocket_discovered_manifest.csv.report.md --results_root benchmarks/cd38/results --truth_file benchmarks/cd38/cd38_active_site_truth.txt --chain_filter A --near_threshold 4.5 --run
```

After benchmark results exist, update the panel summary and parameter sensitivity tables:

```powershell
python summarize_cd38_benchmarks.py --results_root benchmarks\cd38\results
python analyze_cd38_pocket_parameter_sensitivity.py --manifest_csv D:\minimal_ML_test\benchmarks\cd38\readiness\fpocket_discovered_manifest.csv --results_root benchmarks\cd38\results --truth_file benchmarks\cd38\cd38_active_site_truth.txt --out_dir benchmarks\cd38\parameter_sensitivity
```

## Interpretation

- `Runnable manifest rows > 0` means the directory structure is usable for the current benchmark pipeline.
- `Skipped files > 0` usually means the PDB ID was not present in the fpocket output path; use `--rcsb_pdb_id` for single-structure batches.
- This script does not run external `fpocket`; it only imports existing `pocket*_atm.pdb` files into the local benchmark workflow.
