# CD38 P2Rank Benchmark Readiness Report

This report checks whether real `P2Rank` prediction CSV files are ready to be converted into the CD38 benchmark manifest.

## Summary

- Manifest CSV: `D:\minimal_ML_test\benchmarks\cd38\readiness\p2rank_discovered_manifest.csv`
- Summary JSON: `D:\minimal_ML_test\benchmarks\cd38\readiness\p2rank_discovered_manifest.csv.summary.json`
- Report MD: `D:\minimal_ML_test\benchmarks\cd38\readiness\p2rank_discovered_manifest.csv.report.md`
- Discovered prediction files: `2`
- Runnable manifest rows: `2`
- Skipped files: `0`
- Chain filter: `A`
- Near-hit threshold: `4.5`
- Fixed rank: `not fixed`
- Rank overrides: `3ROP=2,4OGW=1`
- Top-N when rank is not fixed: `1`
- Recommended next action: Manifest rows were created; run the generated manifest or rerun this script with --run.

## Scanned Roots

| Input | Resolved Path | Exists | Directory | prediction CSV files |
| --- | --- | --- | --- | ---: |
| benchmarks/cd38/results | D:\minimal_ML_test\benchmarks\cd38\results | True | True | 2 |

## Manifest Rows By PDB ID

| PDB ID | Manifest rows |
| --- | ---: |
| 3ROP | 1 |
| 4OGW | 1 |

## Skipped Files

No discovered prediction files were skipped.

## Commands

Regenerate manifest and report only:

```powershell
python prepare_cd38_p2rank_panel.py --p2rank_root benchmarks/cd38/results --manifest_out D:\minimal_ML_test\benchmarks\cd38\readiness\p2rank_discovered_manifest.csv --summary_json D:\minimal_ML_test\benchmarks\cd38\readiness\p2rank_discovered_manifest.csv.summary.json --report_md D:\minimal_ML_test\benchmarks\cd38\readiness\p2rank_discovered_manifest.csv.report.md --results_root benchmarks/cd38/results --truth_file benchmarks/cd38/cd38_active_site_truth.txt --chain_filter A --near_threshold 4.5 --rank_by_pdb 3ROP=2,4OGW=1 --top_n 1
```

Regenerate manifest, then run the P2Rank benchmark panel:

```powershell
python prepare_cd38_p2rank_panel.py --p2rank_root benchmarks/cd38/results --manifest_out D:\minimal_ML_test\benchmarks\cd38\readiness\p2rank_discovered_manifest.csv --summary_json D:\minimal_ML_test\benchmarks\cd38\readiness\p2rank_discovered_manifest.csv.summary.json --report_md D:\minimal_ML_test\benchmarks\cd38\readiness\p2rank_discovered_manifest.csv.report.md --results_root benchmarks/cd38/results --truth_file benchmarks/cd38/cd38_active_site_truth.txt --chain_filter A --near_threshold 4.5 --rank_by_pdb 3ROP=2,4OGW=1 --top_n 1 --run
```

After benchmark results exist, update the panel summary and parameter sensitivity tables:

```powershell
python summarize_cd38_benchmarks.py --results_root benchmarks\cd38\results
python analyze_cd38_pocket_parameter_sensitivity.py --manifest_csv D:\minimal_ML_test\benchmarks\cd38\readiness\p2rank_discovered_manifest.csv --results_root benchmarks\cd38\results --truth_file benchmarks\cd38\cd38_active_site_truth.txt --out_dir benchmarks\cd38\parameter_sensitivity
```

## Interpretation

- `Runnable manifest rows > 0` means the directory structure and CSV schema are usable for the current benchmark pipeline.
- `Skipped files > 0` usually means the PDB ID was not present in the filename/path or the CSV lacks P2Rank columns.
- This script does not run external `P2Rank`; it only imports existing `*_predictions.csv` files into the local benchmark workflow.
- Use `--rank_by_pdb 3ROP=2,4OGW=1` when chain-specific active pockets are not P2Rank global rank 1.
