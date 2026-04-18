# CD38 Benchmark Readiness Refresh

This report consolidates ligand suitability, benchmark expansion gaps, and optional external P2Rank/fpocket discovery checks.

## Summary

- Ligand scan status: `ready`
- Expansion plan status: `ready`
- P2Rank discovery status: `ready`
- fpocket discovery status: `not_run`
- Missing or pending structure-method pairs: `4`

## Next Actions

- Review the P2Rank readiness report; rerun with --run_discovered or run the generated P2Rank manifest when ready.
- Run external fpocket for structures listed as needs_fpocket_output, then rerun with --fpocket_root.

## Detail Files

- Readiness summary JSON: `D:\minimal_ML_test\benchmarks\cd38\readiness\cd38_benchmark_readiness_summary.json`
- Expansion missing actions CSV: `benchmarks\cd38\expansion_plan\cd38_benchmark_missing_actions.csv`
- Ligand candidate report: `benchmarks\cd38\ligand_candidates\cd38_ligand_candidate_report.md`
- P2Rank readiness report: `D:\minimal_ML_test\benchmarks\cd38\readiness\p2rank_discovered_manifest.csv.report.md`
- fpocket readiness report: `D:\minimal_ML_test\benchmarks\cd38\readiness\fpocket_discovered_manifest.csv.report.md`

## Tool Status

| Tool | Status | Key Count | Notes |
| --- | --- | ---: | --- |
| ligand_candidates | ready | 3 | recommended ligand candidates |
| expansion_plan | ready | 4 | missing or pending pairs |
| p2rank | ready | 2 | runnable manifest rows |
| fpocket | not_run |  | runnable manifest rows |
