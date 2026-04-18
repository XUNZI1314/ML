# CD38 Proxy Calibration Report

This report compares the runtime `pocket_shape_overwide_proxy` against CD38 truth-based benchmark metrics.
It is a calibration aid only; it does not change ranking defaults.

## Recommendation

- Evidence level: `low`
- Policy: `keep_default_penalty_off`
- Recommended runtime proxy threshold: `0.5500`
- Recommended default penalty weight: `0.0000`
- Optional experimental penalty weight: `0.1500`
- Reason: Current benchmark is useful for direction, but not large/diverse enough to change defaults.

## Current Evidence Coverage

- Benchmark rows: `4`
- PDB structures: `2`
- Methods: `2`
- fpocket rows: `0`
- Truth-risk rows: `1`
- Runtime proxy computed rows: `4`

## Blockers Before Changing Defaults

- benchmark structures are too few (2 < 5)
- truth risk/non-risk split is too small (1/3)
- no real fpocket rows and fewer than three method families are present

## Next Actions

- Add real fpocket outputs for 3ROP/4OGW/3F6Y and rerun finalize_cd38_external_benchmark.py --run_discovered --run_sensitivity.
- Add at least one non-CD38 benchmark protein before changing global ranking defaults.
- Keep --pocket_overwide_penalty_weight at 0.0 by default; use 0.15 only for sensitivity review runs.

## Calibration Rows

| result_name | method | pdb_id | exact_truth_coverage | exact_predicted_precision | overwide_pocket_score | pocket_shape_overwide_proxy | truth_risk_label | runtime_proxy_flag | proxy_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3ROP_ligand_contact_chainA_50A_NCA | ligand_contact | 3ROP | 0.8571 | 0.4615 | 0.2995 | 0.1868 | 0.0000 | 0.0000 | ok |
| 3ROP_p2rank_rank2_chainA | p2rank | 3ROP | 1.0000 | 0.5000 | 0.2917 | 0.2190 | 0.0000 | 0.0000 | ok |
| 4OGW_ligand_contact_chainA_NMN | ligand_contact | 4OGW | 1.0000 | 0.5000 | 0.3167 | 0.2168 | 0.0000 | 0.0000 | ok |
| 4OGW_p2rank_rank1_chainA | p2rank | 4OGW | 1.0000 | 0.2692 | 0.6175 | 0.5937 | 1.0000 | 1.0000 | ok |

## Method Summary

| method | row_count | pdb_count | mean_coverage | mean_precision | mean_truth_overwide | mean_runtime_proxy | truth_risk_count | runtime_proxy_flag_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ligand_contact | 2.0000 | 2.0000 | 0.9286 | 0.4808 | 0.3081 | 0.2018 | 0.0000 | 0.0000 |
| p2rank | 2.0000 | 2.0000 | 1.0000 | 0.3846 | 0.4546 | 0.4064 | 1.0000 | 1.0000 |

## Threshold Candidates

| threshold | tp | tn | fp | fn | sensitivity | specificity | precision | balanced_accuracy | f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.2500 | 1.0000 | 3.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 0.3000 | 1.0000 | 3.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 0.3500 | 1.0000 | 3.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 0.4000 | 1.0000 | 3.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 0.4500 | 1.0000 | 3.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 0.5000 | 1.0000 | 3.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 0.5500 | 1.0000 | 3.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 0.2000 | 1.0000 | 1.0000 | 2.0000 | 0.0000 | 1.0000 | 0.3333 | 0.3333 | 0.6667 | 0.5000 |
| 0.6000 | 0.0000 | 3.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.5000 | 0.0000 |
| 0.6500 | 0.0000 | 3.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.5000 | 0.0000 |
| 0.7000 | 0.0000 | 3.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.5000 | 0.0000 |

## Penalty Simulation Summary

| overwide_penalty_weight | scenario_count | paired_scenario_count | mean_utility_score | mean_overwide_risk | mean_adjusted_utility_score | min_adjusted_utility_score |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0000 | 30.0000 | 26.0000 | 0.4684 | 0.7190 | 0.4684 | 0.0000 |
| 0.1500 | 30.0000 | 26.0000 | 0.4684 | 0.7190 | 0.4083 | 0.0000 |
| 0.3000 | 30.0000 | 26.0000 | 0.4684 | 0.7190 | 0.3572 | 0.0000 |

## Output Files

- Summary JSON: `D:\minimal_ML_test\benchmarks\cd38\proxy_calibration\cd38_proxy_calibration_summary.json`
- Report Markdown: `D:\minimal_ML_test\benchmarks\cd38\proxy_calibration\cd38_proxy_calibration_report.md`
- Calibration rows CSV: `D:\minimal_ML_test\benchmarks\cd38\proxy_calibration\cd38_proxy_calibration_rows.csv`
- Threshold candidates CSV: `D:\minimal_ML_test\benchmarks\cd38\proxy_calibration\cd38_proxy_threshold_candidates.csv`
- Method summary CSV: `D:\minimal_ML_test\benchmarks\cd38\proxy_calibration\cd38_proxy_method_summary.csv`
- Penalty summary CSV: `D:\minimal_ML_test\benchmarks\cd38\proxy_calibration\cd38_proxy_penalty_summary.csv`
