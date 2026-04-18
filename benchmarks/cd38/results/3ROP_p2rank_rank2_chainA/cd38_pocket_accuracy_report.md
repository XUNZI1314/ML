# CD38 Pocket Accuracy Benchmark

- pdb_path: `D:\minimal_ML_test\benchmarks\cd38\results\3ROP_p2rank_rank2_chainA\inputs\3ROP.pdb`
- truth_file: `D:\minimal_ML_test\benchmarks\cd38\cd38_active_site_truth.txt`
- predicted_pocket_file: `D:\minimal_ML_test\benchmarks\cd38\results\3ROP_p2rank_rank2_chainA\predicted_pocket.txt`
- structure_chain_ids: `A, B`

## Exact Match Metrics

- exact_truth_coverage: `1.0000`
- exact_predicted_precision: `0.5000`
- exact_jaccard: `0.5000`
- exact_f1: `0.6667`

## Near-Hit Metrics

- truth_near_coverage @ 4.50A: `1.0000`
- predicted_near_precision @ 4.50A: `1.0000`
- mean_truth_min_distance: `0.0000`

## Boundary Tightness Metrics

- predicted_to_truth_residue_ratio: `2.0000`
- extra_predicted_fraction: `0.5000`
- far_predicted_fraction: `0.0000`
- overwide_pocket_score: `0.2917`

## Outputs

- `cd38_pocket_accuracy_summary.json`
- `truth_residue_table.csv`
- `predicted_residue_table.csv`
- `exact_overlap_table.csv`
- `missed_truth_table.csv`
- `extra_predicted_table.csv`
- `near_hit_table.csv`
- `predicted_to_truth_distance_table.csv`
