# CD38 Pocket Accuracy Benchmark

- pdb_path: `D:\minimal_ML_test\benchmarks\cd38\results\3ROP_ligand_contact_chainA_50A_NCA\inputs\3ROP.pdb`
- truth_file: `D:\minimal_ML_test\benchmarks\cd38\cd38_active_site_truth.txt`
- predicted_pocket_file: `D:\minimal_ML_test\benchmarks\cd38\results\3ROP_ligand_contact_chainA_50A_NCA\predicted_pocket.txt`
- structure_chain_ids: `A, B`

## Exact Match Metrics

- exact_truth_coverage: `0.8571`
- exact_predicted_precision: `0.4615`
- exact_jaccard: `0.4286`
- exact_f1: `0.6000`

## Near-Hit Metrics

- truth_near_coverage @ 4.50A: `1.0000`
- predicted_near_precision @ 4.50A: `1.0000`
- mean_truth_min_distance: `0.5045`

## Boundary Tightness Metrics

- predicted_to_truth_residue_ratio: `1.8571`
- extra_predicted_fraction: `0.5385`
- far_predicted_fraction: `0.0000`
- overwide_pocket_score: `0.2995`

## Outputs

- `cd38_pocket_accuracy_summary.json`
- `truth_residue_table.csv`
- `predicted_residue_table.csv`
- `exact_overlap_table.csv`
- `missed_truth_table.csv`
- `extra_predicted_table.csv`
- `near_hit_table.csv`
- `predicted_to_truth_distance_table.csv`
