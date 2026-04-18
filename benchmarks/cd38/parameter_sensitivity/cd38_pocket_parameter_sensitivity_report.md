# CD38 Pocket Parameter Sensitivity

- Ligand-contact scenarios: `10`
- P2Rank rank scenarios: `16`
- fpocket pocket scenarios: `0`
- Method-consensus scenarios: `4`
- Overwide-penalty scenarios: `90`

## Interpretation

- `exact_truth_coverage` measures recovery of known CD38 key residues.
- `exact_predicted_precision` measures how narrow the predicted pocket is against the conservative truth set.
- `utility_score = 0.60 * coverage + 0.40 * precision`; adjusted utility subtracts an overwide penalty.
- This is a robustness check using existing local benchmark files; it does not run external pocket finders.

## Ligand-Contact Cutoff Scenarios

| result_name | parameter_value | predicted_residue_count | exact_truth_coverage | exact_predicted_precision | exact_f1 | utility_score |
| --- | --- | --- | --- | --- | --- | --- |
| 3ROP_ligand_contact_chainA_50A_NCA | 3.5000 | 12 | 0.8571 | 0.5000 | 0.6316 | 0.7143 |
| 3ROP_ligand_contact_chainA_50A_NCA | 4.0000 | 13 | 0.8571 | 0.4615 | 0.6000 | 0.6989 |
| 3ROP_ligand_contact_chainA_50A_NCA | 4.5000 | 13 | 0.8571 | 0.4615 | 0.6000 | 0.6989 |
| 3ROP_ligand_contact_chainA_50A_NCA | 5.0000 | 15 | 0.8571 | 0.4000 | 0.5455 | 0.6743 |
| 3ROP_ligand_contact_chainA_50A_NCA | 5.5000 | 18 | 0.8571 | 0.3333 | 0.4800 | 0.6476 |
| 4OGW_ligand_contact_chainA_NMN | 3.5000 | 12 | 1.0000 | 0.5833 | 0.7368 | 0.8333 |
| 4OGW_ligand_contact_chainA_NMN | 4.0000 | 13 | 1.0000 | 0.5385 | 0.7000 | 0.8154 |
| 4OGW_ligand_contact_chainA_NMN | 4.5000 | 14 | 1.0000 | 0.5000 | 0.6667 | 0.8000 |
| 4OGW_ligand_contact_chainA_NMN | 5.0000 | 15 | 1.0000 | 0.4667 | 0.6364 | 0.7867 |
| 4OGW_ligand_contact_chainA_NMN | 5.5000 | 20 | 1.0000 | 0.3500 | 0.5185 | 0.7400 |

## P2Rank Rank Choice Scenarios

| result_name | parameter_value | pocket_name | predicted_residue_count | exact_truth_coverage | exact_predicted_precision | utility_score |
| --- | --- | --- | --- | --- | --- | --- |
| 3ROP_p2rank_rank2_chainA | 1 | pocket1 | 0 | 0.0000 |  | 0.0000 |
| 3ROP_p2rank_rank2_chainA | 2 | pocket2 | 14 | 1.0000 | 0.5000 | 0.8000 |
| 3ROP_p2rank_rank2_chainA | 3 | pocket3 | 7 | 0.0000 | 0.0000 | 0.0000 |
| 3ROP_p2rank_rank2_chainA | 4 | pocket4 | 9 | 0.0000 | 0.0000 | 0.0000 |
| 3ROP_p2rank_rank2_chainA | 5 | pocket5 | 0 | 0.0000 |  | 0.0000 |
| 3ROP_p2rank_rank2_chainA | 6 | pocket6 | 11 | 0.0000 | 0.0000 | 0.0000 |
| 3ROP_p2rank_rank2_chainA | 7 | pocket7 | 13 | 0.1429 | 0.0769 | 0.1165 |
| 3ROP_p2rank_rank2_chainA | 8 | pocket8 | 6 | 0.0000 | 0.0000 | 0.0000 |
| 3ROP_p2rank_rank2_chainA | 9 | pocket9 | 0 | 0.0000 |  | 0.0000 |
| 3ROP_p2rank_rank2_chainA | 10 | pocket10 | 9 | 0.0000 | 0.0000 | 0.0000 |
| 3ROP_p2rank_rank2_chainA | 11 | pocket11 | 0 | 0.0000 |  | 0.0000 |
| 3ROP_p2rank_rank2_chainA | 12 | pocket12 | 7 | 0.0000 | 0.0000 | 0.0000 |
| 4OGW_p2rank_rank1_chainA | 1 | pocket1 | 26 | 1.0000 | 0.2692 | 0.7077 |
| 4OGW_p2rank_rank1_chainA | 2 | pocket2 | 12 | 0.1429 | 0.0833 | 0.1190 |
| 4OGW_p2rank_rank1_chainA | 3 | pocket3 | 15 | 0.0000 | 0.0000 | 0.0000 |
| 4OGW_p2rank_rank1_chainA | 4 | pocket4 | 8 | 0.0000 | 0.0000 | 0.0000 |

## fpocket Pocket Choice Scenarios

_No rows._

## Method Consensus Threshold Scenarios

| rcsb_pdb_id | parameter_value | predicted_residue_count | exact_truth_coverage | exact_predicted_precision | utility_score |
| --- | --- | --- | --- | --- | --- |
| 3ROP | 1 | 16 | 1.0000 | 0.4375 | 0.7750 |
| 3ROP | 2 | 11 | 0.8571 | 0.5455 | 0.7325 |
| 4OGW | 1 | 27 | 1.0000 | 0.2593 | 0.7037 |
| 4OGW | 2 | 13 | 1.0000 | 0.5385 | 0.8154 |

## Best Adjusted Utility Examples

| scenario_type | result_name | parameter_value | overwide_penalty_weight | exact_truth_coverage | exact_predicted_precision | adjusted_utility_score |
| --- | --- | --- | --- | --- | --- | --- |
| ligand_contact_cutoff | 4OGW_ligand_contact_chainA_NMN | 3.5000 | 0.0000 | 1.0000 | 0.5833 | 0.8333 |
| ligand_contact_cutoff | 4OGW_ligand_contact_chainA_NMN | 4.0000 | 0.0000 | 1.0000 | 0.5385 | 0.8154 |
| method_consensus_threshold | 4OGW_method_consensus_min2 | 2.0000 | 0.0000 | 1.0000 | 0.5385 | 0.8154 |
| ligand_contact_cutoff | 4OGW_ligand_contact_chainA_NMN | 4.5000 | 0.0000 | 1.0000 | 0.5000 | 0.8000 |
| p2rank_rank_choice | 3ROP_p2rank_rank2_chainA | 2.0000 | 0.0000 | 1.0000 | 0.5000 | 0.8000 |
| ligand_contact_cutoff | 4OGW_ligand_contact_chainA_NMN | 5.0000 | 0.0000 | 1.0000 | 0.4667 | 0.7867 |
| method_consensus_threshold | 3ROP_method_consensus_min1 | 1.0000 | 0.0000 | 1.0000 | 0.4375 | 0.7750 |
| ligand_contact_cutoff | 4OGW_ligand_contact_chainA_NMN | 3.5000 | 0.1500 | 1.0000 | 0.5833 | 0.7708 |
| method_consensus_threshold | 4OGW_method_consensus_min2 | 2.0000 | 0.1500 | 1.0000 | 0.5385 | 0.7462 |
| ligand_contact_cutoff | 4OGW_ligand_contact_chainA_NMN | 4.0000 | 0.1500 | 1.0000 | 0.5385 | 0.7462 |
| ligand_contact_cutoff | 4OGW_ligand_contact_chainA_NMN | 5.5000 | 0.0000 | 1.0000 | 0.3500 | 0.7400 |
| method_consensus_threshold | 3ROP_method_consensus_min2 | 2.0000 | 0.0000 | 0.8571 | 0.5455 | 0.7325 |

## Outputs

- `contact_cutoff_sensitivity.csv`
- `p2rank_rank_sensitivity.csv`
- `fpocket_pocket_sensitivity.csv`
- `method_consensus_threshold_sensitivity.csv`
- `overwide_penalty_sensitivity.csv`
- `cd38_pocket_parameter_sensitivity_summary.json`