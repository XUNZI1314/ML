# Pocket Method Consensus Analysis

- Methods: `2` (ligand_contact, p2rank)
- Union residues: `16`
- Consensus residues: `11`
- Method-specific residues: `5`
- Effective minimum method count: `2`

## Risk Notes

- Truth residues: `7`
- Consensus truth coverage: `0.8571`
- Consensus truth precision: `0.5455`
- Missing truth risk: `0.1429`
- Overwide risk: `0.4545`
- Missing truth residues: `A:155`
- Extra consensus residues: `A:126, A:145, A:220, A:221, A:222`

## Method Summary

| method | residues | consensus_overlap | method_specific | count_to_median |
|---|---:|---:|---:|---:|
| ligand_contact | 13 | 11 | 2 | 0.9630 |
| p2rank | 14 | 11 | 3 | 1.0370 |

## Pairwise Overlap

| method_a | method_b | overlap | jaccard |
|---|---|---:|---:|
| ligand_contact | p2rank | 11 | 0.6875 |

## Outputs

- `consensus_pocket_residues.txt`: residues supported by the configured minimum number of methods.
- `union_pocket_residues.txt`: all residues found by at least one method.
- `residue_method_membership.csv`: per-residue support across methods.
- `method_specific_residues.csv`: residues found by exactly one method.
- `method_overlap_matrix.csv`: pairwise method overlap metrics.