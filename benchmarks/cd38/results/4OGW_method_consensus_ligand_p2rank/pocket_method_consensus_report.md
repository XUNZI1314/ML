# Pocket Method Consensus Analysis

- Methods: `2` (ligand_contact, p2rank)
- Union residues: `27`
- Consensus residues: `13`
- Method-specific residues: `14`
- Effective minimum method count: `2`

## Risk Notes

- Truth residues: `7`
- Consensus truth coverage: `1.0000`
- Consensus truth precision: `0.5385`
- Missing truth risk: `0.0000`
- Overwide risk: `0.4615`
- Extra consensus residues: `A:124, A:126, A:129, A:145, A:221, A:222`

## Method Summary

| method | residues | consensus_overlap | method_specific | count_to_median |
|---|---:|---:|---:|---:|
| ligand_contact | 14 | 13 | 1 | 0.7000 |
| p2rank | 26 | 13 | 13 | 1.3000 |

## Pairwise Overlap

| method_a | method_b | overlap | jaccard |
|---|---|---:|---:|
| ligand_contact | p2rank | 13 | 0.4815 |

## Outputs

- `consensus_pocket_residues.txt`: residues supported by the configured minimum number of methods.
- `union_pocket_residues.txt`: all residues found by at least one method.
- `residue_method_membership.csv`: per-residue support across methods.
- `method_specific_residues.csv`: residues found by exactly one method.
- `method_overlap_matrix.csv`: pairwise method overlap metrics.