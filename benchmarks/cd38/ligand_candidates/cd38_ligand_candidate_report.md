# CD38 Ligand Candidate Inspection

This report scans HETATM residues and flags ligand-like residues that contact the known CD38 truth residues.

## Summary

- Structures inspected: `3`
- HETATM residue candidates: `608`
- Recommended ligand-contact candidates: `3`
- Structures without active-site ligand candidates: `3F6Y`

## Recommended Ligand Candidates

| pdb_id | residue_label | atom_count | min_distance_to_truth | truth_contacts_within_threshold | recommendation_reason |
| --- | --- | --- | --- | --- | --- |
| 3ROP | A:50A:301 | 13 | 1.407 | 4 | ligand-like residue contacts known CD38 truth residues |
| 3ROP | A:NCA:302 | 9 | 2.541 | 5 | ligand-like residue contacts known CD38 truth residues |
| 4OGW | A:NMN:401 | 22 | 2.708 | 7 | ligand-like residue contacts known CD38 truth residues |

## Suggested Target Rows

| pdb_id | desired_methods | ligand_chain | ligand_resnames | ligand_resseqs | notes |
| --- | --- | --- | --- | --- | --- |
| 3ROP | ligand_contact,p2rank,fpocket | A | 50A,NCA |  | Current ligand-contact and P2Rank baseline; real fpocket output still needed. |
| 4OGW | ligand_contact,p2rank,fpocket | A | NMN |  | Current mutant ligand-contact and P2Rank baseline; real fpocket output still needed. |
| 3F6Y | p2rank,fpocket |  |  |  | Apo-like CD38 structure for pocket-finder testing; ligand scan found no active-site ligand candidate, so ligand-contact baseline is skipped. |

## Interpretation

- `recommended_for_ligand_contact=True` requires a ligand-like HETATM residue with contact to known CD38 truth residues.
- Water, common ions, buffers, and glycans are listed in the CSV but not recommended as ligand-contact baselines.
- If a structure has no recommended ligand candidate, use it for P2Rank/fpocket pocket detection rather than ligand-contact baseline.
