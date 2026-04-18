# CD38 Benchmark Expansion Plan

This report audits which CD38 structures and pocket methods are already covered, and what is still needed before expanding the benchmark panel.

## Summary

- Target structures: `3`
- Target structure-method pairs: `8`
- Missing or pending pairs: `4`
- Structure targets CSV: `D:\minimal_ML_test\benchmarks\cd38\cd38_structure_targets.csv`
- Manifest CSV: `D:\minimal_ML_test\benchmarks\cd38\cd38_benchmark_manifest.csv`
- Panel CSV: `D:\minimal_ML_test\benchmarks\cd38\results\cd38_benchmark_panel.csv`

Status counts:

- `complete`: `4`
- `needs_fpocket_output`: `3`
- `needs_p2rank_output`: `1`

## Plan Table

| pdb_id | method | status | manifest_row_count | result_count | recommended_action |
| --- | --- | --- | --- | --- | --- |
| 3ROP | ligand_contact | complete | 1 | 1 | Result already exists; keep it in the summary panel. |
| 3ROP | p2rank | complete | 1 | 1 | Result already exists; keep it in the summary panel. |
| 3ROP | fpocket | needs_fpocket_output | 0 | 0 | Run fpocket externally and fill fpocket_root, then generate a discovered manifest. |
| 4OGW | ligand_contact | complete | 1 | 1 | Result already exists; keep it in the summary panel. |
| 4OGW | p2rank | complete | 1 | 1 | Result already exists; keep it in the summary panel. |
| 4OGW | fpocket | needs_fpocket_output | 0 | 0 | Run fpocket externally and fill fpocket_root, then generate a discovered manifest. |
| 3F6Y | p2rank | needs_p2rank_output | 0 | 0 | Run P2Rank externally, then generate a discovered manifest from its predictions.csv output. |
| 3F6Y | fpocket | needs_fpocket_output | 0 | 0 | Run fpocket externally and fill fpocket_root, then generate a discovered manifest. |

## Recommended Commands

### 3ROP fpocket (needs_fpocket_output)

```powershell
python prepare_cd38_fpocket_panel.py --fpocket_root your_fpocket_outputs\3ROP_out --rcsb_pdb_id 3ROP --chain_filter A --results_root benchmarks\cd38\results --near_threshold 4.5 --manifest_out benchmarks\cd38\fpocket_3ROP_manifest.csv
```

### 4OGW fpocket (needs_fpocket_output)

```powershell
python prepare_cd38_fpocket_panel.py --fpocket_root your_fpocket_outputs\4OGW_out --rcsb_pdb_id 4OGW --chain_filter A --results_root benchmarks\cd38\results --near_threshold 4.5 --manifest_out benchmarks\cd38\fpocket_4OGW_manifest.csv
```

### 3F6Y p2rank (needs_p2rank_output)

```powershell
python prepare_cd38_p2rank_panel.py --p2rank_root your_p2rank_outputs\3F6Y --rcsb_pdb_id 3F6Y --chain_filter A --results_root benchmarks\cd38\results --near_threshold 4.5 --manifest_out benchmarks\cd38\p2rank_3F6Y_manifest.csv
```

### 3F6Y fpocket (needs_fpocket_output)

```powershell
python prepare_cd38_fpocket_panel.py --fpocket_root your_fpocket_outputs\3F6Y_out --rcsb_pdb_id 3F6Y --chain_filter A --results_root benchmarks\cd38\results --near_threshold 4.5 --manifest_out benchmarks\cd38\fpocket_3F6Y_manifest.csv
```

## Interpretation

- `complete` means a result already exists in the summarized benchmark panel.
- `manifest_ready` means the manifest has a row but the result directory is missing.
- `needs_ligand_metadata` means the structure may be usable, but ligand chain/name or residue number still needs curation.
- `needs_p2rank_output` and `needs_fpocket_output` mean the external tool output must be generated before this repository can benchmark it.
