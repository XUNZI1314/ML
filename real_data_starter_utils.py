from __future__ import annotations

import csv
import json
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in fieldnames})
    path.write_text(buffer.getvalue(), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _input_pose_rows() -> list[dict[str, Any]]:
    return [
        {
            "nanobody_id": "NB001",
            "conformer_id": "CF001",
            "pose_id": "P001",
            "pdb_path": "data/pdb/NB001_CF001_P001.pdb",
            "antigen_chain": "A",
            "nanobody_chain": "H",
            "pocket_file": "pocket_residues_template.txt",
            "catalytic_file": "catalytic_residues_template.txt",
            "ligand_file": "data/ligand/substrate_template.pdb",
            "label": "",
        },
        {
            "nanobody_id": "NB001",
            "conformer_id": "CF001",
            "pose_id": "P002",
            "pdb_path": "data/pdb/NB001_CF001_P002.pdb",
            "antigen_chain": "A",
            "nanobody_chain": "H",
            "pocket_file": "pocket_residues_template.txt",
            "catalytic_file": "catalytic_residues_template.txt",
            "ligand_file": "data/ligand/substrate_template.pdb",
            "label": "",
        },
        {
            "nanobody_id": "NB002",
            "conformer_id": "CF001",
            "pose_id": "P001",
            "pdb_path": "data/pdb/NB002_CF001_P001.pdb",
            "antigen_chain": "A",
            "nanobody_chain": "H",
            "pocket_file": "pocket_residues_template.txt",
            "catalytic_file": "catalytic_residues_template.txt",
            "ligand_file": "data/ligand/substrate_template.pdb",
            "label": "",
        },
    ]


def _pose_feature_rows() -> list[dict[str, Any]]:
    return [
        {
            "nanobody_id": "NB001",
            "conformer_id": "CF001",
            "pose_id": "P001",
            "status": "ok",
            "pocket_hit_fraction": 0.72,
            "catalytic_hit_fraction": 0.21,
            "mouth_occlusion_score": 0.61,
            "ligand_path_block_score": 0.68,
            "min_distance_to_pocket": 2.8,
            "mmgbsa": -62.5,
            "interface_dg": -9.2,
            "label": "",
        },
        {
            "nanobody_id": "NB001",
            "conformer_id": "CF001",
            "pose_id": "P002",
            "status": "ok",
            "pocket_hit_fraction": 0.53,
            "catalytic_hit_fraction": 0.10,
            "mouth_occlusion_score": 0.43,
            "ligand_path_block_score": 0.47,
            "min_distance_to_pocket": 4.1,
            "mmgbsa": -49.8,
            "interface_dg": -7.5,
            "label": "",
        },
        {
            "nanobody_id": "NB002",
            "conformer_id": "CF001",
            "pose_id": "P001",
            "status": "ok",
            "pocket_hit_fraction": 0.31,
            "catalytic_hit_fraction": 0.04,
            "mouth_occlusion_score": 0.24,
            "ligand_path_block_score": 0.29,
            "min_distance_to_pocket": 6.7,
            "mmgbsa": -38.2,
            "interface_dg": -5.1,
            "label": "",
        },
    ]


def _experiment_override_rows() -> list[dict[str, Any]]:
    return [
        {
            "nanobody_id": "NB001",
            "plan_override": "include",
            "experiment_status": "pending",
            "experiment_result": "",
            "validation_label": "",
            "experiment_owner": "your_name",
            "experiment_cost": 2,
            "experiment_note": "High-priority candidate from real-data run.",
            "manual_plan_reason": "Manual priority from domain knowledge.",
        },
        {
            "nanobody_id": "NB002",
            "plan_override": "standby",
            "experiment_status": "planned",
            "experiment_result": "",
            "validation_label": "",
            "experiment_owner": "",
            "experiment_cost": 1,
            "experiment_note": "Backup candidate.",
            "manual_plan_reason": "",
        },
    ]


def _mini_pose_specs() -> list[dict[str, Any]]:
    """Return small toy PDB poses used to validate the raw-input path."""
    specs: list[dict[str, Any]] = []
    raw_specs = [
        ("NB001", "CF001", "P001", 1.7, 0.0, 0.0, 1),
        ("NB001", "CF001", "P002", 3.0, 0.4, -0.2, 1),
        ("NB001", "CF002", "P003", 6.6, 1.0, 0.4, 0),
        ("NB002", "CF001", "P001", 2.4, -0.3, 0.5, 1),
        ("NB002", "CF001", "P002", 5.2, 0.8, 0.8, 0),
        ("NB002", "CF002", "P003", 7.8, -1.0, 0.2, 0),
        ("NB003", "CF001", "P001", 2.0, 0.2, 0.9, 1),
        ("NB003", "CF001", "P002", 5.8, 1.2, -0.6, 0),
        ("NB003", "CF002", "P003", 8.8, -1.4, -0.2, 0),
        ("NB004", "CF001", "P001", 1.5, -0.6, -0.6, 1),
        ("NB004", "CF001", "P002", 3.8, 0.0, 1.1, 1),
        ("NB004", "CF002", "P003", 9.4, 1.6, 0.6, 0),
    ]
    for nanobody_id, conformer_id, pose_id, z_offset, x_shift, y_shift, label in raw_specs:
        pdb_name = f"{nanobody_id}_{conformer_id}_{pose_id}.pdb"
        specs.append(
            {
                "nanobody_id": nanobody_id,
                "conformer_id": conformer_id,
                "pose_id": pose_id,
                "pdb_name": pdb_name,
                "z_offset": float(z_offset),
                "x_shift": float(x_shift),
                "y_shift": float(y_shift),
                "label": int(label),
            }
        )
    return specs


def _mini_input_pose_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in _mini_pose_specs():
        rows.append(
            {
                "nanobody_id": spec["nanobody_id"],
                "conformer_id": spec["conformer_id"],
                "pose_id": spec["pose_id"],
                "pdb_path": f"data/pdb/{spec['pdb_name']}",
                "antigen_chain": "A",
                "nanobody_chain": "H",
                "pocket_file": "pocket_residues.txt",
                "catalytic_file": "catalytic_residues.txt",
                "ligand_file": "data/ligand/substrate_template.pdb",
                "label": spec["label"],
            }
        )
    return rows


def _format_pdb_atom(
    serial: int,
    record: str,
    atom_name: str,
    resname: str,
    chain_id: str,
    resseq: int,
    x: float,
    y: float,
    z: float,
    element: str,
) -> str:
    """Format one simple PDB ATOM/HETATM line."""
    return (
        f"{record:<6}{serial:5d} {atom_name:^4} {resname:>3} {chain_id:1}{resseq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{20.00:6.2f}          {element:>2}"
    )


def _append_residue_atoms(
    lines: list[str],
    *,
    serial_start: int,
    resname: str,
    chain_id: str,
    resseq: int,
    center: tuple[float, float, float],
) -> int:
    """Append a tiny heavy-atom residue around a center point."""
    atom_specs = [
        ("N", (-0.78, -0.35, -0.12), "N"),
        ("CA", (0.00, 0.00, 0.00), "C"),
        ("C", (0.82, 0.38, 0.08), "C"),
        ("O", (1.18, -0.32, 0.16), "O"),
        ("CB", (-0.12, 0.92, 0.42), "C"),
    ]
    serial = int(serial_start)
    cx, cy, cz = center
    for atom_name, (dx, dy, dz), element in atom_specs:
        if resname == "GLY" and atom_name == "CB":
            continue
        lines.append(
            _format_pdb_atom(
                serial=serial,
                record="ATOM",
                atom_name=atom_name,
                resname=resname,
                chain_id=chain_id,
                resseq=int(resseq),
                x=float(cx + dx),
                y=float(cy + dy),
                z=float(cz + dz),
                element=element,
            )
        )
        serial += 1
    return serial


def _toy_complex_pdb_text(z_offset: float, x_shift: float = 0.0, y_shift: float = 0.0) -> str:
    """Build a compact two-chain toy complex PDB.

    Chain A is the antigen. Chain H is the nanobody-like binder. Moving chain H
    along z changes whether it covers the toy pocket.
    """
    lines: list[str] = [
        "REMARK Toy two-chain complex for ML input-path validation only.",
        "REMARK Chain A antigen residues 37-40 and 45 define a toy pocket.",
        "REMARK Chain H is shifted to create close and far blocking examples.",
    ]
    serial = 1
    antigen_centers = [
        ("GLY", 37, (0.0, 0.0, 0.0)),
        ("SER", 38, (3.0, 0.0, 0.1)),
        ("TYR", 39, (0.0, 3.0, -0.1)),
        ("ASP", 40, (3.0, 3.0, 0.0)),
        ("HIS", 45, (1.5, 1.5, 1.1)),
        ("GLU", 67, (7.0, 0.4, 0.2)),
    ]
    for resname, resseq, center in antigen_centers:
        serial = _append_residue_atoms(
            lines,
            serial_start=serial,
            resname=resname,
            chain_id="A",
            resseq=resseq,
            center=center,
        )

    lines.append(f"TER   {serial:5d}      GLU A  67")
    serial += 1

    z = float(z_offset)
    xs = float(x_shift)
    ys = float(y_shift)
    nanobody_centers = [
        ("TYR", 101, (1.2 + xs, 1.1 + ys, z)),
        ("SER", 102, (2.5 + xs, 1.4 + ys, z + 0.3)),
        ("ASN", 103, (1.4 + xs, 2.8 + ys, z + 0.1)),
        ("LYS", 104, (3.3 + xs, 3.1 + ys, z + 0.6)),
        ("GLY", 105, (5.4 + xs, 1.5 + ys, z + 0.9)),
    ]
    for resname, resseq, center in nanobody_centers:
        serial = _append_residue_atoms(
            lines,
            serial_start=serial,
            resname=resname,
            chain_id="H",
            resseq=resseq,
            center=center,
        )

    lines.append(f"TER   {serial:5d}      GLY H 105")
    lines.append("END")
    return "\n".join(lines) + "\n"


def _toy_ligand_pdb_text() -> str:
    lines = [
        "REMARK Toy ligand template near antigen chain A pocket.",
        _format_pdb_atom(1, "HETATM", "C1", "LIG", "L", 1, 1.4, 1.3, 0.5, "C"),
        _format_pdb_atom(2, "HETATM", "O1", "LIG", "L", 1, 2.0, 1.6, 0.7, "O"),
        _format_pdb_atom(3, "HETATM", "N1", "LIG", "L", 1, 1.1, 2.1, 0.8, "N"),
        "END",
    ]
    return "\n".join(lines) + "\n"


def write_mini_pdb_example(out_dir: str | Path) -> dict[str, Any]:
    """Write a runnable tiny PDB example for build_feature_table.py."""
    root = Path(out_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    pdb_dir = root / "data" / "pdb"
    ligand_dir = root / "data" / "ligand"
    input_csv = root / "input_pose_table.csv"
    pocket_file = root / "pocket_residues.txt"
    catalytic_file = root / "catalytic_residues.txt"
    ligand_file = ligand_dir / "substrate_template.pdb"
    readme_md = root / "README_MINI_PDB_EXAMPLE.md"
    manifest_json = root / "mini_pdb_example_manifest.json"

    input_fields = [
        "nanobody_id",
        "conformer_id",
        "pose_id",
        "pdb_path",
        "antigen_chain",
        "nanobody_chain",
        "pocket_file",
        "catalytic_file",
        "ligand_file",
        "label",
    ]
    _write_csv(input_csv, _mini_input_pose_rows(), input_fields)
    _write_text(
        pocket_file,
        "\n".join(
            [
                "# Toy pocket on chain A. A:37-40 expands to A:37,A:38,A:39,A:40.",
                "A:37-40",
                "A:45",
                "",
            ]
        ),
    )
    _write_text(
        catalytic_file,
        "\n".join(
            [
                "# Toy functional residues on chain A.",
                "A:45",
                "A:67",
                "",
            ]
        ),
    )
    _write_text(ligand_file, _toy_ligand_pdb_text())

    pdb_outputs: list[str] = []
    for spec in _mini_pose_specs():
        pdb_path = pdb_dir / str(spec["pdb_name"])
        _write_text(
            pdb_path,
            _toy_complex_pdb_text(
                z_offset=float(spec["z_offset"]),
                x_shift=float(spec["x_shift"]),
                y_shift=float(spec["y_shift"]),
            ),
        )
        pdb_outputs.append(str(pdb_path))

    readme_lines = [
        "# Runnable Mini PDB Example",
        "",
        "This folder is a tiny toy dataset for checking the real `input_csv -> build_feature_table.py` path.",
        "It is structurally valid enough for parser/geometry smoke testing, but it is not a biological benchmark.",
        "",
        "## What Is Inside",
        "",
        "- `input_pose_table.csv`: 12 toy nanobody-antigen poses with explicit `A` antigen and `H` nanobody chains.",
        "- `data/pdb/*.pdb`: compact two-chain toy complex PDB files.",
        "- `data/ligand/substrate_template.pdb`: compact toy ligand template near the pocket.",
        "- `pocket_residues.txt`: uses `A:37-40`, equivalent to `A:37,A:38,A:39,A:40`.",
        "- `catalytic_residues.txt`: toy functional residues.",
        "",
        "## Build Features",
        "",
        "Run from the repository root and replace `<REAL_DATA_STARTER>` with this starter folder path:",
        "",
        "```bash",
        "python build_feature_table.py --input_csv <REAL_DATA_STARTER>/MINI_PDB_EXAMPLE/input_pose_table.csv --out_csv <REAL_DATA_STARTER>/MINI_PDB_EXAMPLE/mini_pose_features.csv",
        "```",
        "",
        "## Run The Recommended Pipeline",
        "",
        "```bash",
        "python run_recommended_pipeline.py --input_csv <REAL_DATA_STARTER>/MINI_PDB_EXAMPLE/input_pose_table.csv --out_dir <REAL_DATA_STARTER>/MINI_PDB_EXAMPLE/mini_outputs --train_epochs 1 --disable_label_aware_steps",
        "```",
        "",
        "## Expected Boundary",
        "",
        "Close chain-H poses have label `1`; far chain-H poses have label `0`. These labels only test workflow behavior and should not be used as experimental evidence.",
        "",
    ]
    _write_text(readme_md, "\n".join(readme_lines))

    manifest = {
        "manifest_type": "mini_pdb_example",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "row_count": len(_mini_input_pose_rows()),
        "pdb_count": len(pdb_outputs),
        "boundary": "Toy PDBs validate parser and geometry workflow only; not biological evidence.",
        "outputs": {
            "readme_md": str(readme_md),
            "input_pose_table_csv": str(input_csv),
            "pocket_residues_txt": str(pocket_file),
            "catalytic_residues_txt": str(catalytic_file),
            "ligand_template_pdb": str(ligand_file),
            "pdb_files": pdb_outputs,
            "manifest_json": str(manifest_json),
        },
    }
    _write_text(manifest_json, json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def write_real_data_starter_kit(out_dir: str | Path) -> dict[str, Any]:
    """Write a small starter kit for migrating from demo data to real project data."""

    root = Path(out_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    input_csv = root / "input_pose_table_template.csv"
    feature_csv = root / "pose_features_template.csv"
    override_csv = root / "experiment_plan_override_template.csv"
    pocket_file = root / "pocket_residues_template.txt"
    catalytic_file = root / "catalytic_residues_template.txt"
    folder_layout = root / "folder_layout.txt"
    checklist_csv = root / "real_data_checklist.csv"
    readme_md = root / "README_REAL_DATA_STARTER.md"
    manifest_json = root / "real_data_starter_manifest.json"
    mini_example_dir = root / "MINI_PDB_EXAMPLE"

    input_fields = [
        "nanobody_id",
        "conformer_id",
        "pose_id",
        "pdb_path",
        "antigen_chain",
        "nanobody_chain",
        "pocket_file",
        "catalytic_file",
        "ligand_file",
        "label",
    ]
    feature_fields = [
        "nanobody_id",
        "conformer_id",
        "pose_id",
        "status",
        "pocket_hit_fraction",
        "catalytic_hit_fraction",
        "mouth_occlusion_score",
        "ligand_path_block_score",
        "min_distance_to_pocket",
        "mmgbsa",
        "interface_dg",
        "label",
    ]
    override_fields = [
        "nanobody_id",
        "plan_override",
        "experiment_status",
        "experiment_result",
        "validation_label",
        "experiment_owner",
        "experiment_cost",
        "experiment_note",
        "manual_plan_reason",
    ]

    _write_csv(input_csv, _input_pose_rows(), input_fields)
    _write_csv(feature_csv, _pose_feature_rows(), feature_fields)
    _write_csv(override_csv, _experiment_override_rows(), override_fields)
    _write_text(
        pocket_file,
        "\n".join(
            [
                "# Replace these examples with target pocket residues.",
                "# Supported syntax examples:",
                "A:37-40",
                "A:45,67",
                "B:102A",
                "",
            ]
        ),
    )
    _write_text(
        catalytic_file,
        "\n".join(
            [
                "# Replace these examples with known catalytic / functional residues.",
                "A:45",
                "A:67",
                "",
            ]
        ),
    )
    _write_text(
        folder_layout,
        "\n".join(
            [
                "real_project/",
                "  input_pose_table.csv",
                "  pocket_residues.txt",
                "  catalytic_residues.txt",
                "  data/",
                "    pdb/",
                "      NB001_CF001_P001.pdb",
                "      NB001_CF001_P002.pdb",
                "    ligand/",
                "      substrate_template.pdb",
                "  outputs/",
                "  MINI_PDB_EXAMPLE/",
                "    input_pose_table.csv",
                "    pocket_residues.txt",
                "    catalytic_residues.txt",
                "    data/pdb/*.pdb",
                "    data/ligand/substrate_template.pdb",
                "",
            ]
        ),
    )
    _write_csv(
        checklist_csv,
        [
            {"step": 1, "check_item": "Every pdb_path exists on this machine.", "why": "Missing PDB paths block feature building."},
            {"step": 2, "check_item": "nanobody_id groups all poses from the same candidate.", "why": "Ranking aggregates pose -> conformer -> nanobody."},
            {"step": 3, "check_item": "pocket/catalytic residue files use chain-aware notation such as A:37-40.", "why": "Chain-aware residue mapping avoids wrong-site scoring."},
            {"step": 4, "check_item": "label is blank unless you have true labels.", "why": "Do not mix synthetic labels with real experiment labels."},
            {"step": 5, "check_item": "experiment_result or validation_label is explicit before retraining.", "why": "completed status alone is not treated as biological evidence."},
            {"step": 6, "check_item": "Use MINI_PDB_EXAMPLE only as a parser/workflow check.", "why": "Toy labels and toy coordinates are not biological validation."},
        ],
        ["step", "check_item", "why"],
    )

    mini_manifest = write_mini_pdb_example(mini_example_dir)

    readme_lines = [
        "# Real Data Starter Kit",
        "",
        "This folder helps you replace the synthetic demo with your own project data.",
        "",
        "## Recommended Path",
        "",
        "1. Copy this folder next to your real PDB files.",
        "2. Rename `input_pose_table_template.csv` to `input_pose_table.csv`.",
        "3. Replace every `pdb_path` with a real local PDB path.",
        "4. Replace `pocket_residues_template.txt` and `catalytic_residues_template.txt` with your real residue definitions.",
        "5. Leave `label` blank unless you have true labels.",
        "6. Run:",
        "",
        "```bash",
        "python run_recommended_pipeline.py --input_csv input_pose_table.csv --out_dir real_outputs",
        "```",
        "",
        "## Files",
        "",
        "- `input_pose_table_template.csv`: best starting point when you have PDB structures.",
        "- `pose_features_template.csv`: use only if features were already built elsewhere.",
        "- `experiment_plan_override_template.csv`: optional manual include/exclude/status/validation file.",
        "- `pocket_residues_template.txt`: pocket residue notation examples.",
        "- `catalytic_residues_template.txt`: functional residue notation examples.",
        "- `real_data_checklist.csv`: quick checks before a real run.",
        "",
        "## Runnable Mini PDB Example",
        "",
        "`MINI_PDB_EXAMPLE/` contains a tiny toy PDB dataset that can run through the real `input_csv -> build_feature_table.py` path.",
        "It is for parser, residue mapping and geometry-workflow checks only; it is not biological evidence.",
        "",
        "From the repository root, run:",
        "",
        "```bash",
        "python build_feature_table.py --input_csv <REAL_DATA_STARTER>/MINI_PDB_EXAMPLE/input_pose_table.csv --out_csv <REAL_DATA_STARTER>/MINI_PDB_EXAMPLE/mini_pose_features.csv",
        "```",
        "",
        "Then optionally run the whole pipeline:",
        "",
        "```bash",
        "python run_recommended_pipeline.py --input_csv <REAL_DATA_STARTER>/MINI_PDB_EXAMPLE/input_pose_table.csv --out_dir <REAL_DATA_STARTER>/MINI_PDB_EXAMPLE/mini_outputs --train_epochs 1 --disable_label_aware_steps",
        "```",
        "",
        "Read `MINI_PDB_EXAMPLE/README_MINI_PDB_EXAMPLE.md` before interpreting the outputs.",
        "",
        "## Important Boundary",
        "",
        "The demo labels are synthetic. For a real project, only explicit `experiment_result=positive/negative` or `validation_label=1/0` should be treated as validation evidence.",
        "",
    ]
    _write_text(readme_md, "\n".join(readme_lines))

    manifest = {
        "manifest_type": "real_data_starter_kit",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "outputs": {
            "readme_md": str(readme_md),
            "input_pose_table_template_csv": str(input_csv),
            "pose_features_template_csv": str(feature_csv),
            "experiment_plan_override_template_csv": str(override_csv),
            "pocket_residues_template_txt": str(pocket_file),
            "catalytic_residues_template_txt": str(catalytic_file),
            "folder_layout_txt": str(folder_layout),
            "real_data_checklist_csv": str(checklist_csv),
            "mini_pdb_example_dir": str(mini_example_dir),
            "mini_pdb_example_readme_md": str(mini_manifest.get("outputs", {}).get("readme_md") or ""),
            "mini_pdb_example_input_csv": str(mini_manifest.get("outputs", {}).get("input_pose_table_csv") or ""),
            "mini_pdb_example_manifest_json": str(mini_manifest.get("outputs", {}).get("manifest_json") or ""),
            "manifest_json": str(manifest_json),
        },
        "mini_pdb_example": mini_manifest,
    }
    _write_text(manifest_json, json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


__all__ = ["write_real_data_starter_kit", "write_mini_pdb_example"]
