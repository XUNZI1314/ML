from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from pocket_io import normalize_residue_key


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract one residue file from P2Rank predictions.csv output.")
    parser.add_argument("--predictions_csv", required=True, help="Path to <protein>_predictions.csv")
    parser.add_argument("--out_file", required=True, help="Output residue file")
    parser.add_argument("--summary_json", default=None, help="Optional summary JSON")
    parser.add_argument("--rank", type=int, default=None, help="Select specific pocket rank")
    parser.add_argument("--name", default=None, help="Select specific pocket name, e.g. pocket2")
    parser.add_argument("--chain_filter", default=None, help="Keep only residues from this chain")
    parser.add_argument("--top_n", type=int, default=1, help="When rank/name not given, merge top N pockets after filtering")
    parser.add_argument("--min_probability", type=float, default=None, help="Optional minimum pocket probability filter")
    return parser


def _parse_residue_token(token: str) -> str:
    text = str(token).strip()
    if not text:
        raise ValueError("Empty residue token.")
    if "_" not in text:
        raise ValueError(f"Unsupported P2Rank residue token: {token!r}")
    parts = [p for p in text.split("_") if p != ""]
    if len(parts) < 2:
        raise ValueError(f"Unsupported P2Rank residue token: {token!r}")
    chain_id = parts[0]
    resseq = int(parts[1])
    icode = parts[2] if len(parts) >= 3 and len(parts[2]) == 1 and not parts[2].isdigit() else ""
    return normalize_residue_key(chain_id, resseq, icode)


def main() -> None:
    args = _build_parser().parse_args()
    predictions_csv = Path(args.predictions_csv).expanduser().resolve()
    if not predictions_csv.exists():
        raise FileNotFoundError(f"P2Rank predictions CSV not found: {predictions_csv}")

    df = pd.read_csv(predictions_csv, skipinitialspace=True)
    df.columns = [str(col).strip() for col in df.columns]
    if "rank" not in df.columns or "residue_ids" not in df.columns:
        raise ValueError("Expected columns 'rank' and 'residue_ids' in P2Rank predictions CSV.")

    work = df.copy()
    if args.min_probability is not None and "probability" in work.columns:
        work = work.loc[pd.to_numeric(work["probability"], errors="coerce") >= float(args.min_probability)].reset_index(drop=True)
    if args.rank is not None:
        work = work.loc[pd.to_numeric(work["rank"], errors="coerce") == int(args.rank)].reset_index(drop=True)
    if args.name:
        work = work.loc[work["name"].astype(str).str.strip() == str(args.name).strip()].reset_index(drop=True)
    if work.empty:
        raise ValueError("No P2Rank pockets left after filtering.")
    if args.rank is None and not args.name:
        work = work.sort_values(by="rank", ascending=True).head(max(1, int(args.top_n))).reset_index(drop=True)

    chain_filter = str(args.chain_filter).strip() if args.chain_filter else None
    residue_keys: list[str] = []
    selected_rows: list[dict[str, Any]] = []
    for _, row in work.iterrows():
        tokens = [tok for tok in str(row["residue_ids"]).split() if tok.strip()]
        pocket_keys: list[str] = []
        for token in tokens:
            key = _parse_residue_token(token)
            if chain_filter and key.split(":", 1)[0] != chain_filter:
                continue
            pocket_keys.append(key)
        residue_keys.extend(pocket_keys)
        selected_rows.append(
            {
                "name": str(row.get("name")).strip(),
                "rank": int(pd.to_numeric(pd.Series([row.get("rank")]), errors="coerce").iloc[0]),
                "score": float(pd.to_numeric(pd.Series([row.get("score")]), errors="coerce").iloc[0]),
                "probability": float(pd.to_numeric(pd.Series([row.get("probability")]), errors="coerce").iloc[0]) if "probability" in row.index else None,
                "residue_count_after_filter": int(len(pocket_keys)),
            }
        )

    residue_keys = sorted(set(residue_keys), key=lambda x: (x.split(":")[0], int(x.split(":")[1]), x.split(":")[2] if x.count(":") >= 2 else ""))
    out_file = Path(args.out_file).expanduser().resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(residue_keys) + ("\n" if residue_keys else ""), encoding="utf-8")

    summary = {
        "predictions_csv": str(predictions_csv),
        "selected_pockets": selected_rows,
        "chain_filter": chain_filter,
        "top_n": None if (args.rank is not None or args.name) else int(args.top_n),
        "residue_count": int(len(residue_keys)),
        "residue_keys": residue_keys,
    }
    if args.summary_json:
        summary_path = Path(args.summary_json).expanduser().resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=True, indent=2)

    print(f"Saved: {out_file}")
    if args.summary_json:
        print(f"Saved: {Path(args.summary_json).expanduser().resolve()}")
    print(f"Residues extracted: {len(residue_keys)}")


if __name__ == "__main__":
    main()
