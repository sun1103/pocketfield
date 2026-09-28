#!/usr/bin/env python3
"""Export frag_linker entries as PocketField-compatible linker CSV.

Converts bare * atoms to mapped [*:1]/[*:2], generates both orientations
for asymmetric linkers, deduplicates, and writes a minimal CSV.

Usage:
  conda run -n rxn-env python3 scripts/export_linkers.py --output linkers.csv
  conda run -n rxn-env python3 scripts/export_linkers.py --sources enumerated_spiro --output spiro_linkers.csv
"""

import argparse
import csv
import sys
from pathlib import Path

from rdkit import Chem
from rdkit import RDLogger

RDLogger.logger().setLevel(RDLogger.ERROR)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pocketfield.linker_export import map_dummies, compute_probes

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "fragenv" / "scripts"))
from vendor_pipeline.load_fast import _connect_db, DB_URL


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export frag_linker entries as PocketField-compatible linker CSV"
    )
    parser.add_argument("--output", default="linkers_export.csv", help="Output CSV path")
    parser.add_argument(
        "--sources", default="enumerated_biaryl,enumerated_spiro",
        help="Comma-separated source names to filter (default: enumerated_biaryl,enumerated_spiro)"
    )
    parser.add_argument("--max-n-heavy", type=int, default=18, help="Max heavy atoms (default: 18)")
    parser.add_argument("--min-n-heavy", type=int, default=4, help="Min heavy atoms (default: 4)")
    parser.add_argument("--max-n-rings", type=int, default=None, help="Max ring count")
    parser.add_argument("--min-attach-span", type=int, default=None, help="Min attach_span")
    parser.add_argument("--max-attach-span", type=int, default=None, help="Max attach_span")
    parser.add_argument(
        "--attach-elements",
        default=None,
        help="Comma-separated attach element kinds (e.g. 'C,N'). Filters linkers "
             "where BOTH attachments match any of these elements."
    )
    parser.add_argument("--limit", type=int, default=None, help="Max DB rows to fetch")
    parser.add_argument("--with-probes", action="store_true", help="Include probes column")
    parser.add_argument("--dry-run", action="store_true", help="Count without exporting")
    args = parser.parse_args()

    conn = _connect_db(DB_URL)
    cur = conn.cursor()

    source_list = [s.strip() for s in args.sources.split(",")]

    conditions = ["n_heavy >= %s", "n_heavy <= %s"]
    params: list = [args.min_n_heavy, args.max_n_heavy]

    if args.max_n_rings is not None:
        conditions.append("n_rings <= %s")
        params.append(args.max_n_rings)
    if args.min_attach_span is not None:
        conditions.append("attach_span >= %s")
        params.append(args.min_attach_span)
    if args.max_attach_span is not None:
        conditions.append("attach_span <= %s")
        params.append(args.max_attach_span)
    if source_list:
        conditions.append("sources && %s::text[]")
        params.append(source_list)
    if args.attach_elements:
        elements = [e.strip() for e in args.attach_elements.split(",")]
        conditions.append("attach_elements <@ %s::varchar[]")
        params.append(elements)

    query = f"SELECT DISTINCT smiles FROM frag_linker WHERE {' AND '.join(conditions)}"
    limit_clause = f" LIMIT {int(args.limit)}" if args.limit else ""
    cur.execute(query + limit_clause, params)
    rows = cur.fetchall()
    print(f"DB rows: {len(rows):,}")

    if args.dry_run:
        cur.close()
        conn.close()
        return

    # Convert and deduplicate
    exported: dict[str, str] = {}  # canonical_smi -> name (db_smi prefix)
    for (smi,) in rows:
        for mapped in map_dummies(smi):
            if mapped not in exported:
                exported[mapped] = smi

    # Build header
    fieldnames = ["smiles"]
    if args.with_probes:
        fieldnames.append("probes")

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for mapped_smi in sorted(exported):
            row = {"smiles": mapped_smi}
            if args.with_probes:
                row["probes"] = compute_probes(mapped_smi)
            writer.writerow(row)

    print(f"Exported {len(exported):,} unique mapped SMILES to {args.output}")
    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
