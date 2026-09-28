#!/usr/bin/env python3
"""Convert docked ARM fragments into a PocketField fragment library.

The ARM docking CSV lists candidate "arms" as plain SMILES with no attachment
mark. PocketField fragments must carry a ``[*:1]`` dummy at the atom that bonds
to the anchor. In the reference decomposition the arm attaches to the anchor's
aryl ring via its amide carbonyl (``anchor-aryl-C(=O)-N(arm)``), so we cut the
``aryl-C(=O)`` bond and re-cap the carbonyl with ``[*:1]``::

    Ar-C(=O)-N(R)  ->  [*:1]C(=O)-N(R)

Only the aryl carbon *directly* bonded to the amide carbonyl is treated as the
anchor side (SMARTS ``[c][C](=[O])[N]``). Arms with a single such amide — the
overwhelming majority — are cut unambiguously; arms with more than one are cut
at the first match and reported for review.

Usage:
  python scripts/prepare_arm_fragments.py \
      --input test2/arm_dock_scores.csv \
      --output test2/fragments.csv
"""

from __future__ import annotations

import argparse
import csv
import sys

from rdkit import Chem

_ATTACH_PATT = Chem.MolFromSmarts("[c:1][C:2](=[O:3])[N:4]")


def cut_aryl_amide(smiles: str) -> tuple[str | None, int]:
    """Cut the first aryl-C(=O) amide bond and re-cap with [*:1].

    Returns ``(fragment_smiles, n_aromatic_amides)``. ``fragment_smiles`` is
    None when the SMILES cannot be parsed.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, -1
    matches = mol.GetSubstructMatches(_ATTACH_PATT)
    if not matches:
        return None, 0
    n_matches = len(matches)
    aryl_c, carbonyl_c, _o, _n = matches[0]

    # Collect every atom on the aryl side of the aryl-C(=O) bond (BFS from
    # aryl_c without crossing carbonyl_c), then delete them and re-cap.
    aryl_side: set[int] = set()
    stack = [aryl_c]
    while stack:
        atom = stack.pop()
        if atom in aryl_side:
            continue
        aryl_side.add(atom)
        for nb in mol.GetAtomWithIdx(atom).GetNeighbors():
            ni = nb.GetIdx()
            if ni != carbonyl_c and ni not in aryl_side:
                stack.append(ni)

    mol.GetAtomWithIdx(carbonyl_c).SetIsotope(999)
    rw = Chem.RWMol(mol)
    for idx in sorted(aryl_side, reverse=True):
        rw.RemoveAtom(idx)

    carb = None
    for atom in rw.GetAtoms():
        if atom.GetIsotope() == 999:
            carb = atom.GetIdx()
            atom.SetIsotope(0)
            break
    assert carb is not None

    dummy = Chem.Atom(0)
    dummy.SetAtomMapNum(1)
    di = rw.AddAtom(dummy)
    rw.AddBond(carb, di, Chem.BondType.SINGLE)

    result = rw.GetMol()
    Chem.SanitizeMol(result)
    return Chem.MolToSmiles(result), n_matches


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="ARM docking CSV with a SMILES column.")
    parser.add_argument("--output", required=True, help="Output fragment library CSV.")
    parser.add_argument("--smiles-col", default="arm_smiles", help="SMILES column name.")
    args = parser.parse_args(argv)

    with open(args.input, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if args.smiles_col not in rows[0]:
        raise ValueError(f"Column '{args.smiles_col}' not found in {args.input}")

    fieldnames = ["name", "smiles", "cluster", "vina_kcal_mol", "sb_dist_A", "salt_bridge"]
    out_rows: list[dict[str, str]] = []
    ambiguous: list[str] = []
    failed: list[str] = []
    n_ok = 0

    for index, row in enumerate(rows, start=1):
        smiles = str(row[args.smiles_col]).strip()
        fragment, n_amide = cut_aryl_amide(smiles)
        if fragment is None:
            failed.append(f"{index}: {smiles}")
            continue
        if n_amide > 1:
            ambiguous.append(f"{index}: {smiles} -> {fragment}")
        out_rows.append(
            {
                "name": f"arm_{row.get('cluster', index)}",
                "smiles": fragment,
                "cluster": str(row.get("cluster", "")),
                "vina_kcal_mol": str(row.get("vina_kcal_mol", "")),
                "sb_dist_A": str(row.get("sb_dist_A", "")),
                "salt_bridge": str(row.get("salt_bridge", "")),
            }
        )
        n_ok += 1

    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote {len(out_rows)} fragments to {args.output}")
    print(f"  unambiguous (1 aromatic amide): {n_ok - len(ambiguous)}")
    print(f"  ambiguous  (>1 aromatic amide): {len(ambiguous)}")
    print(f"  failed (parse/no amide):        {len(failed)}")
    for line in ambiguous:
        print(f"    AMBIGUOUS {line}")
    for line in failed:
        print(f"    FAILED    {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
