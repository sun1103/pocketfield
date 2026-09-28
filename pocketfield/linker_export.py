"""Linker SMILES normalisation and probe annotation.

Helpers shared by :mod:`scripts.export_linkers` for converting database linker
SMILES into PocketField-compatible two-dummy linkers (``[*:1]...[*:2]``) and
deriving probe tags from linker atom types.

Requires RDKit.
"""

from __future__ import annotations

from rdkit import Chem


_ATOM_PROBES = {
    6:  {"hydrophobic"},
    7:  {"hbond_acceptor"},
    8:  {"hbond_acceptor"},
    9:  {"hydrophobic", "hbond_acceptor"},
    15: {"hydrophobic"},
    16: {"hydrophobic", "hbond_acceptor"},
    17: {"hydrophobic"},
    35: {"hydrophobic"},
    53: {"hydrophobic"},
}


def map_dummies(smi: str) -> list[str]:
    """Convert bare ``*`` atoms to mapped ``[*:1]``/``[*:2]``.

    Returns both orientations, deduplicated. Non-two-dummy SMILES return ``[]``.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return []

    stars = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if len(stars) != 2:
        return []

    seen: set[str] = set()

    # Orientation A: first * → [*:1], second * → [*:2]
    for a in mol.GetAtoms():
        if a.GetAtomicNum() == 0:
            a.SetAtomMapNum(0)
    stars[0].SetAtomMapNum(1)
    stars[1].SetAtomMapNum(2)
    seen.add(Chem.MolToSmiles(mol, canonical=True))

    # Orientation B
    stars[0].SetAtomMapNum(2)
    stars[1].SetAtomMapNum(1)
    seen.add(Chem.MolToSmiles(mol, canonical=True))

    return sorted(seen)


def compute_probes(smi: str) -> str:
    """Derive comma-separated probe tags from linker atom types."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""

    probes: set[str] = set()
    for atom in mol.GetAtoms():
        anum = atom.GetAtomicNum()
        if anum == 0:
            continue
        probes.update(_ATOM_PROBES.get(anum, set()))

        if anum == 6 and atom.GetIsAromatic():
            probes.add("aromatic")
        if anum == 7 and atom.GetTotalNumHs() > 0:
            probes.add("hbond_donor")
            if atom.GetFormalCharge() == 0:
                probes.add("positive")
        if anum == 8 and atom.GetTotalNumHs() > 0:
            probes.add("hbond_donor")
            if atom.GetFormalCharge() < 0:
                probes.add("negative")

    return ",".join(sorted(probes))
