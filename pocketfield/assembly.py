"""RDKit molecule assembly and 3D alignment helpers.

Bonds fragments/linkers onto the anchor through mapped dummy atoms, removes
the dummies, strips isotopes, and aligns the assembled candidate toward a
target direction in the pocket.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _find_dummy(mol: Any, map_number: int) -> Any:
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0 and atom.GetAtomMapNum() == map_number:
            if len(atom.GetNeighbors()) != 1:
                raise ValueError("Dummy atoms must have exactly one neighbor.")
            return atom
    raise ValueError(f"Missing dummy attachment atom [*:{map_number}]")


def _attach_fragment(Chem: Any, base: Any, fragment: Any, map_number: int) -> Any:
    base_dummy = _find_dummy(base, map_number)
    frag_dummy = _find_dummy(fragment, 1)
    base_neighbor = base_dummy.GetNeighbors()[0].GetIdx()
    frag_neighbor = frag_dummy.GetNeighbors()[0].GetIdx()
    offset = base.GetNumAtoms()

    combined = Chem.CombineMols(base, fragment)
    rw = Chem.RWMol(combined)
    rw.AddBond(base_neighbor, offset + frag_neighbor, Chem.BondType.SINGLE)
    rw.GetAtomWithIdx(base_neighbor).SetIsotope(88)
    rw.GetAtomWithIdx(offset + frag_neighbor).SetIsotope(99)
    for idx in sorted([base_dummy.GetIdx(), offset + frag_dummy.GetIdx()], reverse=True):
        rw.RemoveAtom(idx)
    return rw.GetMol()


def _bridge_anchor_dummies(Chem: Any, base: Any, map_a: int, map_b: int, linker: Any) -> Any:
    linker_mol = Chem.MolFromSmiles(linker.smiles)
    if linker_mol is None:
        raise ValueError(f"Invalid linker SMILES: {linker.smiles}")
    linker_dummy_a = _find_dummy(linker_mol, 1)
    linker_dummy_b = _find_dummy(linker_mol, 2)
    linker_neighbor_a = linker_dummy_a.GetNeighbors()[0].GetIdx()
    linker_neighbor_b = linker_dummy_b.GetNeighbors()[0].GetIdx()

    for atom in linker_mol.GetAtoms():
        if atom.GetAtomicNum() != 0:
            atom.SetProp("pocketfield_fragment", linker.name)

    base_dummy_a = _find_dummy(base, map_a)
    base_dummy_b = _find_dummy(base, map_b)
    base_neighbor_a = base_dummy_a.GetNeighbors()[0].GetIdx()
    base_neighbor_b = base_dummy_b.GetNeighbors()[0].GetIdx()
    offset = base.GetNumAtoms()

    combined = Chem.CombineMols(base, linker_mol)
    rw = Chem.RWMol(combined)
    rw.AddBond(base_neighbor_a, offset + linker_neighbor_a, Chem.BondType.SINGLE)
    rw.AddBond(base_neighbor_b, offset + linker_neighbor_b, Chem.BondType.SINGLE)
    rw.GetAtomWithIdx(base_neighbor_a).SetIsotope(88)
    rw.GetAtomWithIdx(base_neighbor_a).SetIntProp("pocketfield_anchor_map", int(map_a))
    rw.GetAtomWithIdx(base_neighbor_b).SetIsotope(88)
    rw.GetAtomWithIdx(base_neighbor_b).SetIntProp("pocketfield_anchor_map", int(map_b))
    rw.GetAtomWithIdx(offset + linker_neighbor_a).SetIsotope(99)
    rw.GetAtomWithIdx(offset + linker_neighbor_b).SetIsotope(99)
    rw.GetAtomWithIdx(offset + linker_neighbor_a).SetIntProp("pocketfield_linker_map", 1)
    rw.GetAtomWithIdx(offset + linker_neighbor_b).SetIntProp("pocketfield_linker_map", 2)
    remove_indices = sorted(
        [
            base_dummy_a.GetIdx(),
            base_dummy_b.GetIdx(),
            offset + linker_dummy_a.GetIdx(),
            offset + linker_dummy_b.GetIdx(),
        ],
        reverse=True,
    )
    for idx in remove_indices:
        rw.RemoveAtom(idx)
    return rw.GetMol()


def _assemble_molecule(
    Chem: Any,
    anchor_smiles: str,
    attachments: list[dict[str, Any]],
    connect_anchor_dummies: bool = False,
) -> Any:
    mol = Chem.MolFromSmiles(anchor_smiles)
    if mol is None:
        raise ValueError(f"Invalid anchor SMILES: {anchor_smiles}")
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 0:
            atom.SetBoolProp("pocketfield_anchor", True)

    if connect_anchor_dummies and len(attachments) == 2:
        mol = _bridge_anchor_dummies(
            Chem,
            mol,
            int(attachments[0]["map"]),
            int(attachments[1]["map"]),
            attachments[0]["linker"],
        )
    else:
        for attachment in attachments:
            fragment = attachment["fragment"]
            frag = Chem.MolFromSmiles(fragment.smiles)
            if frag is None:
                raise ValueError(f"Invalid fragment SMILES: {fragment.smiles}")
            for atom in frag.GetAtoms():
                if atom.GetAtomicNum() != 0:
                    atom.SetProp("pocketfield_fragment", fragment.name)

            mol = _attach_fragment(Chem, mol, frag, int(attachment["map"]))

    mol = _remove_remaining_dummies(Chem, mol)
    Chem.SanitizeMol(mol)
    return mol


def _remove_remaining_dummies(Chem: Any, mol: Any) -> Any:
    rw = Chem.RWMol(mol)
    dummy_indices = [atom.GetIdx() for atom in rw.GetAtoms() if atom.GetAtomicNum() == 0]
    for idx in sorted(dummy_indices, reverse=True):
        rw.RemoveAtom(idx)
    return rw.GetMol()


def _strip_isotopes(Chem: Any, mol: Any) -> Any:
    clean = Chem.Mol(mol)
    for atom in clean.GetAtoms():
        atom.SetIsotope(0)
    return clean


def _align_candidate_to_field(mol: Any, center: np.ndarray, direction: np.ndarray, conf_id: int = 0) -> None:
    conf = mol.GetConformer(conf_id)
    positions = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    anchor_indices = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.HasProp("pocketfield_anchor") and atom.GetAtomicNum() > 1
    ]
    fragment_indices = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.HasProp("pocketfield_fragment") and atom.GetAtomicNum() > 1
    ]
    anchor_center = positions[anchor_indices].mean(axis=0) if anchor_indices else positions.mean(axis=0)
    fragment_center = (
        positions[fragment_indices].mean(axis=0) if fragment_indices else positions.mean(axis=0)
    )
    current = fragment_center - anchor_center
    if np.linalg.norm(current) < 1e-8:
        current = np.array([1.0, 0.0, 0.0])
    target = direction / max(np.linalg.norm(direction), 1e-8)
    rotation = _rotation_between(current, target)
    aligned = (positions - anchor_center) @ rotation.T + center
    for idx, xyz in enumerate(aligned):
        conf.SetAtomPosition(idx, tuple(float(v) for v in xyz))


def _rotation_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    a = source / max(np.linalg.norm(source), 1e-8)
    b = target / max(np.linalg.norm(target), 1e-8)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c > 1.0 - 1e-8:
        return np.eye(3)
    if c < -1.0 + 1e-8:
        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-8:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0]))
        axis = axis / np.linalg.norm(axis)
        return _axis_angle(axis, np.pi)
    skew = np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )
    return np.eye(3) + skew + skew @ skew * (1.0 / (1.0 + c))


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ]
    )


def _count_bad_bonds(mol: Any, threshold: float = 2.5, conf_id: int = 0) -> int:
    """Return number of bonds whose length exceeds *threshold* (Angstrom)."""
    conf = mol.GetConformer(conf_id)
    bad = 0
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if conf.GetAtomPosition(i).Distance(conf.GetAtomPosition(j)) > threshold:
            bad += 1
    return bad
