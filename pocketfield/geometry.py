"""Anchor placement and 3D geometry helpers.

Reading anchor structures (SDF/MOL/PDB), extracting anchor poses and dummy
exit vectors, rigid alignment (Kabsch / axis-angle rotations), and assigning
pocket sectors to each anchor dummy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _read_anchor_structure(Chem: Any, path: str | Path) -> Any:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".sdf":
        supplier = Chem.SDMolSupplier(str(source), removeHs=False, sanitize=False)
        mol = supplier[0] if supplier and len(supplier) else None
    elif suffix == ".mol":
        mol = Chem.MolFromMolFile(str(source), removeHs=False, sanitize=False)
    elif suffix == ".pdb":
        mol = Chem.MolFromPDBFile(str(source), removeHs=False, sanitize=False)
    else:
        raise ValueError("--anchor-structure must be .sdf, .mol, or .pdb")
    if mol is None:
        raise ValueError(f"Could not read anchor structure: {source}")
    if mol.GetNumConformers() == 0:
        raise ValueError("--anchor-structure must contain 3D coordinates.")
    return mol


def _anchor_pose(Chem: Any, anchor_structure: str | Path | None) -> dict[str, Any] | None:
    if anchor_structure is None:
        return None
    mol = _read_anchor_structure(Chem, anchor_structure)
    conf = mol.GetConformer()
    heavy_coords = []
    dummy_coords: dict[int, np.ndarray] = {}
    dummy_neighbor_coords: dict[int, np.ndarray] = {}

    for atom in mol.GetAtoms():
        pos = np.asarray(list(conf.GetAtomPosition(atom.GetIdx())), dtype=np.float64)
        if atom.GetAtomicNum() == 0:
            anchor_map = atom.GetIsotope() or atom.GetAtomMapNum()
            if anchor_map <= 0:
                continue
            dummy_coords[anchor_map] = pos
            neighbors = atom.GetNeighbors()
            if len(neighbors) == 1:
                dummy_neighbor_coords[anchor_map] = np.asarray(
                    list(conf.GetAtomPosition(neighbors[0].GetIdx())),
                    dtype=np.float64,
                )
        elif atom.GetAtomicNum() > 1:
            heavy_coords.append(pos)

    return {
        "heavy_coords": np.asarray(heavy_coords, dtype=np.float64),
        "dummy_coords": dummy_coords,
        "dummy_neighbor_coords": dummy_neighbor_coords,
    }


def _anchor_coord_map(
    Chem: Any,
    mol: Any,
    anchor_smiles: str,
    anchor_structure: str | Path | None,
    linker_fit_mode: str = "rigid",
) -> tuple[dict[int, Any], tuple[float, float, float]]:
    """Build a coordMap for EmbedMolecule from the full anchor structure.

    Returns ``(coordMap, translation)`` where *coordMap* positions are
    centred at the origin and *translation* is the SDF centroid that
    should be added back after embedding.

    Matches each fragment from *anchor_smiles* against both the assembled
    *mol* and the anchor SDF, then constrains every matched fragment atom
    to its docked coordinate.  Only linker atoms are left free.

    Falls back to constraining only the two connection-point atoms (tagged
    with ``pocketfield_anchor_map``) when full substructure matching fails.

    ``linker_fit_mode`` controls bridge-mode rigidity. ``"rigid"`` (default)
    pins both anchor components and both linker attachment atoms to their
    docked coordinates; ``"flexible"`` pins only the first anchor component
    (the smallest anchor map) and its linker attachment, leaving the second
    component and the rest of the linker free to close the gap.
    """
    from rdkit import Geometry

    zero_translation = (0.0, 0.0, 0.0)

    if anchor_structure is None:
        return {}, zero_translation

    anchor_mol = _read_anchor_structure(Chem, anchor_structure)
    try:
        Chem.SanitizeMol(anchor_mol)
    except Exception:
        pass
    anchor_conf = anchor_mol.GetConformer()

    coordMap: dict[int, Any] = {}
    fixed_map: int | None = None
    if linker_fit_mode == "flexible":
        anchor_maps = _anchor_maps(Chem, anchor_smiles)
        fixed_map = anchor_maps[0] if anchor_maps else None

    # ---- primary: full substructure matching ----
    for frag_smi in anchor_smiles.split("."):
        frag_mol = Chem.MolFromSmiles(frag_smi)
        if frag_mol is None:
            continue
        if fixed_map is not None:
            frag_map = next(
                (
                    atom.GetAtomMapNum() or atom.GetIsotope()
                    for atom in frag_mol.GetAtoms()
                    if atom.GetAtomicNum() == 0
                ),
                None,
            )
            if frag_map != fixed_map:
                continue
        frag_no_dummy = Chem.RWMol(frag_mol)
        for idx in sorted(
            [a.GetIdx() for a in frag_no_dummy.GetAtoms() if a.GetAtomicNum() == 0],
            reverse=True,
        ):
            frag_no_dummy.RemoveAtom(idx)
        frag_query = frag_no_dummy.GetMol()
        Chem.SanitizeMol(frag_query)

        anchor_match = anchor_mol.GetSubstructMatch(frag_query)
        if not anchor_match:
            continue
        mol_match = mol.GetSubstructMatch(frag_query)
        if not mol_match or len(mol_match) != len(anchor_match):
            continue

        for mol_idx, anchor_idx in zip(mol_match, anchor_match):
            pos = anchor_conf.GetAtomPosition(anchor_idx)
            coordMap[mol_idx] = Geometry.Point3D(pos.x, pos.y, pos.z)

    # ---- fallback: connection-point-only constraints ----
    if len(coordMap) < 2:
        print(
            f"[pocketfield] Full substructure matching produced only {len(coordMap)} "
            f"coordMap entries (expected ~{anchor_mol.GetNumHeavyAtoms()}). "
            f"Falling back to connection-point constraints."
        )
        sdf_neighbor_pos: dict[int, Geometry.Point3D] = {}
        for atom in anchor_mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                anchor_map = atom.GetIsotope() or atom.GetAtomMapNum()
                if anchor_map <= 0:
                    continue
                neighbors = atom.GetNeighbors()
                if len(neighbors) == 1:
                    pos = anchor_conf.GetAtomPosition(neighbors[0].GetIdx())
                    sdf_neighbor_pos[anchor_map] = Geometry.Point3D(pos.x, pos.y, pos.z)

        for atom in mol.GetAtoms():
            if atom.HasProp("pocketfield_anchor_map") and atom.GetAtomicNum() > 1:
                anchor_map = atom.GetIntProp("pocketfield_anchor_map")
                if fixed_map is not None and anchor_map != fixed_map:
                    continue
                if anchor_map in sdf_neighbor_pos:
                    coordMap[atom.GetIdx()] = sdf_neighbor_pos[anchor_map]

    # ---- linker attachment pins (bridge mode) ----
    # Pin the linker's attachment atoms to the anchor dummy coordinates so the
    # linker is geometrically closed between the fixed anchor points. In
    # flexible mode only the first (fixed) end is pinned; the far end is left
    # free so the linker can close the gap regardless of its exact length.
    dummy_pos: dict[int, Geometry.Point3D] = {}
    for atom in anchor_mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            anchor_map = atom.GetIsotope() or atom.GetAtomMapNum()
            if anchor_map > 0:
                pos = anchor_conf.GetAtomPosition(atom.GetIdx())
                dummy_pos[anchor_map] = Geometry.Point3D(pos.x, pos.y, pos.z)
    link_to_anchor = _linker_anchor_mapping(mol)
    for atom in mol.GetAtoms():
        if atom.HasProp("pocketfield_linker_map") and atom.GetAtomicNum() > 1:
            linker_map = atom.GetIntProp("pocketfield_linker_map")
            if fixed_map is not None and linker_map != 1:
                continue
            anchor_map = link_to_anchor.get(linker_map)
            if anchor_map is not None and anchor_map in dummy_pos:
                coordMap[atom.GetIdx()] = dummy_pos[anchor_map]

    # ---- centre coordinates at origin so RDKit distance geometry can use them ----
    if len(coordMap) >= 2:
        cx = sum(p.x for p in coordMap.values()) / len(coordMap)
        cy = sum(p.y for p in coordMap.values()) / len(coordMap)
        cz = sum(p.z for p in coordMap.values()) / len(coordMap)
        centred = {}
        for idx, p in coordMap.items():
            centred[idx] = Geometry.Point3D(p.x - cx, p.y - cy, p.z - cz)
        return centred, (cx, cy, cz)

    return coordMap, zero_translation


def _fit_to_anchor_pose(mol: Any, anchor_pose: dict[str, Any], conf_id: int = 0) -> None:
    target = np.asarray(anchor_pose["heavy_coords"], dtype=np.float64)
    if target.size == 0:
        return
    conf = mol.GetConformer(conf_id)
    anchor_indices = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.HasProp("pocketfield_anchor") and atom.GetAtomicNum() > 1
    ]
    if len(anchor_indices) != len(target):
        return
    current = np.asarray(
        [list(conf.GetAtomPosition(index)) for index in anchor_indices],
        dtype=np.float64,
    )
    all_positions = np.asarray(
        [list(conf.GetAtomPosition(index)) for index in range(mol.GetNumAtoms())],
        dtype=np.float64,
    )

    if len(anchor_indices) == 1:
        transformed = all_positions + (target[0] - current[0])
    else:
        rotation, translation = _kabsch_transform(current, target)
        transformed = all_positions @ rotation + translation
    for index, xyz in enumerate(transformed):
        conf.SetAtomPosition(index, tuple(float(value) for value in xyz))
    for atom_index, xyz in zip(anchor_indices, target, strict=True):
        conf.SetAtomPosition(atom_index, tuple(float(value) for value in xyz))


def _kabsch_transform(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_centroid = np.mean(source, axis=0)
    target_centroid = np.mean(target, axis=0)
    source_centered = source - source_centroid
    target_centered = target - target_centroid
    covariance = source_centered.T @ target_centered
    u, _, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1.0
    rotation = u @ correction @ vt
    translation = target_centroid - source_centroid @ rotation
    return rotation, translation


def _dummy_sector_assignments(
    Chem: Any,
    anchor_structure: str | Path | None,
    anchor_maps: list[int],
    center: np.ndarray,
    sector_coefficients: dict[int, dict[str, Any]],
    sectors_per_dummy: int,
    angle_cutoff: float,
) -> dict[int, list[dict[str, Any]]]:
    if anchor_structure is None:
        ranked = sorted(
            sector_coefficients.values(),
            key=lambda item: float(item.get("score", 0.0)),
        )
        return {
            anchor_map: [
                _sector_assignment(item, angle_deg=None, angular_score=0.0)
                for item in ranked[: max(1, sectors_per_dummy)]
            ]
            for anchor_map in anchor_maps
        }

    mol = _read_anchor_structure(Chem, anchor_structure)
    conf = mol.GetConformer()
    assignments: dict[int, list[dict[str, Any]]] = {}
    for dummy in [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]:
        anchor_map = dummy.GetIsotope() or dummy.GetAtomMapNum()
        if anchor_map not in anchor_maps:
            continue
        neighbors = dummy.GetNeighbors()
        if len(neighbors) != 1:
            raise ValueError("Anchor dummy atoms in --anchor-structure must have exactly one neighbor.")
        dummy_pos = np.asarray(list(conf.GetAtomPosition(dummy.GetIdx())), dtype=np.float64)
        neighbor_pos = np.asarray(list(conf.GetAtomPosition(neighbors[0].GetIdx())), dtype=np.float64)
        exit_vector = dummy_pos - neighbor_pos
        if np.linalg.norm(exit_vector) < 1e-8:
            exit_vector = dummy_pos - center
        exit_vector = exit_vector / max(np.linalg.norm(exit_vector), 1e-8)
        ranked = []
        for sector in sector_coefficients.values():
            direction = np.asarray(sector["direction"], dtype=np.float64)
            direction = direction / max(np.linalg.norm(direction), 1e-8)
            cosine = float(np.clip(np.dot(exit_vector, direction), -1.0, 1.0))
            angle = float(np.degrees(np.arccos(cosine)))
            if angle > angle_cutoff:
                continue
            # Favor sectors near this dummy vector, then sector field score.
            angular_score = angle / max(angle_cutoff, 1e-8)
            ranked.append(_sector_assignment(sector, angle_deg=angle, angular_score=angular_score))
        ranked.sort(key=lambda item: (item["angular_score"], item["sector_score"]))
        assignments[anchor_map] = ranked[: max(1, sectors_per_dummy)]

    missing = [anchor_map for anchor_map in anchor_maps if anchor_map not in assignments]
    if missing:
        raise ValueError(f"--anchor-structure is missing dummy isotope/map labels for: {missing}")
    return assignments


def _sector_assignment(
    sector: dict[str, Any],
    angle_deg: float | None,
    angular_score: float,
) -> dict[str, Any]:
    return {
        "sector_id": int(sector["sector_id"]),
        "sector_score": float(sector.get("score", 0.0)),
        "angle_deg": angle_deg,
        "angular_score": float(angular_score),
        "direction": sector["direction"],
    }


def _anchor_maps(Chem: Any, anchor_smiles: str) -> list[int]:
    mol = Chem.MolFromSmiles(anchor_smiles)
    if mol is None:
        raise ValueError(f"Invalid anchor SMILES: {anchor_smiles}")
    dummy_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
    unmapped = [atom.GetIdx() for atom in dummy_atoms if atom.GetAtomMapNum() <= 0]
    if unmapped:
        raise ValueError(
            "Anchor dummy atoms must use atom-map labels, for example [*:1]SMI1.[*:2]SMI2."
        )
    maps = sorted(atom.GetAtomMapNum() for atom in dummy_atoms)
    if len(maps) != len(set(maps)):
        raise ValueError("Anchor dummy atom-map labels must be unique.")
    return maps


def _linker_anchor_mapping(mol: Any) -> dict[int, int]:
    """Map linker attachment atom-map (1/2) to the anchor dummy map it bonds to.

    Reads the correspondence from the assembled molecule's topology (each linker
    attachment atom, tagged ``pocketfield_linker_map``, is bonded to the anchor
    connection atom tagged ``pocketfield_anchor_map``), so it holds for any
    anchor dummy map labels and for 3+-dummy anchors alike.
    """
    mapping: dict[int, int] = {}
    for atom in mol.GetAtoms():
        if not (atom.HasProp("pocketfield_linker_map") and atom.GetAtomicNum() > 1):
            continue
        for neighbor in atom.GetNeighbors():
            if neighbor.HasProp("pocketfield_anchor_map"):
                mapping[atom.GetIntProp("pocketfield_linker_map")] = neighbor.GetIntProp(
                    "pocketfield_anchor_map"
                )
                break
    return mapping


def _mean_direction(directions: list[list[float]]) -> np.ndarray:
    direction = np.sum(np.asarray(directions, dtype=np.float64), axis=0)
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0])
    return direction / norm


def _dummy_assignment_payload(assignments: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    return {
        "schema": "pocketfield.dummy_sector_assignments.v1",
        "assignments": [
            {"anchor_map": anchor_map, "sectors": sectors}
            for anchor_map, sectors in sorted(assignments.items())
        ],
    }


def _reference_occupied_sectors(
    Chem: Any,
    reference_path: str | Path,
    center: np.ndarray,
    sector_centers: np.ndarray,
    exclude_path: str | Path | None = None,
) -> tuple[list[int], dict[int, int], list[int]]:
    """Map a reference ligand's non-anchor ("arm") atoms to pocket sectors.

    Returns ``(occupied_ids, occupancy_counts, excluded_ids)``. When the
    reference contains the amide junction ``aryl-C(=O)-N(arm)``, the arm atoms
    are identified directly and projected (atom-level). Otherwise the full
    ligand's heavy atoms are projected and, if *exclude_path* is given, any
    sector the anchor touches is removed (sector-level fallback).
    """
    mol = _read_anchor_structure(Chem, reference_path)
    arm_counts = _arm_heavy_atom_sector_counts(Chem, mol, center, sector_centers)
    if arm_counts is not None:
        return sorted(arm_counts), arm_counts, []

    occupied = _project_heavy_atoms_to_sectors(mol, center, sector_centers)
    excluded: dict[int, int] = {}
    if exclude_path is not None:
        excluded = _project_heavy_atoms_to_sectors(
            _read_anchor_structure(Chem, exclude_path), center, sector_centers,
        )
    arm_occupied = {sid: count for sid, count in occupied.items() if sid not in excluded}
    return sorted(arm_occupied), arm_occupied, sorted(excluded)


def _arm_heavy_atom_sector_counts(
    Chem: Any,
    mol: Any,
    center: np.ndarray,
    sector_centers: np.ndarray,
) -> dict[int, int] | None:
    """Identify a reference ligand's arm heavy atoms via the amide junction.

    The arm attaches to the anchor's aryl ring as ``aryl-C(=O)-N(arm)`` (the
    same decomposition :mod:`scripts.prepare_arm_fragments` uses). The aryl side
    of that bond is the anchor; everything else (carbonyl C, =O, N, and its
    substituents) is the arm. Returns ``None`` when no such amide is found, so
    the caller can fall back to whole-ligand / sector-subtraction occupancy.
    """
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    pattern = Chem.MolFromSmarts("[c:1][C:2](=[O:3])[N:4]")
    if pattern is None:
        return None
    matches = mol.GetSubstructMatches(pattern)
    if not matches:
        return None
    aryl_c, carbonyl_c, _o, _n = matches[0]

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

    conf = mol.GetConformer()
    counts: dict[int, int] = {}
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        if atom.GetAtomicNum() <= 1 or idx in aryl_side:
            continue
        coord = np.asarray(list(conf.GetAtomPosition(idx)), dtype=np.float64)
        direction = coord - center
        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            continue
        direction = direction / norm
        sid = int(np.argmax(sector_centers @ direction))
        counts[sid] = counts.get(sid, 0) + 1
    return counts


def _project_heavy_atoms_to_sectors(
    mol: Any,
    center: np.ndarray,
    sector_centers: np.ndarray,
) -> dict[int, int]:
    conf = mol.GetConformer()
    counts: dict[int, int] = {}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        coord = np.asarray(list(conf.GetAtomPosition(atom.GetIdx())), dtype=np.float64)
        direction = coord - center
        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            continue
        direction = direction / norm
        sid = int(np.argmax(sector_centers @ direction))
        counts[sid] = counts.get(sid, 0) + 1
    return counts
