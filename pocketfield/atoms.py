from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


VDW_RADII = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "P": 1.80,
    "S": 1.80,
    "CL": 1.75,
    "BR": 1.85,
    "I": 1.98,
    "FE": 1.80,
    "ZN": 1.39,
    "MG": 1.73,
    "CA": 2.31,
    "NA": 2.27,
    "K": 2.75,
}

ATOM_EPSILON = {
    "H": 0.02,
    "C": 0.08,
    "N": 0.12,
    "O": 0.12,
    "F": 0.10,
    "P": 0.16,
    "S": 0.20,
    "CL": 0.18,
    "BR": 0.22,
    "I": 0.26,
}

AROMATIC_RESIDUES = {"PHE", "TYR", "TRP", "HIS"}
POSITIVE_RESIDUES = {"LYS", "ARG", "HIP"}
NEGATIVE_RESIDUES = {"ASP", "GLU"}


@dataclass(frozen=True)
class Atom:
    serial: int
    name: str
    residue: str
    chain: str
    residue_id: int
    element: str
    coord: np.ndarray
    radius: float
    epsilon: float
    charge: float
    donor: bool
    acceptor: bool
    hydrophobic: bool
    aromatic: bool

    def to_json(self) -> dict:
        return {
            "serial": self.serial,
            "name": self.name,
            "residue": self.residue,
            "chain": self.chain,
            "residue_id": self.residue_id,
            "element": self.element,
            "coord": self.coord.tolist(),
            "radius": self.radius,
            "charge": self.charge,
            "donor": self.donor,
            "acceptor": self.acceptor,
            "hydrophobic": self.hydrophobic,
            "aromatic": self.aromatic,
        }


def read_pdb_atoms(path: str | Path, include_hetatm: bool = True) -> list[Atom]:
    atoms: list[Atom] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            record = line[:6].strip()
            if record != "ATOM" and not (include_hetatm and record == "HETATM"):
                continue
            if len(line) < 54:
                continue

            try:
                serial = int(line[6:11])
                name = line[12:16].strip()
                residue = line[17:20].strip().upper()
                chain = line[21].strip()
                residue_id = int(line[22:26])
                coord = np.array(
                    [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                    dtype=np.float64,
                )
            except ValueError:
                continue

            element = _pdb_element(line, name)
            if element == "H":
                continue
            atoms.append(_type_atom(serial, name, residue, chain, residue_id, element, coord))
    return atoms


def read_pdb_coordinates(path: str | Path) -> np.ndarray:
    coords = [atom.coord for atom in read_pdb_atoms(path, include_hetatm=True)]
    if not coords:
        raise ValueError(f"No atoms found in {path}")
    return np.vstack(coords)


def select_pocket_atoms(atoms: list[Atom], center: np.ndarray, radius: float) -> list[Atom]:
    selected = [atom for atom in atoms if np.linalg.norm(atom.coord - center) <= radius]
    if not selected:
        raise ValueError("No pocket atoms found. Increase --pocket-radius or check the center.")
    return selected


def _pdb_element(line: str, name: str) -> str:
    raw = line[76:78].strip().upper() if len(line) >= 78 else ""
    if raw:
        return raw
    stripped = "".join(ch for ch in name.upper() if ch.isalpha())
    if not stripped:
        return "C"
    if len(stripped) >= 2 and stripped[:2] in VDW_RADII:
        return stripped[:2]
    return stripped[0]


def _type_atom(
    serial: int,
    name: str,
    residue: str,
    chain: str,
    residue_id: int,
    element: str,
    coord: np.ndarray,
) -> Atom:
    radius = VDW_RADII.get(element, 1.70)
    epsilon = ATOM_EPSILON.get(element, 0.10)
    donor = element in {"N", "O", "S"} and not name.startswith("OXT")
    acceptor = element in {"O", "N", "S"}
    aromatic = residue in AROMATIC_RESIDUES and element == "C"
    hydrophobic = element in {"C", "S", "CL", "BR", "I"} or aromatic

    charge = 0.0
    atom_name = name.upper()
    if residue in POSITIVE_RESIDUES and element == "N":
        charge = 0.35
    elif residue in NEGATIVE_RESIDUES and element == "O":
        charge = -0.35
    elif element == "N":
        charge = 0.15
    elif element == "O":
        charge = -0.15
    elif element == "S":
        charge = -0.05
    elif element in {"ZN", "MG", "CA", "NA", "K", "FE"}:
        charge = 0.5
    elif atom_name.startswith("C") and residue in NEGATIVE_RESIDUES:
        charge = 0.15

    return Atom(
        serial=serial,
        name=name,
        residue=residue,
        chain=chain,
        residue_id=residue_id,
        element=element,
        coord=coord,
        radius=radius,
        epsilon=epsilon,
        charge=charge,
        donor=donor,
        acceptor=acceptor,
        hydrophobic=hydrophobic,
        aromatic=aromatic,
    )

