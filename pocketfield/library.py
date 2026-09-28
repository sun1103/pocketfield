"""Fragment and linker library definitions, loading, and validation.

Holds the built-in :class:`Fragment`/:class:`Linker` libraries and the
CSV/JSON/JSONL loaders used by ``pocketfield grow``. Validation parses SMILES,
so this module requires RDKit.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pocketfield.assembly import _find_dummy


@dataclass(frozen=True)
class Fragment:
    name: str
    smiles: str
    probes: tuple[str, ...]


@dataclass(frozen=True)
class Linker:
    name: str
    smiles: str
    probes: tuple[str, ...]


FRAGMENTS = [
    Fragment("methyl", "[*:1]C", ("hydrophobic",)),
    Fragment("ethyl", "[*:1]CC", ("hydrophobic",)),
    Fragment("isopropyl", "[*:1]C(C)C", ("hydrophobic",)),
    Fragment("phenyl", "[*:1]c1ccccc1", ("hydrophobic", "aromatic")),
    Fragment("hydroxyl", "[*:1]O", ("hbond_donor", "hbond_acceptor")),
    Fragment("methoxy", "[*:1]OC", ("hbond_acceptor",)),
    Fragment("amine", "[*:1]N", ("hbond_donor", "positive")),
    Fragment("dimethylamine", "[*:1]N(C)C", ("positive", "hbond_acceptor")),
    Fragment("amide", "[*:1]C(=O)N", ("hbond_donor", "hbond_acceptor")),
    Fragment("carbonyl", "[*:1]C=O", ("hbond_acceptor",)),
    Fragment("carboxyl", "[*:1]C(=O)O", ("negative", "hbond_acceptor", "hbond_donor")),
    Fragment("fluoro", "[*:1]F", ("hydrophobic", "hbond_acceptor")),
    Fragment("chloro", "[*:1]Cl", ("hydrophobic",)),
]

LINKERS = [
    Linker("methylene", "[*:1]C[*:2]", ("hydrophobic",)),
    Linker("ethylene", "[*:1]CC[*:2]", ("hydrophobic",)),
    Linker("propylene", "[*:1]CCC[*:2]", ("hydrophobic",)),
    Linker("ether", "[*:1]CO[*:2]", ("hbond_acceptor",)),
    Linker("amino_methylene", "[*:1]CN[*:2]", ("hbond_donor", "positive")),
    Linker("amide", "[*:1]C(=O)N[*:2]", ("hbond_donor", "hbond_acceptor")),
    Linker("reverse_amide", "[*:1]NC(=O)[*:2]", ("hbond_donor", "hbond_acceptor")),
    Linker("urea", "[*:1]NC(=O)N[*:2]", ("hbond_donor", "hbond_acceptor")),
    Linker("phenylene", "[*:1]c1ccc([*:2])cc1", ("hydrophobic", "aromatic")),
]


def load_fragments(path: str | Path | None = None) -> list[Fragment]:
    if path is None:
        return FRAGMENTS
    return [
        Fragment(name=name, smiles=smiles, probes=probes)
        for name, smiles, probes in _load_rows(path, "Fragment")
    ]


def load_linkers(path: str | Path | None = None) -> list[Linker]:
    if path is None:
        return LINKERS
    return [
        Linker(name=name, smiles=smiles, probes=probes)
        for name, smiles, probes in _load_rows(path, "Linker")
    ]


def _load_rows(path: str | Path, label: str) -> list[tuple[str, str, tuple[str, ...]]]:
    source = Path(path)
    if not source.exists():
        raise ValueError(f"{label} library does not exist: {source}")

    loaded: list[tuple[str, str, tuple[str, ...]]] = []
    for index, row in enumerate(_iter_library_rows(source, label), start=1):
        if "smiles" not in row:
            raise ValueError(f"{label} library must contain a 'smiles' column/key.")
        smiles = str(row["smiles"]).strip()
        if not smiles:
            continue
        name = str(row.get("name") or row.get("catalog_id") or row.get("id") or f"{label.lower()}_{index}")
        probes = row.get("probes", [])
        if isinstance(probes, str):
            probes = [item.strip() for item in probes.replace(";", ",").split(",") if item.strip()]
        loaded.append((name, smiles, tuple(str(probe) for probe in probes)))
    if not loaded:
        raise ValueError(f"{label} library is empty.")
    return loaded


def _iter_library_rows(source: Path, label: str):
    suffix = source.suffix.lower()
    if suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        key = "fragments" if label == "Fragment" else "linkers"
        rows = payload.get(key, payload.get("items", [])) if isinstance(payload, dict) else payload
        yield from rows
    elif suffix == ".jsonl":
        with source.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)
    elif suffix == ".csv":
        with source.open("r", encoding="utf-8", newline="") as handle:
            yield from csv.DictReader(handle)
    else:
        raise ValueError(f"{label} library must be .json, .jsonl, or .csv")


def _validate_fragment_library(Chem: Any, fragments: list[Fragment]) -> None:
    for fragment in fragments:
        mol = Chem.MolFromSmiles(fragment.smiles)
        if mol is None:
            raise ValueError(f"Invalid fragment SMILES for {fragment.name}: {fragment.smiles}")
        _find_dummy(mol, 1)


def _validate_linker_library(Chem: Any, linkers: list[Linker]) -> None:
    for linker in linkers:
        mol = Chem.MolFromSmiles(linker.smiles)
        if mol is None:
            raise ValueError(f"Invalid linker SMILES for {linker.name}: {linker.smiles}")
        maps = sorted(
            atom.GetAtomMapNum()
            for atom in mol.GetAtoms()
            if atom.GetAtomicNum() == 0 and atom.GetAtomMapNum() > 0
        )
        if maps != [1, 2]:
            raise ValueError(
                f"Linker {linker.name} must contain exactly two mapped dummies [*:1] and [*:2]."
            )
        _find_dummy(mol, 1)
        _find_dummy(mol, 2)
