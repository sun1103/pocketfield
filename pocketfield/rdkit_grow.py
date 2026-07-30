from __future__ import annotations

import csv
import importlib.util
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pocketfield.io import write_json


VECTOR_KEYS = [
    "hydrophobic",
    "hbond_donor",
    "hbond_acceptor",
    "positive",
    "negative",
    "aromatic",
    "steric_openness",
    "small_size",
    "medium_size",
    "large_size",
    "rigidity",
    "polarity",
    "depth",
    "water_displacement",
    "buried_polar_support",
]

CHEMICAL_PROBE_KEYS = (
    "hydrophobic",
    "hbond_donor",
    "hbond_acceptor",
    "positive",
    "negative",
    "aromatic",
)
POLAR_KEYS = ("hbond_donor", "hbond_acceptor", "positive", "negative")
DESOLVATION_PENALTY_WEIGHT = 0.55


LIGAND_RADII = {
    6: 1.70,
    7: 1.55,
    8: 1.52,
    9: 1.47,
    15: 1.80,
    16: 1.80,
    17: 1.75,
    35: 1.85,
    53: 1.98,
}

PROBE_BY_ATOM = {
    6: ["hydrophobic", "aromatic"],
    7: ["hbond_acceptor", "hbond_donor", "positive"],
    8: ["hbond_acceptor", "negative"],
    9: ["hbond_acceptor", "hydrophobic"],
    15: ["hydrophobic"],
    16: ["hydrophobic", "hbond_acceptor"],
    17: ["hydrophobic"],
    35: ["hydrophobic"],
    53: ["hydrophobic"],
}


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


def grow_candidates(
    field_path: str | Path,
    plan_path: str | Path,
    metadata_path: str | Path,
    out_dir: str | Path,
    anchor_smiles: str,
    anchor_structure: str | Path | None,
    sectors_per_dummy: int,
    dummy_angle_cutoff: float,
    connect_anchor_dummies: bool,
    linker_distance_tolerance: float,
    linker_conformers: int,
    max_candidates: int,
    top_sectors: int,
    fragments_per_sector: int,
    growth_depth: int,
    request_limit: int,
    include_pairs: bool,
    fragment_library: str | Path | None,
    linker_library: str | Path | None,
    max_heavy_atoms: int,
    max_clash_score: float,
    max_field_score: float | None,
    sector_match_weight: float,
    retro_rules: str | Path | None,
    retro_script: str | Path | None,
    random_seed: int,
) -> dict[str, Any]:
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as exc:
        raise RuntimeError(
            "RDKit is required for 'pocketfield grow'. Run with a conda env that has RDKit, "
            "for example: /opt/anaconda3/envs/crem/bin/python -m pocketfield.cli grow ..."
        ) from exc

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    field = np.load(field_path, allow_pickle=False)
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    retro_validator = _load_retro_validator(retro_rules, retro_script)

    center = np.asarray(field["center"], dtype=np.float64)
    shells = np.asarray(field["shells"], dtype=np.float64)
    directions = np.asarray(field["directions"], dtype=np.float64)
    energies = np.asarray(field["energies"], dtype=np.float64)
    probe_names = [str(item) for item in field["probe_names"].tolist()]
    sector_ids = np.asarray(field["sector_ids"], dtype=np.int32)
    pocket_atoms = metadata.get("pocket_atoms", [])
    fragments = load_fragments(fragment_library)
    _validate_fragment_library(Chem, fragments)
    linkers = load_linkers(linker_library)
    _validate_linker_library(Chem, linkers)
    sector_coefficients = _sector_coefficients(
        plan=plan,
        energies=energies,
        shells=shells,
        sector_ids=sector_ids,
        probe_names=probe_names,
        pocket_atoms=pocket_atoms,
        center=center,
    )
    fragment_features = _fragment_features(Chem, fragments)
    linker_features = _linker_features(Chem, linkers)
    fragment_rankings = _rank_fragments_by_sector(sector_coefficients, fragment_features)
    linker_rankings = _rank_linkers_by_sector(sector_coefficients, linker_features)
    write_json(out / "sector_coefficients.json", _vector_payload("sector_coefficients", sector_coefficients))
    write_json(out / "fragment_features.json", _vector_payload("fragment_features", fragment_features))
    write_json(out / "linker_features.json", _vector_payload("linker_features", linker_features))
    anchor_maps = _anchor_maps(Chem, anchor_smiles)
    if not anchor_maps:
        raise ValueError("Anchor SMILES must contain at least one dummy atom such as [*:1].")
    anchor_pose = _anchor_pose(Chem, anchor_structure)
    dummy_sectors = _dummy_sector_assignments(
        Chem=Chem,
        anchor_structure=anchor_structure,
        anchor_maps=anchor_maps,
        center=center,
        sector_coefficients=sector_coefficients,
        sectors_per_dummy=sectors_per_dummy,
        angle_cutoff=dummy_angle_cutoff,
    )
    write_json(out / "dummy_sector_assignments.json", _dummy_assignment_payload(dummy_sectors))

    growth_depth = min(max(1, growth_depth), len(anchor_maps), top_sectors)
    requests = _growth_requests(
        plan=plan,
        fragment_rankings=fragment_rankings,
        linker_rankings=linker_rankings,
        dummy_sectors=dummy_sectors,
        top_sectors=top_sectors,
        fragments_per_sector=fragments_per_sector,
        growth_depth=growth_depth,
        request_limit=request_limit,
        include_pairs=include_pairs,
        connect_anchor_dummies=connect_anchor_dummies,
        anchor_pose=anchor_pose,
        linker_distance_tolerance=linker_distance_tolerance,
        linker_conformers=linker_conformers,
        Chem=Chem,
        anchor_maps=anchor_maps,
    )
    candidates = []
    attempted = 0
    rejections: dict[str, int] = {}

    for request_index, request in enumerate(requests):
        attempted += 1
        try:
            mol = _assemble_molecule(
                Chem,
                anchor_smiles,
                request["attachments"],
                connect_anchor_dummies=connect_anchor_dummies,
            )
            if retro_validator is not None and not retro_validator.validate(mol):
                _record_rejection(rejections, "retrosynthesis_filter")
                continue
            mol = _strip_isotopes(Chem, mol)
            smiles = Chem.MolToSmiles(mol)
            if mol.GetNumHeavyAtoms() > max_heavy_atoms:
                _record_rejection(rejections, "too_many_heavy_atoms")
                continue

            mol3d = Chem.AddHs(mol)
            status = AllChem.EmbedMolecule(mol3d, randomSeed=random_seed + request_index)
            if status != 0:
                _record_rejection(rejections, "embedding_failed")
                continue
            try:
                AllChem.UFFOptimizeMolecule(mol3d, maxIters=200)
            except Exception:
                pass
            if anchor_pose is not None:
                _fit_to_anchor_pose(mol3d, anchor_pose)
            else:
                _align_candidate_to_field(mol3d, center, request["direction"])
            score = _score_candidate(
                mol3d,
                center,
                shells,
                directions,
                energies,
                probe_names,
                pocket_atoms,
            )
            if score["clash_score"] > max_clash_score:
                _record_rejection(rejections, "clash_filter")
                continue
            if max_field_score is not None and score["field_score"] > max_field_score:
                _record_rejection(rejections, "field_filter")
                continue
            sector_match_score = float(
                np.mean([attachment["sector_fragment_score"] for attachment in request["attachments"]])
            )
            final_score = score["score"] + sector_match_weight * sector_match_score
            candidate = {
                "rank": 0,
                "smiles": smiles,
                "fragments": [
                    attachment["fragment"].name
                    for attachment in request["attachments"]
                    if "fragment" in attachment
                ],
                "linkers": [
                    attachment["linker"].name
                    for attachment in request["attachments"]
                    if "linker" in attachment
                ],
                "sector_ids": request["sector_ids"],
                "dummy_sector_angles": request["dummy_sector_angles"],
                "linker_geometry": request.get("linker_geometry"),
                "sector_fragment_scores": [
                    attachment["sector_fragment_score"] for attachment in request["attachments"]
                ],
                "sector_match_score": sector_match_score,
                "attachment_maps": [attachment["map"] for attachment in request["attachments"]],
                "mode": request["mode"],
                "field_score": score["field_score"],
                "clash_score": score["clash_score"],
                "raw_score": score["score"],
                "score": final_score,
                "heavy_atoms": mol.GetNumHeavyAtoms(),
            }
            mol3d.SetProp("_Name", f"pocketfield_{len(candidates) + 1}")
            for key, value in candidate.items():
                mol3d.SetProp(str(key), json.dumps(value) if isinstance(value, list) else str(value))
            candidates.append((candidate, mol3d))
        except Exception:
            _record_rejection(rejections, "assembly_or_scoring_failed")
            continue

    candidates.sort(key=lambda item: item[0]["score"])
    candidates = _dedupe_scored_candidates(candidates, rejections)
    candidates = candidates[:max_candidates]
    for index, (candidate, mol) in enumerate(candidates, start=1):
        candidate["rank"] = index
        mol.SetProp("_Name", f"pocketfield_{index}")
        mol.SetProp("rank", str(index))

    writer = Chem.SDWriter(str(out / "candidates.sdf"))
    for _, mol in candidates:
        writer.write(mol)
    writer.close()

    result = {
        "schema": "pocketfield.candidates.v1",
        "anchor_smiles": anchor_smiles,
        "anchor_structure": str(anchor_structure) if anchor_structure else None,
        "field": str(field_path),
        "plan": str(plan_path),
        "metadata": str(metadata_path),
        "candidate_count": len(candidates),
        "attempted_requests": attempted,
        "total_requests": len(requests),
        "rejections": rejections,
        "filters": {
            "max_heavy_atoms": max_heavy_atoms,
            "max_clash_score": max_clash_score,
            "max_field_score": max_field_score,
            "sector_match_weight": sector_match_weight,
            "sectors_per_dummy": sectors_per_dummy,
            "dummy_angle_cutoff": dummy_angle_cutoff,
            "connect_anchor_dummies": connect_anchor_dummies,
            "linker_distance_tolerance": linker_distance_tolerance,
            "linker_conformers": linker_conformers,
            "retro_rules": str(retro_rules) if retro_rules else None,
        },
        "fragment_library": str(fragment_library) if fragment_library else "built-in",
        "linker_library": str(linker_library) if linker_library else "built-in",
        "candidates": [candidate for candidate, _ in candidates],
        "outputs": {
            "sdf": "candidates.sdf",
            "json": "candidates.json",
            "sector_coefficients": "sector_coefficients.json",
            "fragment_features": "fragment_features.json",
            "linker_features": "linker_features.json",
            "dummy_sector_assignments": "dummy_sector_assignments.json",
        },
    }
    write_json(out / "candidates.json", result)
    return result


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


def _load_retro_validator(rules_path: str | Path | None, script_path: str | Path | None) -> Any | None:
    if rules_path is None:
        return None
    script = Path(script_path) if script_path else Path("/Users/mai/Software/fragenv/scripts/merge_utils.py")
    if not script.exists():
        raise ValueError(f"Retrosynthesis validator script does not exist: {script}")
    try:
        spec = importlib.util.spec_from_file_location("pocketfield_retro_merge_utils", script)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module spec from {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        validator_cls = getattr(module, "RetrosynthesisValidator")
        return validator_cls(str(rules_path))
    except ImportError as exc:
        raise RuntimeError(
            "Retrosynthesis validation requires an environment with rdchiral. "
            "Run with /opt/anaconda3/envs/rxn-env/bin/python or disable --retro-rules."
        ) from exc


def _sector_coefficients(
    plan: dict[str, Any],
    energies: np.ndarray,
    shells: np.ndarray,
    sector_ids: np.ndarray,
    probe_names: list[str],
    pocket_atoms: list[dict[str, Any]],
    center: np.ndarray,
) -> dict[int, dict[str, Any]]:
    coefficients: dict[int, dict[str, Any]] = {}
    probe_index = {name: index for index, name in enumerate(probe_names)}

    for sector in plan.get("single_sector_growing", []):
        sector_id = int(sector["sector_id"])
        mask = sector_ids == sector_id
        if not np.any(mask):
            continue
        vector = {key: 0.0 for key in VECTOR_KEYS}
        for probe in CHEMICAL_PROBE_KEYS:
            if probe not in probe_index:
                continue
            probe_values = energies[probe_index[probe], :, :][:, mask]
            well = _well_strength(float(np.percentile(probe_values, 10.0)))
            vector[probe] = well

        best_probe_values = np.min(energies[:, :, mask], axis=0)
        shell_min = np.min(best_probe_values, axis=1)
        best_shell_index = int(np.argmin(shell_min))
        normalized_depth = float(shells[best_shell_index] / max(float(shells[-1]), 1e-8))
        vector["depth"] = normalized_depth
        vector["small_size"] = max(0.0, 1.0 - normalized_depth)
        vector["medium_size"] = 1.0 - abs(normalized_depth - 0.55) / 0.55
        vector["large_size"] = normalized_depth
        vector["steric_openness"] = _sector_openness(
            direction=np.asarray(sector["direction"], dtype=np.float64),
            pocket_atoms=pocket_atoms,
            center=center,
        )
        vector["rigidity"] = max(vector["hydrophobic"], vector["aromatic"])
        vector["polarity"] = max(
            vector["hbond_donor"],
            vector["hbond_acceptor"],
            vector["positive"],
            vector["negative"],
        )
        _add_sector_desolvation_features(vector)
        coefficients[sector_id] = {
            "sector_id": sector_id,
            "vector": _ordered_vector(vector),
            "features": vector,
            "best_probe": sector.get("best_probe"),
            "score": sector.get("score"),
            "direction": sector.get("direction"),
        }
    return coefficients


def _fragment_features(Chem: Any, fragments: list[Fragment]) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {}
    for fragment in fragments:
        mol = Chem.MolFromSmiles(fragment.smiles)
        if mol is None:
            continue
        raw, heavy_atoms = _chemical_feature_vector(mol, fragment.probes)
        features[fragment.name] = {
            "name": fragment.name,
            "smiles": fragment.smiles,
            "probes": list(fragment.probes),
            "heavy_atoms": heavy_atoms,
            "vector": _ordered_vector(raw),
            "features": raw,
            "fragment": fragment,
        }
    return features


def _linker_features(Chem: Any, linkers: list[Linker]) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {}
    for linker in linkers:
        mol = Chem.MolFromSmiles(linker.smiles)
        if mol is None:
            continue
        raw, heavy_atoms = _chemical_feature_vector(mol, linker.probes)
        features[linker.name] = {
            "name": linker.name,
            "smiles": linker.smiles,
            "probes": list(linker.probes),
            "heavy_atoms": heavy_atoms,
            "vector": _ordered_vector(raw),
            "features": raw,
            "linker": linker,
        }
    return features


def _chemical_feature_vector(mol: Any, probes: tuple[str, ...]) -> tuple[dict[str, float], int]:
    real_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
    heavy_atoms = len(real_atoms)
    aromatic_atoms = sum(1 for atom in real_atoms if atom.GetIsAromatic())
    donor_count = sum(1 for atom in real_atoms if atom.GetAtomicNum() in {7, 8} and atom.GetTotalNumHs() > 0)
    acceptor_count = sum(1 for atom in real_atoms if atom.GetAtomicNum() in {7, 8, 16})
    hydrophobic_atoms = sum(
        1 for atom in real_atoms if atom.GetAtomicNum() in {6, 9, 16, 17, 35, 53}
    )
    positive = 1.0 if any(atom.GetFormalCharge() > 0 for atom in real_atoms) or "positive" in probes else 0.0
    negative = 1.0 if any(atom.GetFormalCharge() < 0 for atom in real_atoms) or "negative" in probes else 0.0
    bond_count = max(1, mol.GetNumBonds())
    rotatable_proxy = sum(1 for bond in mol.GetBonds() if bond.GetBondTypeAsDouble() == 1.0)

    raw = {
        "hydrophobic": _clip01(hydrophobic_atoms / max(heavy_atoms, 1)),
        "hbond_donor": _clip01(donor_count / 2.0),
        "hbond_acceptor": _clip01(acceptor_count / 3.0),
        "positive": positive,
        "negative": negative,
        "aromatic": _clip01(aromatic_atoms / max(heavy_atoms, 1)),
        "steric_openness": _clip01(heavy_atoms / 10.0),
        "small_size": _size_membership(heavy_atoms, center=1.5, width=2.0),
        "medium_size": _size_membership(heavy_atoms, center=4.5, width=3.5),
        "large_size": _clip01((heavy_atoms - 4.0) / 8.0),
        "rigidity": _clip01((aromatic_atoms + max(0, bond_count - rotatable_proxy)) / max(bond_count, 1)),
        "polarity": _clip01((donor_count + acceptor_count + positive + negative) / 4.0),
        "depth": _clip01(heavy_atoms / 10.0),
    }
    for probe in probes:
        if probe in raw:
            raw[probe] = max(raw[probe], 1.0)
    _add_candidate_desolvation_features(raw)
    return raw, heavy_atoms


def _add_sector_desolvation_features(features: dict[str, float]) -> None:
    burial = _clip01(1.0 - features["steric_openness"])
    polar_support = _polar_support(features)
    hydrophobic_support = _hydrophobic_support(features)
    unsupported_burial = burial * (1.0 - polar_support)

    features["burial"] = burial
    features["desolvation_risk"] = _clip01(unsupported_burial)
    features["water_displacement"] = _clip01(
        burial * max(hydrophobic_support, 0.5 * (1.0 - polar_support))
    )
    features["buried_polar_support"] = _clip01(burial * polar_support)


def _add_candidate_desolvation_features(features: dict[str, float]) -> None:
    polar = _polar_support(features)
    hydrophobic = _hydrophobic_support(features)

    features["water_displacement"] = _clip01(
        0.75 * hydrophobic + 0.25 * features["rigidity"] - 0.35 * polar
    )
    features["buried_polar_support"] = polar
    features["polar_desolvation_cost"] = _clip01(polar * (1.0 - 0.5 * hydrophobic))


def _score_vector_match(sector: dict[str, Any], feature: dict[str, Any]) -> dict[str, float]:
    sector_vector = np.asarray(sector["vector"], dtype=np.float64)
    candidate_vector = np.asarray(feature["vector"], dtype=np.float64)
    dot = float(np.dot(sector_vector, candidate_vector))
    mismatch = float(np.linalg.norm(sector_vector - candidate_vector))
    vector_match_score = -dot + 0.35 * mismatch
    desolvation_penalty = _desolvation_penalty(sector["features"], feature["features"])
    return {
        "score": float(vector_match_score + desolvation_penalty),
        "vector_match_score": float(vector_match_score),
        "desolvation_penalty": float(desolvation_penalty),
        "dot": dot,
        "mismatch": mismatch,
    }


def _desolvation_penalty(sector_features: dict[str, float], candidate_features: dict[str, float]) -> float:
    sector_risk = float(sector_features.get("desolvation_risk", 0.0))
    candidate_cost = float(candidate_features.get("polar_desolvation_cost", 0.0))
    polar_support_match = min(
        float(sector_features.get("buried_polar_support", 0.0)),
        float(candidate_features.get("buried_polar_support", 0.0)),
    )
    return DESOLVATION_PENALTY_WEIGHT * sector_risk * candidate_cost * (1.0 - polar_support_match)


def _polar_support(features: dict[str, float]) -> float:
    return _clip01(max(float(features.get(key, 0.0)) for key in POLAR_KEYS))


def _hydrophobic_support(features: dict[str, float]) -> float:
    return _clip01(
        max(float(features.get("hydrophobic", 0.0)), float(features.get("aromatic", 0.0)))
    )


def _rank_fragments_by_sector(
    sector_coefficients: dict[int, dict[str, Any]],
    fragment_features: dict[str, dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    rankings: dict[int, list[dict[str, Any]]] = {}
    for sector_id, sector in sector_coefficients.items():
        ranked = []
        for feature in fragment_features.values():
            match = _score_vector_match(sector, feature)
            ranked.append(
                {
                    "fragment": feature["fragment"],
                    "fragment_name": feature["name"],
                    **match,
                }
            )
        ranked.sort(key=lambda item: item["score"])
        rankings[sector_id] = ranked
    return rankings


def _rank_linkers_by_sector(
    sector_coefficients: dict[int, dict[str, Any]],
    linker_features: dict[str, dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    rankings: dict[int, list[dict[str, Any]]] = {}
    for sector_id, sector in sector_coefficients.items():
        ranked = []
        for feature in linker_features.values():
            match = _score_vector_match(sector, feature)
            ranked.append(
                {
                    "linker": feature["linker"],
                    "linker_name": feature["name"],
                    **match,
                }
            )
        ranked.sort(key=lambda item: item["score"])
        rankings[sector_id] = ranked
    return rankings


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


def _fit_to_anchor_pose(mol: Any, anchor_pose: dict[str, Any]) -> None:
    target = np.asarray(anchor_pose["heavy_coords"], dtype=np.float64)
    if target.size == 0:
        return
    conf = mol.GetConformer()
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


def _dummy_assignment_payload(assignments: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    return {
        "schema": "pocketfield.dummy_sector_assignments.v1",
        "assignments": [
            {"anchor_map": anchor_map, "sectors": sectors}
            for anchor_map, sectors in sorted(assignments.items())
        ],
    }


def _vector_payload(schema_name: str, items: dict[Any, dict[str, Any]]) -> dict[str, Any]:
    payload_items = []
    for item in items.values():
        clean = {key: value for key, value in item.items() if key not in {"fragment", "linker"}}
        payload_items.append(clean)
    return {"schema": f"pocketfield.{schema_name}.v1", "vector_keys": VECTOR_KEYS, "items": payload_items}


def _ordered_vector(features: dict[str, float]) -> list[float]:
    return [float(features.get(key, 0.0)) for key in VECTOR_KEYS]


def _well_strength(value: float) -> float:
    return _clip01(-value / 8.0)


def _sector_openness(
    direction: np.ndarray,
    pocket_atoms: list[dict[str, Any]],
    center: np.ndarray,
) -> float:
    if not pocket_atoms:
        return 1.0
    direction = direction / max(np.linalg.norm(direction), 1e-8)
    nearest = 99.0
    for atom in pocket_atoms:
        coord = np.asarray(atom["coord"], dtype=np.float64)
        vec = coord - center
        projection = float(np.dot(vec, direction))
        if projection <= 0.0:
            continue
        lateral = np.linalg.norm(vec - projection * direction)
        if lateral <= 2.0:
            nearest = min(nearest, projection - float(atom.get("radius", 1.7)))
    return _clip01((nearest - 1.0) / 5.0)


def _size_membership(value: float, center: float, width: float) -> float:
    return _clip01(1.0 - abs(value - center) / max(width, 1e-8))


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _growth_requests(
    plan: dict[str, Any],
    fragment_rankings: dict[int, list[dict[str, Any]]],
    linker_rankings: dict[int, list[dict[str, Any]]],
    dummy_sectors: dict[int, list[dict[str, Any]]],
    top_sectors: int,
    fragments_per_sector: int,
    growth_depth: int,
    request_limit: int,
    include_pairs: bool,
    connect_anchor_dummies: bool,
    anchor_pose: dict[str, Any] | None,
    linker_distance_tolerance: float,
    linker_conformers: int,
    Chem: Any,
    anchor_maps: list[int],
) -> list[dict[str, Any]]:
    del plan
    del top_sectors
    requests: list[dict[str, Any]] = []
    request_limit = max(1, request_limit)
    dummy_options = [
        (anchor_map, dummy_sectors.get(anchor_map, []))
        for anchor_map in anchor_maps
        if dummy_sectors.get(anchor_map)
    ]

    if connect_anchor_dummies:
        return _bridge_requests(
            linker_rankings=linker_rankings,
            dummy_options=dummy_options,
            fragments_per_sector=fragments_per_sector,
            request_limit=request_limit,
            anchor_pose=anchor_pose,
            linker_distance_tolerance=linker_distance_tolerance,
            linker_conformers=linker_conformers,
            Chem=Chem,
        )

    for anchor_map, sectors in dummy_options:
        for sector in sectors:
            for ranked_fragment in _ranked_fragments(
                fragment_rankings, sector["sector_id"], fragments_per_sector
            ):
                requests.append(
                    {
                        "mode": "single",
                        "sector_ids": [sector["sector_id"]],
                        "dummy_sector_angles": [sector["angle_deg"]],
                        "direction": np.asarray(sector["direction"], dtype=np.float64),
                        "attachments": [
                            {
                                "map": anchor_map,
                                "fragment": ranked_fragment["fragment"],
                                "sector_fragment_score": _dummy_adjusted_score(
                                    ranked_fragment["score"], sector
                                ),
                            }
                        ],
                    }
                )
                if len(requests) >= request_limit:
                    return requests

    if include_pairs and len(dummy_options) >= 2:
        for dummy_pair in itertools.combinations(dummy_options, 2):
            (map_a, sectors_a), (map_b, sectors_b) = dummy_pair
            for sector_a in sectors_a:
                for sector_b in sectors_b:
                    first = _ranked_fragments(
                        fragment_rankings, sector_a["sector_id"], max(1, fragments_per_sector // 2)
                    )
                    second = _ranked_fragments(
                        fragment_rankings, sector_b["sector_id"], max(1, fragments_per_sector // 2)
                    )
                    for frag_a in first:
                        for frag_b in second:
                            direction = _mean_direction([sector_a["direction"], sector_b["direction"]])
                            requests.append(
                                {
                                    "mode": "pair",
                                    "sector_ids": [sector_a["sector_id"], sector_b["sector_id"]],
                                    "dummy_sector_angles": [
                                        sector_a["angle_deg"],
                                        sector_b["angle_deg"],
                                    ],
                                    "direction": direction,
                                    "attachments": [
                                        {
                                            "map": map_a,
                                            "fragment": frag_a["fragment"],
                                            "sector_fragment_score": _dummy_adjusted_score(
                                                frag_a["score"], sector_a
                                            ),
                                        },
                                        {
                                            "map": map_b,
                                            "fragment": frag_b["fragment"],
                                            "sector_fragment_score": _dummy_adjusted_score(
                                                frag_b["score"], sector_b
                                            ),
                                        },
                                    ],
                                }
                            )
                            if len(requests) >= request_limit:
                                return requests

    for depth in range(3, min(growth_depth, len(dummy_options)) + 1):
        for dummy_combo in itertools.combinations(dummy_options, depth):
            sector_choices = [sectors for _, sectors in dummy_combo]
            for sector_combo in itertools.product(*sector_choices):
                fragment_choices = [
                    _ranked_fragments(fragment_rankings, sector["sector_id"], fragments_per_sector)
                    for sector in sector_combo
                ]
                for fragment_combo in itertools.product(*fragment_choices):
                    direction = _mean_direction([sector["direction"] for sector in sector_combo])
                    requests.append(
                        {
                            "mode": f"multi_{depth}",
                            "sector_ids": [sector["sector_id"] for sector in sector_combo],
                            "dummy_sector_angles": [sector["angle_deg"] for sector in sector_combo],
                            "direction": direction,
                            "attachments": [
                                {
                                    "map": dummy_combo[index][0],
                                    "fragment": ranked_fragment["fragment"],
                                    "sector_fragment_score": _dummy_adjusted_score(
                                        ranked_fragment["score"], sector_combo[index]
                                    ),
                                }
                                for index, ranked_fragment in enumerate(fragment_combo)
                            ],
                        }
                    )
                    if len(requests) >= request_limit:
                        return requests
    return requests


def _dummy_adjusted_score(fragment_score: float, sector: dict[str, Any]) -> float:
    return float(fragment_score + 0.25 * sector["angular_score"])


def _bridge_requests(
    linker_rankings: dict[int, list[dict[str, Any]]],
    dummy_options: list[tuple[int, list[dict[str, Any]]]],
    fragments_per_sector: int,
    request_limit: int,
    anchor_pose: dict[str, Any] | None,
    linker_distance_tolerance: float,
    linker_conformers: int,
    Chem: Any,
) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    for dummy_pair in itertools.combinations(dummy_options, 2):
        (map_a, sectors_a), (map_b, sectors_b) = dummy_pair
        for sector_a in sectors_a:
            for sector_b in sectors_b:
                ranked_linkers = _rank_bridge_linkers(
                    linker_rankings=linker_rankings,
                    sector_a=sector_a,
                    sector_b=sector_b,
                    limit=fragments_per_sector,
                    anchor_pose=anchor_pose,
                    map_a=map_a,
                    map_b=map_b,
                    linker_distance_tolerance=linker_distance_tolerance,
                    linker_conformers=linker_conformers,
                    Chem=Chem,
                )
                for linker in ranked_linkers:
                    direction = _mean_direction([sector_a["direction"], sector_b["direction"]])
                    score_a = _dummy_adjusted_score(linker["score"], sector_a)
                    score_b = _dummy_adjusted_score(linker["score"], sector_b)
                    requests.append(
                        {
                            "mode": "bridge",
                            "sector_ids": [sector_a["sector_id"], sector_b["sector_id"]],
                            "dummy_sector_angles": [sector_a["angle_deg"], sector_b["angle_deg"]],
                            "linker_geometry": {
                                "target_span": linker.get("target_span"),
                                "linker_span": linker.get("linker_span"),
                                "span_error": linker.get("span_error"),
                            },
                            "direction": direction,
                            "attachments": [
                                {
                                    "map": map_a,
                                    "linker": linker["linker"],
                                    "sector_fragment_score": score_a,
                                },
                                {
                                    "map": map_b,
                                    "linker": linker["linker"],
                                    "sector_fragment_score": score_b,
                                },
                            ],
                        }
                    )
                    if len(requests) >= request_limit:
                        return requests
    return requests


def _rank_bridge_linkers(
    linker_rankings: dict[int, list[dict[str, Any]]],
    sector_a: dict[str, Any],
    sector_b: dict[str, Any],
    limit: int,
    anchor_pose: dict[str, Any] | None,
    map_a: int,
    map_b: int,
    linker_distance_tolerance: float,
    linker_conformers: int,
    Chem: Any,
) -> list[dict[str, Any]]:
    by_name: dict[str, dict[str, Any]] = {}
    for ranked in linker_rankings.get(int(sector_a["sector_id"]), []):
        by_name[ranked["linker_name"]] = {
            "linker": ranked["linker"],
            "linker_name": ranked["linker_name"],
            "score_a": ranked["score"],
        }
    ranked_linkers = []
    for ranked in linker_rankings.get(int(sector_b["sector_id"]), []):
        existing = by_name.get(ranked["linker_name"])
        if existing is None:
            continue
        geometry = _linker_geometry_score(
            Chem=Chem,
            linker=ranked["linker"],
            anchor_pose=anchor_pose,
            map_a=map_a,
            map_b=map_b,
            tolerance=linker_distance_tolerance,
            conformers=linker_conformers,
        )
        if geometry is None:
            continue
        score = 0.5 * (existing["score_a"] + ranked["score"])
        ranked_linkers.append(
            {
                "linker": ranked["linker"],
                "linker_name": ranked["linker_name"],
                "score": float(score + geometry["penalty"]),
                "linker_span": geometry["span"],
                "target_span": geometry["target"],
                "span_error": geometry["error"],
            }
        )
    ranked_linkers.sort(key=lambda item: item["score"])
    return ranked_linkers[: max(1, limit)]


def _ranked_fragments(
    fragment_rankings: dict[int, list[dict[str, Any]]],
    sector_id: int,
    limit: int,
) -> list[dict[str, Any]]:
    ranked = fragment_rankings.get(int(sector_id), [])
    return ranked[: max(1, limit)]


def _linker_geometry_score(
    Chem: Any,
    linker: Linker,
    anchor_pose: dict[str, Any] | None,
    map_a: int,
    map_b: int,
    tolerance: float,
    conformers: int,
) -> dict[str, float] | None:
    if anchor_pose is None:
        return {"target": 0.0, "span": 0.0, "error": 0.0, "penalty": 0.0}
    dummy_coords = anchor_pose.get("dummy_coords", {})
    if map_a not in dummy_coords or map_b not in dummy_coords:
        return {"target": 0.0, "span": 0.0, "error": 0.0, "penalty": 0.0}
    target = float(np.linalg.norm(dummy_coords[map_a] - dummy_coords[map_b]))
    spans = _sample_linker_spans(Chem, linker, conformers)
    if not spans:
        return None
    best_span = min(spans, key=lambda span: abs(span - target))
    error = abs(best_span - target)
    if tolerance >= 0.0 and error > tolerance:
        return None
    return {
        "target": target,
        "span": float(best_span),
        "error": float(error),
        "penalty": float(error / max(tolerance, 1e-6)) if tolerance > 0.0 else 0.0,
    }


def _sample_linker_spans(Chem: Any, linker: Linker, conformers: int) -> list[float]:
    cache = getattr(_sample_linker_spans, "_cache", None)
    if cache is None:
        cache = {}
        setattr(_sample_linker_spans, "_cache", cache)
    cache_key = (linker.smiles, max(1, conformers))
    if cache_key in cache:
        return cache[cache_key]

    try:
        from rdkit import RDLogger
        from rdkit.Chem import AllChem
    except ImportError:
        return []

    RDLogger.DisableLog("rdApp.*")
    try:
        mol = Chem.AddHs(Chem.MolFromSmiles(linker.smiles))
        dummy_a = _find_dummy(mol, 1).GetIdx()
        dummy_b = _find_dummy(mol, 2).GetIdx()
        conf_ids = AllChem.EmbedMultipleConfs(
            mol,
            numConfs=max(1, conformers),
            randomSeed=17,
            pruneRmsThresh=0.25,
        )
        spans = []
        for conf_id in conf_ids:
            conf = mol.GetConformer(conf_id)
            span = conf.GetAtomPosition(dummy_a).Distance(conf.GetAtomPosition(dummy_b))
            spans.append(float(span))
        cache[cache_key] = spans
        return spans
    finally:
        RDLogger.EnableLog("rdApp.*")


def _mean_direction(directions: list[list[float]]) -> np.ndarray:
    direction = np.sum(np.asarray(directions, dtype=np.float64), axis=0)
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0])
    return direction / norm


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


def _record_rejection(rejections: dict[str, int], reason: str) -> None:
    rejections[reason] = rejections.get(reason, 0) + 1


def _dedupe_scored_candidates(
    candidates: list[tuple[dict[str, Any], Any]],
    rejections: dict[str, int],
) -> list[tuple[dict[str, Any], Any]]:
    seen: set[str] = set()
    deduped = []
    for candidate in candidates:
        smiles = candidate[0]["smiles"]
        if smiles in seen:
            _record_rejection(rejections, "duplicate_smiles_after_scoring")
            continue
        seen.add(smiles)
        deduped.append(candidate)
    return deduped


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


def _bridge_anchor_dummies(Chem: Any, base: Any, map_a: int, map_b: int, linker: Linker) -> Any:
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
    rw.GetAtomWithIdx(base_neighbor_b).SetIsotope(88)
    rw.GetAtomWithIdx(offset + linker_neighbor_a).SetIsotope(99)
    rw.GetAtomWithIdx(offset + linker_neighbor_b).SetIsotope(99)
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


def _find_dummy(mol: Any, map_number: int) -> Any:
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0 and atom.GetAtomMapNum() == map_number:
            if len(atom.GetNeighbors()) != 1:
                raise ValueError("Dummy atoms must have exactly one neighbor.")
            return atom
    raise ValueError(f"Missing dummy attachment atom [*:{map_number}]")


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


def _align_candidate_to_field(mol: Any, center: np.ndarray, direction: np.ndarray) -> None:
    conf = mol.GetConformer()
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


def _score_candidate(
    mol: Any,
    center: np.ndarray,
    shells: np.ndarray,
    directions: np.ndarray,
    energies: np.ndarray,
    probe_names: list[str],
    pocket_atoms: list[dict[str, Any]],
) -> dict[str, float]:
    conf = mol.GetConformer()
    probe_index = {name: index for index, name in enumerate(probe_names)}
    field_terms = []
    clash_score = 0.0

    pocket_coords = np.asarray([atom["coord"] for atom in pocket_atoms], dtype=np.float64)
    pocket_radii = np.asarray([atom.get("radius", 1.7) for atom in pocket_atoms], dtype=np.float64)

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        idx = atom.GetIdx()
        pos = np.asarray(list(conf.GetAtomPosition(idx)), dtype=np.float64)
        vec = pos - center
        radius = np.linalg.norm(vec)
        if radius < 1e-8:
            continue
        direction = vec / radius
        shell_idx = int(np.argmin(np.abs(shells - radius)))
        sample_idx = int(np.argmax(np.einsum("ij,j->i", directions, direction, optimize=False)))
        compatible = [
            probe_index[name]
            for name in PROBE_BY_ATOM.get(atom.GetAtomicNum(), ["hydrophobic"])
            if name in probe_index
        ]
        if compatible:
            field_terms.append(float(np.min(energies[compatible, shell_idx, sample_idx])))

        if len(pocket_coords):
            ligand_radius = LIGAND_RADII.get(atom.GetAtomicNum(), 1.70)
            distances = np.linalg.norm(pocket_coords - pos[None, :], axis=1)
            overlap = np.maximum(0.0, pocket_radii + ligand_radius - 0.45 - distances)
            clash_score += float(np.sum(overlap**2))

    field_score = float(np.mean(field_terms)) if field_terms else 0.0
    score = field_score + 25.0 * clash_score + 0.02 * mol.GetNumHeavyAtoms()
    return {"field_score": field_score, "clash_score": clash_score, "score": score}
