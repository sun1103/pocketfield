"""Feature-vector construction and matching for sectors, fragments, and linkers.

Defines the shared 15-dim vector layout and the functions that turn a sector
(what the pocket wants) and a fragment/linker (what the molecule is) into
comparable vectors, then score their match.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


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


def _sector_coefficients(
    plan: dict[str, Any],
    energies: np.ndarray,
    shells: np.ndarray,
    sector_ids: np.ndarray,
    probe_names: list[str],
    pocket_atoms: list[dict[str, Any]],
    center: np.ndarray,
) -> dict[int, dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for sector in plan.get("single_sector_growing", []):
        sector_id = int(sector["sector_id"])
        mask = sector_ids == sector_id
        if not np.any(mask):
            continue
        records.append(
            {
                "sector_id": sector_id,
                "mask": mask,
                "direction": np.asarray(sector["direction"], dtype=np.float64),
                "best_probe": sector.get("best_probe"),
                "score": sector.get("score"),
            }
        )
    return _build_sector_coefficients(
        records, energies, shells, probe_names, pocket_atoms, center,
    )


def _build_sector_coefficients(
    records: list[dict[str, Any]],
    energies: np.ndarray,
    shells: np.ndarray,
    probe_names: list[str],
    pocket_atoms: list[dict[str, Any]],
    center: np.ndarray,
) -> dict[int, dict[str, Any]]:
    """Build per-sector 15-dim coefficient vectors from a set of sector records.

    Probe well strengths are normalised relative to the range observed across
    the sectors in ``records`` (min/max of the 10th-percentile probe energy),
    mirroring the library-relative normalisation used for fragment features.
    This removes the fixed 8 kcal/mol divisor that saturated every deep well to
    1.0 and collapsed distinct sectors into near-identical vectors.
    """
    if not records:
        return {}
    probe_index = {name: index for index, name in enumerate(probe_names)}

    raw_wells: dict[int, dict[str, float]] = {}
    for record in records:
        wells: dict[str, float] = {}
        for probe in CHEMICAL_PROBE_KEYS:
            if probe not in probe_index:
                continue
            probe_values = energies[probe_index[probe], :, :][:, record["mask"]]
            wells[probe] = float(np.percentile(probe_values, 10.0))
        raw_wells[int(record["sector_id"])] = wells

    bounds: dict[str, tuple[float, float]] = {}
    for probe in CHEMICAL_PROBE_KEYS:
        values = [raw_wells[sid][probe] for sid in raw_wells if probe in raw_wells[sid]]
        if values:
            bounds[probe] = (float(min(values)), float(max(values)))

    coefficients: dict[int, dict[str, Any]] = {}
    for record in records:
        sector_id = int(record["sector_id"])
        vector = {key: 0.0 for key in VECTOR_KEYS}
        for probe in CHEMICAL_PROBE_KEYS:
            if probe in raw_wells[sector_id]:
                lo, hi = bounds[probe]
                vector[probe] = _well_strength(raw_wells[sector_id][probe], lo, hi)

        best_probe_values = np.min(energies[:, :, record["mask"]], axis=0)
        shell_min = np.min(best_probe_values, axis=1)
        best_shell_index = int(np.argmin(shell_min))
        normalized_depth = float(shells[best_shell_index] / max(float(shells[-1]), 1e-8))
        vector["depth"] = normalized_depth
        vector["small_size"] = max(0.0, 1.0 - normalized_depth)
        vector["medium_size"] = 1.0 - abs(normalized_depth - 0.55) / 0.55
        vector["large_size"] = normalized_depth
        vector["steric_openness"] = _sector_openness(
            direction=record["direction"],
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
            "best_probe": record.get("best_probe"),
            "score": record.get("score"),
            "direction": record["direction"].tolist(),
        }
    return coefficients


def _reference_sector_coefficients(
    sector_ids: list[int],
    sector_centers: np.ndarray,
    energies: np.ndarray,
    shells: np.ndarray,
    field_sector_ids: np.ndarray,
    probe_names: list[str],
    pocket_atoms: list[dict[str, Any]],
    center: np.ndarray,
) -> dict[int, dict[str, Any]]:
    """Build coefficient vectors for arbitrary (reference-occupied) sectors."""
    records: list[dict[str, Any]] = []
    for sid in sorted(set(int(s) for s in sector_ids)):
        mask = field_sector_ids == sid
        if not np.any(mask):
            continue
        best_probe, score = _sector_best_probe(energies[:, :, mask], probe_names)
        records.append(
            {
                "sector_id": sid,
                "mask": mask,
                "direction": np.asarray(sector_centers[sid], dtype=np.float64),
                "best_probe": best_probe,
                "score": score,
            }
        )
    return _build_sector_coefficients(
        records, energies, shells, probe_names, pocket_atoms, center,
    )


def _sector_best_probe(
    sector_energies: np.ndarray,
    probe_names: list[str],
) -> tuple[str, float]:
    flat = int(np.argmin(sector_energies))
    probe_index, shell_index, sample_index = np.unravel_index(flat, sector_energies.shape)
    return probe_names[probe_index], float(sector_energies[probe_index, shell_index, sample_index])


def _fragment_features(Chem: Any, fragments: list) -> dict[str, dict[str, Any]]:
    mols = [Chem.MolFromSmiles(fragment.smiles) for fragment in fragments]
    norm_params = _norm_params([(mol, fragment.probes) for mol, fragment in zip(mols, fragments)])
    features: dict[str, dict[str, Any]] = {}
    for fragment, mol in zip(fragments, mols):
        if mol is None:
            continue
        raw, heavy_atoms = _chemical_feature_vector(mol, fragment.probes, norm_params)
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


def _linker_features(Chem: Any, linkers: list) -> dict[str, dict[str, Any]]:
    mols = [Chem.MolFromSmiles(linker.smiles) for linker in linkers]
    norm_params = _norm_params([(mol, linker.probes) for mol, linker in zip(mols, linkers)])
    features: dict[str, dict[str, Any]] = {}
    for linker, mol in zip(linkers, mols):
        if mol is None:
            continue
        raw, heavy_atoms = _chemical_feature_vector(mol, linker.probes, norm_params)
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


def _raw_counts(mol: Any) -> dict[str, Any]:
    real_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
    donor_count = sum(1 for atom in real_atoms if atom.GetAtomicNum() in {7, 8} and atom.GetTotalNumHs() > 0)
    acceptor_count = sum(1 for atom in real_atoms if atom.GetAtomicNum() in {7, 8, 16})
    return {
        "heavy_atoms": len(real_atoms),
        "aromatic_atoms": sum(1 for atom in real_atoms if atom.GetIsAromatic()),
        "donor_count": donor_count,
        "acceptor_count": acceptor_count,
        "hydrophobic_atoms": sum(1 for atom in real_atoms if atom.GetAtomicNum() in {6, 9, 16, 17, 35, 53}),
        "positive": 1.0 if any(atom.GetFormalCharge() > 0 for atom in real_atoms) else 0.0,
        "negative": 1.0 if any(atom.GetFormalCharge() < 0 for atom in real_atoms) else 0.0,
        "bond_count": max(1, mol.GetNumBonds()),
        "rotatable_proxy": sum(1 for bond in mol.GetBonds() if bond.GetBondTypeAsDouble() == 1.0),
    }


def _norm_params(items: list[tuple[Any, tuple[str, ...]]]) -> dict[str, float]:
    """Per-library min/max over heavy-atom and polar counts, scaling features to [0,1].

    Size and polar features are normalised relative to the library being grown,
    not to the hard-coded 1-10 heavy-atom range of the built-in fragments.
    Without this, a library spanning 2-19 heavy atoms saturates ``steric_openness``,
    ``large_size``, ``depth``, ``hbond_acceptor``, and ``polarity`` at 1.0.
    """
    stats: list[dict[str, Any]] = []
    for mol, probes in items:
        if mol is None:
            continue
        counts = _raw_counts(mol)
        if counts["heavy_atoms"] <= 0:
            continue
        positive = 1.0 if counts["positive"] > 0 or "positive" in probes else 0.0
        negative = 1.0 if counts["negative"] > 0 or "negative" in probes else 0.0
        counts["polar_sum"] = counts["donor_count"] + counts["acceptor_count"] + positive + negative
        stats.append(counts)
    if not stats:
        return {
            "size_min": 0.0, "size_max": 10.0,
            "donor_min": 0.0, "donor_max": 2.0,
            "acceptor_min": 0.0, "acceptor_max": 3.0,
            "polar_min": 0.0, "polar_max": 4.0,
        }
    return {
        "size_min": float(min(s["heavy_atoms"] for s in stats)),
        "size_max": float(max(s["heavy_atoms"] for s in stats)),
        "donor_min": float(min(s["donor_count"] for s in stats)),
        "donor_max": float(max(s["donor_count"] for s in stats)),
        "acceptor_min": float(min(s["acceptor_count"] for s in stats)),
        "acceptor_max": float(max(s["acceptor_count"] for s in stats)),
        "polar_min": float(min(s["polar_sum"] for s in stats)),
        "polar_max": float(max(s["polar_sum"] for s in stats)),
    }


def _scale_coord(value: float, params: dict[str, float], prefix: str) -> float:
    lo = float(params.get(f"{prefix}_min", 0.0))
    hi = float(params.get(f"{prefix}_max", 1.0))
    return _clip01((value - lo) / max(hi - lo, 1.0))


def _chemical_feature_vector(
    mol: Any,
    probes: tuple[str, ...],
    norm_params: dict[str, float] | None = None,
) -> tuple[dict[str, float], int]:
    counts = _raw_counts(mol)
    heavy_atoms = counts["heavy_atoms"]
    aromatic_atoms = counts["aromatic_atoms"]
    donor_count = counts["donor_count"]
    acceptor_count = counts["acceptor_count"]
    hydrophobic_atoms = counts["hydrophobic_atoms"]
    positive = 1.0 if counts["positive"] > 0 or "positive" in probes else 0.0
    negative = 1.0 if counts["negative"] > 0 or "negative" in probes else 0.0
    bond_count = counts["bond_count"]
    rotatable_proxy = counts["rotatable_proxy"]

    if norm_params is None:
        donor = _clip01(donor_count / 2.0)
        acceptor = _clip01(acceptor_count / 3.0)
        polarity = _clip01((donor_count + acceptor_count + positive + negative) / 4.0)
        size_coord = _clip01(heavy_atoms / 10.0)
        steric_openness = size_coord
        small_size = _size_membership(heavy_atoms, center=1.5, width=2.0)
        medium_size = _size_membership(heavy_atoms, center=4.5, width=3.5)
        large_size = _clip01((heavy_atoms - 4.0) / 8.0)
        depth = size_coord
    else:
        donor = _scale_coord(donor_count, norm_params, "donor")
        acceptor = _scale_coord(acceptor_count, norm_params, "acceptor")
        polarity = _scale_coord(donor_count + acceptor_count + positive + negative, norm_params, "polar")
        size_coord = _scale_coord(heavy_atoms, norm_params, "size")
        steric_openness = size_coord
        small_size = max(0.0, 1.0 - size_coord)
        medium_size = _clip01(1.0 - abs(size_coord - 0.55) / 0.55)
        large_size = size_coord
        depth = size_coord

    raw = {
        "hydrophobic": _clip01(hydrophobic_atoms / max(heavy_atoms, 1)),
        "hbond_donor": donor,
        "hbond_acceptor": acceptor,
        "positive": positive,
        "negative": negative,
        "aromatic": _clip01(aromatic_atoms / max(heavy_atoms, 1)),
        "steric_openness": steric_openness,
        "small_size": small_size,
        "medium_size": medium_size,
        "large_size": large_size,
        "rigidity": _clip01((aromatic_atoms + max(0, bond_count - rotatable_proxy)) / max(bond_count, 1)),
        "polarity": polarity,
        "depth": depth,
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
) -> tuple[dict[int, list[dict[str, Any]]], dict[int, dict[str, float]]]:
    rankings: dict[int, list[dict[str, Any]]] = {}
    stats: dict[int, dict[str, float]] = {}
    for sector_id, sector in sector_coefficients.items():
        ranked = []
        scores = []
        for feature in fragment_features.values():
            match = _score_vector_match(sector, feature)
            ranked.append(
                {
                    "fragment": feature["fragment"],
                    "fragment_name": feature["name"],
                    **match,
                }
            )
            scores.append(match["score"])
        ranked.sort(key=lambda item: item["score"])
        rankings[sector_id] = ranked
        scores_arr = np.asarray(scores, dtype=np.float64)
        stats[sector_id] = {
            "mean": float(np.mean(scores_arr)),
            "std": float(np.std(scores_arr)),
        }
    return rankings, stats


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


def _ordered_vector(features: dict[str, float]) -> list[float]:
    return [float(features.get(key, 0.0)) for key in VECTOR_KEYS]


def _well_strength(value: float, lo: float, hi: float) -> float:
    """Map a probe well energy to [0,1] relative to the sector set's range.

    ``lo``/``hi`` are the min/max 10th-percentile energy across the sectors
    being compared (more negative energy is a stronger well). Returns 1.0 for
    the deepest well and 0.0 for the shallowest.
    """
    span = hi - lo
    if span <= 1e-8:
        return 0.0
    return _clip01((hi - value) / span)


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


def _vector_payload(schema_name: str, items: dict[Any, dict[str, Any]]) -> dict[str, Any]:
    payload_items = []
    for item in items.values():
        clean = {key: value for key, value in item.items() if key not in {"fragment", "linker"}}
        payload_items.append(clean)
    return {"schema": f"pocketfield.{schema_name}.v1", "vector_keys": VECTOR_KEYS, "items": payload_items}


def _load_vina_lookup(fragment_library: str | None) -> dict[str, str]:
    """Map fragment names to ``vina_kcal_mol`` from a CSV library, if present."""
    if not fragment_library:
        return {}
    source = Path(fragment_library)
    if not source.exists() or source.suffix.lower() != ".csv":
        return {}
    import csv

    lookup: dict[str, str] = {}
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "vina_kcal_mol" not in reader.fieldnames:
            return {}
        for row in reader:
            name = str(row.get("name") or row.get("catalog_id") or row.get("id") or "").strip()
            if not name:
                continue
            lookup[name] = str(row["vina_kcal_mol"]).strip()
    return lookup


def _parse_vina(raw: str | None) -> float | None:
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _reference_screening_report(
    reference_ligand: str,
    anchor_structure: str | None,
    occupied_sectors: list[int],
    occupancy: dict[int, int],
    excluded_sectors: list[int],
    reference_coefficients: dict[int, dict[str, Any]],
    reference_rankings: dict[int, list[dict[str, Any]]],
    fragment_features: dict[str, dict[str, Any]],
    vina_lookup: dict[str, str],
    top: int,
) -> dict[str, Any]:
    del fragment_features
    top = max(1, top)
    per_sector: list[dict[str, Any]] = []
    for sid in occupied_sectors:
        if sid not in reference_rankings:
            continue
        matches = []
        for item in reference_rankings[sid][:top]:
            name = item["fragment_name"]
            matches.append(
                {
                    "fragment_name": name,
                    "smiles": item["fragment"].smiles,
                    "vina_kcal_mol": _parse_vina(vina_lookup.get(name)),
                    "score": float(item["score"]),
                    "vector_match_score": float(item["vector_match_score"]),
                    "desolvation_penalty": float(item["desolvation_penalty"]),
                    "dot": float(item["dot"]),
                    "mismatch": float(item["mismatch"]),
                }
            )
        per_sector.append(
            {"sector_id": sid, "occupancy": occupancy.get(sid, 0), "matches": matches}
        )

    merged = _merged_reference_ranking(
        reference_rankings, occupied_sectors, vina_lookup,
    )
    return {
        "schema": "pocketfield.reference_screening.v1",
        "reference_ligand": str(reference_ligand),
        "anchor_structure": str(anchor_structure) if anchor_structure else None,
        "occupied_sectors": occupied_sectors,
        "excluded_anchor_sectors": excluded_sectors,
        "vector_keys": VECTOR_KEYS,
        "per_sector": per_sector,
        "merged": merged[:top],
    }


def _merged_reference_ranking(
    reference_rankings: dict[int, list[dict[str, Any]]],
    occupied_sectors: list[int],
    vina_lookup: dict[str, str],
) -> list[dict[str, Any]]:
    by_name: dict[str, dict[str, Any]] = {}
    for sid in occupied_sectors:
        for item in reference_rankings.get(sid, []):
            name = item["fragment_name"]
            if name not in by_name:
                by_name[name] = {
                    "fragment_name": name,
                    "smiles": item["fragment"].smiles,
                    "scores": [],
                    "best_score": float("inf"),
                    "best_sector_id": None,
                }
            entry = by_name[name]
            score = float(item["score"])
            entry["scores"].append(score)
            if score < entry["best_score"]:
                entry["best_score"] = score
                entry["best_sector_id"] = sid

    merged = []
    for name, entry in by_name.items():
        merged.append(
            {
                "fragment_name": name,
                "smiles": entry["smiles"],
                "vina_kcal_mol": _parse_vina(vina_lookup.get(name)),
                "mean_score": float(np.mean(entry["scores"])),
                "best_score": float(entry["best_score"]),
                "best_sector_id": entry["best_sector_id"],
            }
        )
    merged.sort(key=lambda item: (item["mean_score"], item["best_score"]))
    return merged
