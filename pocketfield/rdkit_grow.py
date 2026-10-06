"""Fragment/linker growth: the ``pocketfield grow`` orchestrator.

This module is the thin entry point that ties the growth pipeline together:
loading libraries, building feature vectors, ranking fragments/linkers,
assigning sectors to anchor dummies, generating growth requests, assembling
and scoring 3D candidates, and writing the ranked outputs.

The individual pieces live in :mod:`pocketfield.library`,
:mod:`pocketfield.features`, :mod:`pocketfield.geometry`,
:mod:`pocketfield.assembly`, :mod:`pocketfield.scoring`, and
:mod:`pocketfield.growth`. Several names are re-exported here for backward
compatibility with the pre-split single-file layout.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np

from pocketfield.attention import cross_attention_refine
from pocketfield.io import write_json

from pocketfield.assembly import (
    _align_candidate_to_field,
    _assemble_molecule,
    _axis_angle,
    _count_bad_bonds,
    _strip_isotopes,
)
from pocketfield.features import (
    _fragment_features,
    _linker_features,
    _load_vina_lookup,
    _rank_fragments_by_sector,
    _rank_linkers_by_sector,
    _reference_screening_report,
    _reference_sector_coefficients,
    _score_vector_match,  # noqa: F401  (re-export for scripts/demo_attention.py)
    _sector_coefficients,
    _vector_payload,
)
from pocketfield.geometry import (
    _anchor_coord_map,
    _anchor_maps,
    _anchor_pose,
    _dummy_assignment_payload,
    _dummy_sector_assignments,
    _fit_to_anchor_pose,
    _linker_anchor_mapping,
    _reference_occupied_sectors,
)
from pocketfield.growth import (
    _growth_requests,
    _load_linker_span_cache,
    _sample_linker_spans,
    _save_linker_span_cache,
)
from pocketfield.library import (
    _validate_fragment_library,
    _validate_linker_library,
    load_fragments,
    load_linkers,
)
from pocketfield.scoring import (
    _dedupe_scored_candidates,
    _record_rejection,
    _score_candidate,  # noqa: F401  (re-export for scripts/score_docked.py)
)


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
    max_heavy_atoms: int | None,
    max_clash_score: float,
    max_field_score: float | None,
    sector_match_weight: float,
    retro_rules: str | Path | None,
    retro_script: str | Path | None,
    random_seed: int,
    mode: str = "auto",
    cross_attention_weight: float = 0.0,
    reference_ligand: str | Path | None = None,
    conformers: int = 10,
    pose_refine: bool = False,
    pose_refine_iters: int = 60,
    linker_fit_rmsd_tolerance: float = 3.0,
    diversity_threshold: float = 1.0,
    drop_2d: bool = True,
) -> dict[str, Any]:
    try:
        from rdkit import Chem, Geometry
        from rdkit.Chem import AllChem
        from rdkit import DataStructs
        from rdkit.Chem import rdFingerprintGenerator
    except ImportError as exc:
        raise RuntimeError(
            "RDKit is required for 'pocketfield grow'. Run in a Python environment that has "
            "RDKit installed, for example: python -m pocketfield.cli grow ..."
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
    sector_centers = np.asarray(field["sector_centers"], dtype=np.float64)
    pocket_atoms = metadata.get("pocket_atoms", [])

    # Resolve growth mode early so linker features/rankings are computed and
    # written only when bridge mode actually uses linkers.
    anchor_maps = _anchor_maps(Chem, anchor_smiles)
    if not anchor_maps:
        raise ValueError("Anchor SMILES must contain at least one dummy atom such as [*:1].")
    mode = mode.lower()
    if mode == "fragment":
        connect_anchor_dummies = False
    elif mode == "bridge":
        connect_anchor_dummies = True
    # "auto" — keep connect_anchor_dummies flag as-is.

    # Precompute per-(probe, shell) minimum field energy across all directions.
    # Normalisation uses a fixed 50 kcal/mol cap rather than the actual max,
    # because the global max is dominated by the VDW repulsive wall (~10^6
    # kcal/mol), which compresses the chemically meaningful region to near-zero.
    field_mins = energies.min(axis=2)   # (n_probes, n_shells)
    field_cap = 50.0  # kcal/mol above minimum saturates at 1.0
    field_ranges = np.full_like(field_mins, field_cap)
    fragments = load_fragments(fragment_library)
    _validate_fragment_library(Chem, fragments)
    if connect_anchor_dummies:
        linkers = load_linkers(linker_library)
        _validate_linker_library(Chem, linkers)
    else:
        linkers = []
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
    linker_features = _linker_features(Chem, linkers) if connect_anchor_dummies else {}
    fragment_rankings, sector_stats = _rank_fragments_by_sector(sector_coefficients, fragment_features)
    linker_rankings = (
        _rank_linkers_by_sector(sector_coefficients, linker_features)
        if connect_anchor_dummies
        else {}
    )

    # ── Dummy-sector assignments (needed by cross-attention for co-occupancy) ──
    dummy_sectors = _dummy_sector_assignments(
        Chem=Chem,
        anchor_structure=anchor_structure,
        anchor_maps=anchor_maps,
        center=center,
        sector_coefficients=sector_coefficients,
        sectors_per_dummy=sectors_per_dummy,
        angle_cutoff=dummy_angle_cutoff,
    )
    # Build co-assignment map and collect assigned sector IDs.
    # Only sectors assigned to at least one dummy host fragments;
    # unassigned sectors are noise in both ranking and cross-attention.
    co_assignment: dict[int, set[int]] = {}
    assigned_sids: set[int] = set()
    for sectors in dummy_sectors.values():
        sids = [s["sector_id"] for s in sectors]
        assigned_sids.update(sids)
        for sid in sids:
            co_assignment.setdefault(sid, set()).update(sids)

    # ── Cross-attention: fragment attends to nearby sector feature vectors ──
    # Restricted to assigned sectors only — unassigned sectors will never
    # host a fragment and would only add noise.
    if cross_attention_weight > 0:
        if len(assigned_sids) > 1:
            ca_coeffs = {s: sector_coefficients[s] for s in assigned_sids}
            ca_rankings = {s: fragment_rankings[s] for s in assigned_sids}
            ca_rankings = cross_attention_refine(
                ca_rankings, fragment_features,
                ca_coeffs, sector_centers,
                context_weight=cross_attention_weight,
                co_assignment=co_assignment,
            )
            fragment_rankings.update(ca_rankings)
            # Recompute per-sector stats from updated scores.
            for sid, ranked in fragment_rankings.items():
                scores = [item["score"] for item in ranked]
                arr = np.asarray(scores, dtype=np.float64)
                sector_stats[sid] = {"mean": float(np.mean(arr)), "std": float(np.std(arr))}
    write_json(out / "sector_coefficients.json", _vector_payload("sector_coefficients", sector_coefficients))
    write_json(out / "fragment_features.json", _vector_payload("fragment_features", fragment_features))
    if connect_anchor_dummies:
        write_json(out / "linker_features.json", _vector_payload("linker_features", linker_features))
    write_json(out / "dummy_sector_assignments.json", _dummy_assignment_payload(dummy_sectors))
    write_json(out / "sector_stats.json", {
        "schema": "pocketfield.sector_stats.v1",
        "stats": {str(k): v for k, v in sector_stats.items()},
    })

    # ── Reference-occupancy screening ──
    # Screen fragments against the sectors the reference ligand (or its non-anchor
    # arm) actually occupies, rather than the dummy exit-vector sectors. Optional;
    # only runs when --reference-ligand is supplied.
    if reference_ligand is not None:
        occupied_ids, occupancy, excluded_ids = _reference_occupied_sectors(
            Chem=Chem,
            reference_path=reference_ligand,
            center=center,
            sector_centers=sector_centers,
            exclude_path=anchor_structure,
        )
        reference_coefficients = _reference_sector_coefficients(
            sector_ids=occupied_ids,
            sector_centers=sector_centers,
            energies=energies,
            shells=shells,
            field_sector_ids=sector_ids,
            probe_names=probe_names,
            pocket_atoms=pocket_atoms,
            center=center,
        )
        reference_rankings, _ = _rank_fragments_by_sector(reference_coefficients, fragment_features)
        screening = _reference_screening_report(
            reference_ligand=str(reference_ligand),
            anchor_structure=str(anchor_structure) if anchor_structure else None,
            occupied_sectors=occupied_ids,
            occupancy=occupancy,
            excluded_sectors=excluded_ids,
            reference_coefficients=reference_coefficients,
            reference_rankings=reference_rankings,
            fragment_features=fragment_features,
            vina_lookup=_load_vina_lookup(str(fragment_library) if fragment_library else None),
            top=top_sectors,
        )
        write_json(out / "reference_screening.json", screening)
        write_json(
            out / "reference_sector_coefficients.json",
            _vector_payload("reference_sector_coefficients", reference_coefficients),
        )

    anchor_pose = _anchor_pose(Chem, anchor_structure)

    # Restore linker span cache from disk so conformer sampling is skipped on reruns.
    span_cache = _load_linker_span_cache(linker_library) if linker_library is not None else {}
    if span_cache:
        setattr(_sample_linker_spans, "_cache", span_cache)

    if mode == "fragment" and growth_depth > len(anchor_maps):
        growth_depth = len(anchor_maps)

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
        sector_stats=sector_stats,
        drop_2d=drop_2d,
    )
    candidates = []
    attempted = 0
    rejections: dict[str, int] = {}
    accepted_patterns: list[dict[int, Any]] = []
    fp_cache: dict[str, Any] = {}
    linker_ends_cache: dict[str, tuple[Any, Any]] = {}
    morgan_gen: Any | None = None

    if diversity_threshold < 1.0:
        # Pre-step diversity: sort requests by the cheap (pre-3D) sector-match
        # score so the greedy max-min picker keeps the most promising member of
        # each cluster, then filter near-duplicate *fragments* (not whole
        # molecules — the anchor is constant and would swamp the similarity).
        morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

        def _cheap_score(request: dict[str, Any]) -> float:
            scores = [a["sector_fragment_score"] for a in request["attachments"]]
            return float(np.mean(scores)) if scores else 0.0

        def _linker_ends(smi: str) -> tuple[Any, Any]:
            # Morgan ignores atom-map numbers, so to tell the two ends of an
            # asymmetric linker apart we relabel one dummy with a collision-free
            # marker element (Po, Z=84) and fingerprint the linker twice.
            mol = Chem.MolFromSmiles(smi)

            def _end(keep_map: int) -> Any:
                rw = Chem.RWMol(mol)
                for atom in rw.GetAtoms():
                    if atom.GetAtomicNum() != 0:
                        continue
                    if atom.GetAtomMapNum() == keep_map:
                        atom.SetAtomicNum(84)
                    atom.SetAtomMapNum(0)
                return morgan_gen.GetFingerprint(rw.GetMol())

            return _end(1), _end(2)

        def _duplicate(pattern: dict[int, Any], accepted: list[dict[int, Any]]) -> bool:
            # A candidate duplicates an accepted one only when they fill the
            # same anchor maps with similar fragments at every map.
            for acc in accepted:
                if set(acc) != set(pattern):
                    continue
                if all(
                    DataStructs.TanimotoSimilarity(pattern[m], acc[m]) > diversity_threshold
                    for m in pattern
                ):
                    return True
            return False

        requests = sorted(requests, key=_cheap_score)

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
            if max_heavy_atoms is not None and mol.GetNumHeavyAtoms() > max_heavy_atoms:
                _record_rejection(rejections, "too_many_heavy_atoms")
                continue
            if morgan_gen is not None:
                pattern: dict[int, Any] = {}
                linker_end_idx: dict[str, int] = {}
                for a in request["attachments"]:
                    m = a["map"]
                    if "fragment" in a:
                        smi = a["fragment"].smiles
                        fp = fp_cache.get(smi)
                        if fp is None:
                            fp = morgan_gen.GetFingerprint(Chem.MolFromSmiles(smi))
                            fp_cache[smi] = fp
                        pattern[m] = fp
                    else:
                        smi = a["linker"].smiles
                        ends = linker_ends_cache.get(smi)
                        if ends is None:
                            ends = _linker_ends(smi)
                            linker_ends_cache[smi] = ends
                        idx = linker_end_idx.get(smi, 0)
                        pattern[m] = ends[idx]
                        linker_end_idx[smi] = idx + 1
                if _duplicate(pattern, accepted_patterns):
                    _record_rejection(rejections, "duplicate_similar")
                    continue
                accepted_patterns.append(pattern)

            mol3d = Chem.AddHs(mol)
            if anchor_pose is not None:
                coordMap, coordTranslation = _anchor_coord_map(
                    Chem, mol3d, anchor_smiles, anchor_structure,
                )
            else:
                coordMap = {}
                coordTranslation = (0.0, 0.0, 0.0)
            use_coordmap = len(coordMap) >= 2
            params = AllChem.ETKDGv3()
            params.randomSeed = random_seed + request_index
            params.pruneRmsThresh = -1.0
            if use_coordmap:
                params.SetCoordMap(coordMap)
            conf_ids = AllChem.EmbedMultipleConfs(mol3d, max(1, conformers), params)
            if not conf_ids:
                _record_rejection(rejections, "embedding_failed")
                continue
            # Optimize each conformer, then translate from origin-centred back
            # to the SDF coordinate frame.
            cx, cy, cz = coordTranslation
            freeze_anchor = len(coordMap) >= 2
            anchor_frozen_indices = [
                atom.GetIdx()
                for atom in mol3d.GetAtoms()
                if atom.HasProp("pocketfield_anchor") and atom.GetAtomicNum() > 1
            ] if freeze_anchor else []
            for conf_id in list(conf_ids):
                try:
                    if freeze_anchor:
                        ff = AllChem.UFFGetMoleculeForceField(mol3d, confId=conf_id)
                        for idx in anchor_frozen_indices:
                            ff.AddFixedPoint(idx)
                        ff.Minimize(maxIts=200)
                    else:
                        AllChem.UFFOptimizeMolecule(mol3d, confId=conf_id, maxIters=200)
                except Exception:
                    pass
                if len(coordMap) >= 2 and coordTranslation != (0.0, 0.0, 0.0):
                    conf = mol3d.GetConformer(conf_id)
                    for i in range(mol3d.GetNumAtoms()):
                        p = conf.GetAtomPosition(i)
                        conf.SetAtomPosition(i, Geometry.Point3D(p.x + cx, p.y + cy, p.z + cz))

            # Align and score every conformer; keep the best (lowest score).
            best_conf_id = None
            best_score = None
            scored: list[float] = []
            for conf_id in list(conf_ids):
                if _count_bad_bonds(mol3d, conf_id=conf_id) > max(3, mol.GetNumHeavyAtoms() // 10):
                    continue
                if anchor_pose is not None:
                    # Only run Kabsch when we did NOT already transfer coordinates
                    # via coordMap+translation (which leaves the molecule correctly placed).
                    if len(coordMap) < 2:
                        _fit_to_anchor_pose(mol3d, anchor_pose, conf_id=conf_id)
                else:
                    _align_candidate_to_field(mol3d, center, request["direction"], conf_id=conf_id)
                conf_score = _score_candidate(
                    mol3d,
                    center,
                    shells,
                    directions,
                    energies,
                    probe_names,
                    pocket_atoms,
                    field_mins,
                    field_ranges,
                    conf_id=conf_id,
                )
                scored.append(conf_score["score"])
                if best_score is None or conf_score["score"] < best_score["score"]:
                    best_conf_id = conf_id
                    best_score = conf_score
            if best_conf_id is None:
                _record_rejection(rejections, "bad_bond_geometry")
                continue
            score = best_score
            if pose_refine and anchor_pose is None:
                _refine_pose(
                    mol3d,
                    center,
                    shells,
                    directions,
                    energies,
                    probe_names,
                    pocket_atoms,
                    field_mins,
                    field_ranges,
                    conf_id=best_conf_id,
                    iters=pose_refine_iters,
                )
                score = _score_candidate(
                    mol3d,
                    center,
                    shells,
                    directions,
                    energies,
                    probe_names,
                    pocket_atoms,
                    field_mins,
                    field_ranges,
                    conf_id=best_conf_id,
                )
            if score["clash_score"] > max_clash_score:
                _record_rejection(rejections, "clash_filter")
                continue
            if max_field_score is not None and score["field_score"] > max_field_score:
                _record_rejection(rejections, "field_filter")
                continue
            linker_fit_rmsd = _linker_fit_rmsd(mol3d, anchor_pose, conf_id=best_conf_id)
            if (
                linker_fit_rmsd is not None
                and linker_fit_rmsd_tolerance is not None
                and linker_fit_rmsd > linker_fit_rmsd_tolerance
            ):
                _record_rejection(rejections, "poor_linker_closure")
                continue
            _keep_single_conformer(Chem, mol3d, best_conf_id)
            sector_match_score = float(
                np.mean([attachment["sector_fragment_score"] for attachment in request["attachments"]])
            )
            # Sector match was already used for fragment selection.
            # Final score uses only field + clash (orthogonal 3D fit signals).
            final_score = score["score"]
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
                "linker_geometry": (
                    {**request["linker_geometry"], "fit_rmsd": linker_fit_rmsd}
                    if request.get("linker_geometry") is not None and linker_fit_rmsd is not None
                    else request.get("linker_geometry")
                ),
                "sector_fragment_scores": [
                    attachment["sector_fragment_score"] for attachment in request["attachments"]
                ],
                "sector_match_score": sector_match_score,
                "attachment_maps": [attachment["map"] for attachment in request["attachments"]],
                "mode": request["mode"],
                "field_score": score["field_score"],
                "clash_score": score["clash_score"],
                "clash_score_bb": score.get("clash_score_bb", 0.0),
                "clash_score_sc": score.get("clash_score_sc", 0.0),
                "raw_score": score["score"],
                "score": final_score,
                "heavy_atoms": mol.GetNumHeavyAtoms(),
                "n_conformers": len(scored),
                "score_mean": float(np.mean(scored)) if scored else 0.0,
                "score_std": float(np.std(scored)) if scored else 0.0,
                "best_conf_id": int(best_conf_id),
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
            "diversity_threshold": diversity_threshold,
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
            **({"linker_features": "linker_features.json"} if connect_anchor_dummies else {}),
            "dummy_sector_assignments": "dummy_sector_assignments.json",
            **(
                {
                    "reference_screening": "reference_screening.json",
                    "reference_sector_coefficients": "reference_sector_coefficients.json",
                }
                if reference_ligand is not None
                else {}
            ),
        },
    }
    write_json(out / "candidates.json", result)

    # Persist linker span cache to disk for faster reruns.
    span_cache = getattr(_sample_linker_spans, "_cache", None)
    if linker_library is not None and span_cache:
        _save_linker_span_cache(linker_library, span_cache)

    return result


def _load_retro_validator(rules_path: str | Path | None, script_path: str | Path | None) -> Any | None:
    if rules_path is None:
        return None
    if script_path is None:
        script = Path(__file__).resolve().parent / "merge_utils.py"
    else:
        script = Path(script_path)
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
            "Run in a Python environment that has rdchiral, or disable --retro-rules."
        ) from exc


def _keep_single_conformer(Chem: Any, mol: Any, conf_id: int) -> None:
    """Drop all conformers except *conf_id*, re-homed as conformer 0."""
    conf = mol.GetConformer(conf_id)
    positions = [conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())]
    mol.RemoveAllConformers()
    new_conf = Chem.Conformer(mol.GetNumAtoms())
    new_conf.SetId(0)
    for i, p in enumerate(positions):
        new_conf.SetAtomPosition(i, p)
    mol.AddConformer(new_conf, assignId=False)


def _linker_fit_rmsd(mol: Any, anchor_pose: dict[str, Any] | None, conf_id: int = 0) -> float | None:
    """RMSD (Å) of linker attachment atoms vs their target anchor dummy coords."""
    if anchor_pose is None:
        return None
    dummy_coords = anchor_pose.get("dummy_coords", {})
    if not dummy_coords:
        return None
    link_to_anchor = _linker_anchor_mapping(mol)
    conf = mol.GetConformer(conf_id)
    errs = []
    for atom in mol.GetAtoms():
        if atom.HasProp("pocketfield_linker_map") and atom.GetAtomicNum() > 1:
            anchor_map = link_to_anchor.get(atom.GetIntProp("pocketfield_linker_map"))
            if anchor_map is not None and anchor_map in dummy_coords:
                p = conf.GetAtomPosition(atom.GetIdx())
                target = dummy_coords[anchor_map]
                errs.append(float(np.linalg.norm([p.x - target[0], p.y - target[1], p.z - target[2]])))
    if not errs:
        return None
    return float(np.sqrt(np.mean(np.square(errs))))


def _nelder_mead(
    f: Any,
    x0: np.ndarray,
    step: np.ndarray,
    max_iter: int,
    alpha: float = 1.0,
    gamma: float = 2.0,
    rho: float = 0.5,
    sigma: float = 0.5,
) -> tuple[np.ndarray, float]:
    """Deterministic Nelder-Mead simplex (NumPy-only, no scipy)."""
    n = len(x0)
    simplex = [np.asarray(x0, dtype=np.float64)]
    for i in range(n):
        p = np.asarray(x0, dtype=np.float64)
        p[i] += step[i]
        simplex.append(p)
    fvals = [float(f(p)) for p in simplex]

    for _ in range(max_iter):
        order = sorted(range(n + 1), key=lambda i: fvals[i])
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]

        centroid = np.mean(simplex[:-1], axis=0)
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = float(f(xr))

        if fr < fvals[0]:
            xe = centroid + gamma * (xr - centroid)
            fe = float(f(xe))
            if fe < fr:
                simplex[-1], fvals[-1] = xe, fe
            else:
                simplex[-1], fvals[-1] = xr, fr
        elif fr < fvals[-2]:
            simplex[-1], fvals[-1] = xr, fr
        else:
            if fr < fvals[-1]:
                simplex[-1], fvals[-1] = xr, fr
            xc = centroid + rho * (simplex[-1] - centroid)
            fc = float(f(xc))
            if fc < fvals[-1]:
                simplex[-1], fvals[-1] = xc, fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    fvals[i] = float(f(simplex[i]))

    best = int(np.argmin(fvals))
    return simplex[best], fvals[best]


def _refine_pose(
    mol: Any,
    center: np.ndarray,
    shells: np.ndarray,
    directions: np.ndarray,
    energies: np.ndarray,
    probe_names: list[str],
    pocket_atoms: list[dict[str, Any]],
    field_mins: np.ndarray,
    field_ranges: np.ndarray,
    conf_id: int = 0,
    iters: int = 60,
) -> None:
    """Deterministic rigid-body (6-DOF) pose refinement for the de-novo case.

    Minimises the PocketField score over translation + rotation (axis-angle),
    seeded from the current conformer and left at the refined pose.
    """
    from rdkit import Geometry

    conf = mol.GetConformer(conf_id)
    base = np.asarray(
        [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())],
        dtype=np.float64,
    )
    center0 = base.mean(axis=0)

    def objective(x: np.ndarray) -> float:
        tx, ty, tz, rx, ry, rz = x
        t = np.array([tx, ty, tz], dtype=np.float64)
        axis = np.array([rx, ry, rz], dtype=np.float64)
        angle = float(np.linalg.norm(axis))
        rot = _axis_angle(axis / angle, angle) if angle > 1e-8 else np.eye(3)
        moved = (base - center0) @ rot.T + center0 + t
        for i, xyz in enumerate(moved):
            conf.SetAtomPosition(i, Geometry.Point3D(float(xyz[0]), float(xyz[1]), float(xyz[2])))
        return _score_candidate(
            mol, center, shells, directions, energies, probe_names,
            pocket_atoms, field_mins, field_ranges, conf_id=conf_id,
        )["score"]

    step = np.array([1.0, 1.0, 1.0, 0.3, 0.3, 0.3], dtype=np.float64)
    best_x, _ = _nelder_mead(objective, np.zeros(6), step, iters)
    objective(best_x)  # leave the conformer at the refined pose
