"""Growth request generation.

Turns ranked fragments/linkers and dummy-sector assignments into concrete
"growth requests": a list of attachments (fragment or linker + anchor map)
plus a target direction, which :func:`pocketfield.rdkit_grow.grow_candidates`
assembles and scores.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from pocketfield.assembly import _find_dummy
from pocketfield.geometry import _mean_direction


def _fragment_sector_cover(
    fragment_rankings: dict[int, list[dict[str, Any]]],
    sectors: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    """Group fragments by which sectors they cover (are in top-`limit` of).

    Returns list of {fragment, fragment_name, covered_sectors, sector_scores}
    sorted by number of covered sectors (descending), then mean score.
    """
    by_name: dict[str, dict[str, Any]] = {}
    for sector in sectors:
        sid = sector["sector_id"]
        for ranked in _ranked_fragments(fragment_rankings, sid, limit):
            name = ranked["fragment_name"]
            if name not in by_name:
                by_name[name] = {
                    "fragment": ranked["fragment"],
                    "fragment_name": name,
                    "covered_sectors": [],
                    "sector_scores": {},
                }
            entry = by_name[name]
            if not any(s["sector_id"] == sid for s in entry["covered_sectors"]):
                entry["covered_sectors"].append(sector)
                entry["sector_scores"][sid] = ranked["score"]

    result = list(by_name.values())
    result.sort(key=lambda e: (
        -len(e["covered_sectors"]),
        float(np.mean(list(e["sector_scores"].values()))) if e["sector_scores"] else 0.0,
    ))
    return result


def _resolve_sector_competition(
    covers: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    """Resolve overlapping sectors by score competition across fragments.

    For sectors claimed by multiple fragments, the best (lowest) score wins.
    Returns resolved covers with sectors removed from losers, or None if
    any fragment is left with zero sectors.
    """
    sector_claims: dict[int, list[tuple[int, float]]] = {}
    for i, cover in enumerate(covers):
        for sid, score in cover["sector_scores"].items():
            if sid not in sector_claims:
                sector_claims[sid] = []
            sector_claims[sid].append((i, score))

    contested = {sid: claims for sid, claims in sector_claims.items() if len(claims) > 1}

    resolved: list[dict[str, Any]] = []
    for i, cover in enumerate(covers):
        kept_sectors = []
        kept_scores: dict[int, float] = {}
        for sector in cover["covered_sectors"]:
            sid = sector["sector_id"]
            if sid in contested:
                claims = contested[sid]
                best_idx, _best_score = min(claims, key=lambda x: x[1])
                if best_idx == i:
                    kept_sectors.append(sector)
                    kept_scores[sid] = cover["sector_scores"][sid]
            else:
                kept_sectors.append(sector)
                kept_scores[sid] = cover["sector_scores"][sid]

        if not kept_sectors:
            return None

        resolved.append({
            "fragment": cover["fragment"],
            "fragment_name": cover["fragment_name"],
            "covered_sectors": kept_sectors,
            "sector_scores": kept_scores,
        })

    return resolved


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
    sector_stats: dict[int, dict[str, float]],
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
        covers = _fragment_sector_cover(
            fragment_rankings, sectors, fragments_per_sector,
        )
        for cover in covers:
            covered = cover["covered_sectors"]
            direction = _mean_direction([s["direction"] for s in covered])
            scores = [
                _sector_normalized_score(
                    cover["sector_scores"][s["sector_id"]], s,
                    sector_stats[s["sector_id"]],
                )
                for s in covered
            ]
            requests.append(
                {
                    "mode": "single",
                    "sector_ids": [s["sector_id"] for s in covered],
                    "dummy_sector_angles": [s["angle_deg"] for s in covered],
                    "direction": direction,
                    "attachments": [
                        {
                            "map": anchor_map,
                            "fragment": cover["fragment"],
                            "sector_fragment_score": float(np.mean(scores)),
                        }
                    ],
                }
            )
            if len(requests) >= request_limit:
                return requests

    if include_pairs and len(dummy_options) >= 2:
        pair_limit = max(1, fragments_per_sector // 2)
        for dummy_pair in itertools.combinations(dummy_options, 2):
            (map_a, sectors_a), (map_b, sectors_b) = dummy_pair
            covers_a = _fragment_sector_cover(fragment_rankings, sectors_a, pair_limit)
            covers_b = _fragment_sector_cover(fragment_rankings, sectors_b, pair_limit)
            if not covers_a or not covers_b:
                continue
            for cover_a in covers_a:
                for cover_b in covers_b:
                    resolved = _resolve_sector_competition([cover_a, cover_b])
                    if resolved is None:
                        continue
                    res_a, res_b = resolved
                    all_sectors = res_a["covered_sectors"] + res_b["covered_sectors"]
                    direction = _mean_direction([s["direction"] for s in all_sectors])
                    score_a = float(np.mean([
                        _sector_normalized_score(
                            res_a["sector_scores"][s["sector_id"]],
                            s, sector_stats[s["sector_id"]],
                        )
                        for s in res_a["covered_sectors"]
                    ])) if res_a["covered_sectors"] else 0.0
                    score_b = float(np.mean([
                        _sector_normalized_score(
                            res_b["sector_scores"][s["sector_id"]],
                            s, sector_stats[s["sector_id"]],
                        )
                        for s in res_b["covered_sectors"]
                    ])) if res_b["covered_sectors"] else 0.0

                    requests.append(
                        {
                            "mode": "pair",
                            "sector_ids": [s["sector_id"] for s in all_sectors],
                            "dummy_sector_angles": [s["angle_deg"] for s in all_sectors],
                            "direction": direction,
                            "attachments": [
                                {
                                    "map": map_a,
                                    "fragment": res_a["fragment"],
                                    "sector_fragment_score": score_a,
                                },
                                {
                                    "map": map_b,
                                    "fragment": res_b["fragment"],
                                    "sector_fragment_score": score_b,
                                },
                            ],
                        }
                    )
                    if len(requests) >= request_limit:
                        return requests

    for depth in range(3, min(growth_depth, len(dummy_options)) + 1):
        for dummy_combo in itertools.combinations(dummy_options, depth):
            cover_lists = [
                _fragment_sector_cover(
                    fragment_rankings,
                    dummy_sectors[anchor_map],
                    fragments_per_sector,
                )
                for anchor_map, _sectors in dummy_combo
            ]
            if any(not cl for cl in cover_lists):
                continue
            for fragment_combo in itertools.product(*cover_lists):
                resolved = _resolve_sector_competition(list(fragment_combo))
                if resolved is None:
                    continue
                all_sectors = [s for r in resolved for s in r["covered_sectors"]]
                direction = _mean_direction([s["direction"] for s in all_sectors])
                requests.append(
                    {
                        "mode": f"multi_{depth}",
                        "sector_ids": [s["sector_id"] for s in all_sectors],
                        "dummy_sector_angles": [s["angle_deg"] for s in all_sectors],
                        "direction": direction,
                        "attachments": [
                            {
                                "map": dummy_combo[index][0],
                                "fragment": resolved[index]["fragment"],
                                "sector_fragment_score": float(np.mean([
                                    _sector_normalized_score(
                                        resolved[index]["sector_scores"][s["sector_id"]],
                                        s,
                                        sector_stats[s["sector_id"]],
                                    )
                                    for s in resolved[index]["covered_sectors"]
                                ])),
                            }
                            for index in range(len(resolved))
                        ],
                    }
                )
                if len(requests) >= request_limit:
                    return requests
    return requests


def _dummy_adjusted_score(fragment_score: float, sector: dict[str, Any]) -> float:
    return float(fragment_score + 0.25 * sector["angular_score"])


def _sector_normalized_score(raw_score: float, sector: dict[str, Any],
                              stats: dict[str, float]) -> float:
    """Z-score normalise then sigmoid to [0,1] (0=best match), then add angular penalty."""
    z = (raw_score - stats["mean"]) / max(stats["std"], 1e-8)
    norm = float(1.0 / (1.0 + np.exp(-z)))  # sigmoid: 0=best fragment
    return float(norm + 0.25 * sector["angular_score"])


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
        # Union of sectors from both dummies — one linker spans the full pocket
        union_sectors = sectors_a + sectors_b
        union_sector_ids = [s["sector_id"] for s in union_sectors]
        union_angles = [s["angle_deg"] for s in union_sectors]
        union_direction = _mean_direction([s["direction"] for s in union_sectors])
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
                    score_a = _dummy_adjusted_score(linker["score"], sector_a)
                    score_b = _dummy_adjusted_score(linker["score"], sector_b)
                    requests.append(
                        {
                            "mode": "bridge",
                            "sector_ids": union_sector_ids,
                            "dummy_sector_angles": union_angles,
                            "linker_geometry": {
                                "target_span": linker.get("target_span"),
                                "linker_span": linker.get("linker_span"),
                                "span_error": linker.get("span_error"),
                            },
                            "direction": union_direction,
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
    linker: Any,
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
    # Only reject linkers that are too short to span the gap.
    # Longer linkers can adopt a more compact conformation in the pocket;
    # the 3D embedding and clash/field scoring will determine fitness.
    best_span = max(spans)
    shortfall = max(0.0, target - best_span)
    if shortfall > tolerance:
        return None
    return {
        "target": target,
        "span": float(best_span),
        "error": float(shortfall),
        "penalty": float(shortfall / max(tolerance, 1e-6)) if tolerance > 0.0 else 0.0,
    }


def _sample_linker_spans(Chem: Any, linker: Any, conformers: int) -> list[float]:
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
    except Exception:
        cache[cache_key] = []
        return []
    finally:
        RDLogger.EnableLog("rdApp.*")


def _load_linker_span_cache(linker_library: str | Path) -> dict[tuple[str, int], list[float]]:
    """Load persisted linker span cache from disk if it exists."""
    cache_path = Path(str(linker_library) + ".spans.json")
    if not cache_path.exists():
        return {}
    try:
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        return {(str(k.split("||")[0]), int(k.split("||")[1])): v for k, v in raw.items()}
    except Exception:
        return {}


def _save_linker_span_cache(linker_library: str | Path, cache: dict) -> None:
    """Persist linker span cache to disk."""
    cache_path = Path(str(linker_library) + ".spans.json")
    serializable = {f"{smi}||{conf}": spans for (smi, conf), spans in cache.items()}
    cache_path.write_text(json.dumps(serializable, ensure_ascii=False), encoding="utf-8")
