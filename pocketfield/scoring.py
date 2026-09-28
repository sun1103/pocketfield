"""Candidate scoring against the pocket field.

Scores a 3D candidate molecule by its interaction-energy fit to the field
plus a steric clash term, normalised to [0, 1] (lower is better). Also holds
the atom-type probe/radius tables used for field evaluation.
"""

from __future__ import annotations

from typing import Any

import numpy as np


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


def _score_candidate(
    mol: Any,
    center: np.ndarray,
    shells: np.ndarray,
    directions: np.ndarray,
    energies: np.ndarray,
    probe_names: list[str],
    pocket_atoms: list[dict[str, Any]],
    field_mins: np.ndarray,
    field_ranges: np.ndarray,
    clash_cap: float = 3.0,
) -> dict[str, float]:
    """Score a candidate molecule against the pocket field.

    All scores are normalised to [0, 1] where 0 = best.

    Field score
    -----------
    Per-atom field energy normalised to [0, 1] using a fixed 50 kcal/mol cap
    above the per-(probe, shell) minimum:  norm = (raw - min) / 50, clipped.
    The cap avoids compression by the VDW repulsive wall (~10^6 kcal/mol).
    field_score = mean(norm) across heavy atoms.

    Clash score
    -----------
    Steric overlap² between ligand and pocket atoms. Backbone atoms (N, CA,
    C, O) get full penalty; sidechain atoms get 0.3× weight since they can
    yield. Normalised per heavy atom with a clash_cap (Å²) that saturates
    at 1.0.

    Final score
    -----------
    score = 0.35 * field_score + 0.30 * clash_score + 0.15 * (n_heavy / 70)

    Sector match is computed separately (z-scored then sigmoid-normalised)
    and used for fragment selection, but NOT included in the final score
    due to 0.93 correlation with field_score.

    Parameters
    ----------
    clash_cap : float
        Per-heavy-atom clash Å² that saturates the clash score at 1.0.
    """
    BACKBONE_NAMES = {'N', 'CA', 'C', 'O', 'H', 'HA', 'OXT'}
    conf = mol.GetConformer()
    probe_index = {name: index for index, name in enumerate(probe_names)}
    field_terms = []
    clash_score = 0.0  # backbone-weighted (full penalty)
    clash_sc = 0.0     # sidechain-weighted (reduced penalty)

    pocket_coords = np.asarray([atom["coord"] for atom in pocket_atoms], dtype=np.float64)
    pocket_radii = np.asarray([atom.get("radius", 1.7) for atom in pocket_atoms], dtype=np.float64)
    is_backbone = np.array(
        [atom.get("name", "X") in BACKBONE_NAMES for atom in pocket_atoms],
        dtype=bool,
    )
    n_heavy = 0

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        n_heavy += 1
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
            raw = float(np.min(energies[compatible, shell_idx, sample_idx]))
            # Normalise to [0, 1] using the per-(probe, shell) min/max range.
            # We take the best (lowest) compatible probe at this position.
            best_probe = int(np.argmin(energies[compatible, shell_idx, sample_idx]))
            pidx = compatible[best_probe]
            norm = (raw - field_mins[pidx, shell_idx]) / field_ranges[pidx, shell_idx]
            norm = max(0.0, min(1.0, float(norm)))  # clip for safety
            field_terms.append(norm)

        if len(pocket_coords):
            ligand_radius = LIGAND_RADII.get(atom.GetAtomicNum(), 1.70)
            distances = np.linalg.norm(pocket_coords - pos[None, :], axis=1)
            overlap2 = np.maximum(0.0, pocket_radii + ligand_radius - 0.45 - distances) ** 2
            clash_score += float(np.sum(overlap2[is_backbone]))
            clash_sc += float(np.sum(overlap2[~is_backbone]))

    field_score = float(np.mean(field_terms)) if field_terms else 0.0
    # Clash: backbone gets full penalty, sidechain gets reduced (0.3×).
    # Per-atom overlap² capped at clash_cap Å² per heavy atom.
    sidechain_flex = 0.3
    clash_combined = clash_score + sidechain_flex * clash_sc
    clash_norm = min(1.0, clash_combined / max(1, n_heavy) / clash_cap) if n_heavy > 0 else 0.0
    # Heavy-atom penalty: fraction of max allowed.
    heavy_penalty = n_heavy / 70.0  # 70 heavy atoms → 1.0

    score = 0.35 * field_score + 0.30 * clash_norm + 0.15 * heavy_penalty
    return {
        "field_score": field_score,
        "clash_score": clash_norm,
        "clash_score_raw": clash_combined,
        "clash_score_bb": clash_score,
        "clash_score_sc": clash_sc,
        "score": score,
    }


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
