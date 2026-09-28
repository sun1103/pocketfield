"""Multi-head cross-attention for fragment-sector re-ranking.

A fragment bound in one sub-pocket is influenced by what sits in neighbouring
sub-pockets — the "global view". This module refines the per-sector fragment
rankings produced by :func:`pocketfield.rdkit_grow._rank_fragments_by_sector`
without any learned parameters.

Feature vectors are 15-dim and shared by sectors and fragments:

.. code-block:: text

   [0] hydrophobic       [1] hbond_donor   [2] hbond_acceptor
   [3] positive          [4] negative      [5] aromatic
   [6] steric_openness   [7] small_size    [8] medium_size
   [9] large_size        [10] rigidity     [11] polarity
   [12] depth            [13] water_displacement  [14] buried_polar_support

The mechanism (five independent levers, no learning):

1. **Sparse attention** — only the top-k spatially closest active sectors matter.
2. **Co-assignment exclusion** — sectors bound by the same fragment (shared
   dummy) are self, not neighbours.
3. **Per-fragment lambda modulation** — charged/polar fragments are more
   sensitive to their neighbour environment.
4. **Confidence gating** — neighbours agree → signal amplified; disagree → dampened.
5. **Self-sector check** — fragment similarity to its own sector reinforces the
   base score.

Three attention heads are averaged:

* **Pharm head** (dims 0-5): complementarity — donor↔acceptor, pos↔neg.
* **Shape head** (dims 7-9): similarity — size compatibility.
* **Polar head** (dims 6, 10-14): similarity — environmental fit.

The per-fragment loop is vectorised: all fragments in a sector are scored
against the top-k neighbours as a single ``(n_frag, k)`` matrix per head, so
runtime stays flat as the fragment library grows.
"""

from __future__ import annotations

from typing import Any

import numpy as np


PHARM_DIMS = slice(0, 6)        # hydrophobic, donor, acceptor, pos, neg, aromatic
SHAPE_DIMS = slice(7, 10)       # small, medium, large
POLAR_DIMS = [6, 10, 11, 12, 13, 14]

# Pharmacophore complementarity pairing: frag dim -> sec dim it pairs with.
# donor↔acceptor (1→2), acceptor↔donor (2→1), pos↔neg (3→4), neg↔pos (4→3);
# hydrophobic/aromatic are like↔like (0→0, 5→5).
_PHARM_PAIR = [0, 2, 1, 4, 3, 5]

# Charge/polarity modulation saturates smoothly (see charge_mod below), so no
# explicit cap is needed: 1 + 0.8·tanh(0.5·interactivity) ∈ [1.0, ~1.72].
CHARGE_MOD_SCALE = 0.8
CHARGE_MOD_GAIN = 0.5


def _cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between rows of ``a`` (n,d) and ``b`` (m,d) -> (n,m)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    dots = a @ b.T
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    return dots / np.maximum(na[:, None] * nb[None, :], 1e-8)


def _pharm_complementarity_matrix(frag: np.ndarray, sec: np.ndarray) -> np.ndarray:
    """Pharmacophore complementarity between ``frag`` (n,6) and ``sec`` (m,6) -> (n,m).

    ``frag`` and ``sec`` are [hydrophobic, donor, acceptor, pos, neg, aromatic];
    the donor/acceptor and pos/neg axes are crossed so that complementary
    partners score high. Returns values in [-1, 1], 1 = perfectly complementary.
    """
    frag = np.asarray(frag, dtype=np.float64)
    sec = np.asarray(sec, dtype=np.float64)
    paired = sec[:, _PHARM_PAIR]
    dots = frag @ paired.T
    nf = np.linalg.norm(frag, axis=1)
    ns = np.linalg.norm(sec, axis=1)
    return dots / np.maximum(nf[:, None] * ns[None, :], 1e-8)


_ANGULAR_POWER = 2.0  # sharpening exponent on (1+cosθ)/2 (1.0 = flat, un-sharpened)


def _angular_proximity_matrix(centers: np.ndarray) -> np.ndarray:
    """Angular-proximity weights ``((1+cosθ)/2)^p`` between sector centers.

    Smooth and density-independent — a fixed-angle segment map is brittle
    because the nearest-neighbour separation scales with the number of sectors
    (≈55° for 12 sectors, ≈25° for 64). A higher ``_ANGULAR_POWER`` separates
    aligned from lateral neighbours more strongly without the hard zero of a
    step cutoff.
    """
    cos = np.clip(centers @ centers.T, -1.0, 1.0)
    return ((1.0 + cos) / 2.0) ** _ANGULAR_POWER


def _apply_self_check(
    ranked: list[dict[str, Any]],
    items: list[dict[str, Any]],
    frag_matrix: np.ndarray,
    own_vec: np.ndarray,
    self_weight: float,
) -> None:
    """Apply the self-sector check only (no neighbours), mutating item scores."""
    own_pharm = own_vec[PHARM_DIMS][None, :]
    own_shape = own_vec[SHAPE_DIMS][None, :]
    own_polar = own_vec[POLAR_DIMS][None, :]
    sc_pharm = _cosine_matrix(frag_matrix[:, PHARM_DIMS], own_pharm)[:, 0]
    sc_shape = _cosine_matrix(frag_matrix[:, SHAPE_DIMS], own_shape)[:, 0]
    sc_polar = _cosine_matrix(frag_matrix[:, POLAR_DIMS], own_polar)[:, 0]
    self_ctx = (sc_pharm + sc_shape + sc_polar) / 3.0

    for item, sctx in zip(items, self_ctx):
        item["score"] = item["score"] - self_weight * float(sctx)
        item["_ca_neighbor_ctx"] = 0.0
        item["_ca_self_ctx"] = float(sctx)
        item["_ca_top_k"] = []
        item["_ca_lam_eff"] = 0.0
        item["_ca_charge_mod"] = 1.0
        item["_ca_confidence"] = 0.0
        item["_ca_mean_nc"] = 0.0
        item["_ca_mean_sc"] = 0.0
    ranked.sort(key=lambda x: x["score"])


def cross_attention_refine(
    fragment_rankings: dict[int, list[dict[str, Any]]],
    fragment_features: dict[str, dict[str, Any]],
    sector_coefficients: dict[int, dict[str, Any]],
    sector_centers: np.ndarray,
    context_weight: float = 0.25,
    top_k: int = 4,
    self_weight: float = 0.10,
    co_assignment: dict[int, set[int]] | None = None,
) -> dict[int, list[dict[str, Any]]]:
    """Refine per-sector fragment rankings with multi-head cross-attention.

    ``fragment_rankings[sid]`` is a list of dicts, each holding at least
    ``fragment_name`` and ``score`` (lower is better). ``fragment_features`` is
    keyed by fragment name and holds the 15-dim ``vector``. ``sector_coefficients``
    is keyed by sector id and holds the sector ``vector``. ``sector_centers`` is
    the ``(n_sectors, 3)`` array from ``field.npz``.

    Sectors in ``co_assignment[sid]`` (bound by the same dummy as ``sid``) are
    excluded from ``sid``'s neighbour set. Mutates ``fragment_rankings`` in place
    and returns it, with ``score`` adjusted and ``_ca_*`` diagnostics attached.
    """
    sector_ids = sorted(sector_coefficients.keys())
    if len(sector_ids) < 2:
        return fragment_rankings

    centers = np.asarray(sector_centers, dtype=np.float64)
    angular_proximity = _angular_proximity_matrix(centers)  # (n_sec, n_sec)

    for sid in sector_ids:
        ranked = fragment_rankings.get(sid, [])
        if not ranked:
            continue

        # Collect the fragment vectors once; fragments without features are
        # skipped (they carry no vector to attend with).
        items: list[dict[str, Any]] = []
        frag_vecs: list[np.ndarray] = []
        for item in ranked:
            feat = fragment_features.get(item.get("fragment_name", ""))
            if feat is None:
                continue
            items.append(item)
            frag_vecs.append(np.asarray(feat["vector"], dtype=np.float64))
        if not items:
            continue
        F = np.stack(frag_vecs, axis=0)  # (n_frag, 15)

        own_vec = np.asarray(sector_coefficients[sid]["vector"], dtype=np.float64)

        # 1. Sparse attention: keep only top-k closest active neighbours.
        co = co_assignment.get(sid, set()) if co_assignment else set()
        other_sectors = [t for t in sector_ids if t != sid and t not in co]
        if not other_sectors:
            # All other active sectors are co-assigned to the same dummy.
            _apply_self_check(ranked, items, F, own_vec, self_weight)
            continue

        k = min(top_k, len(other_sectors))
        other_sectors = sorted(
            other_sectors, key=lambda t: angular_proximity[sid, t], reverse=True
        )[:k]

        prox_weights = np.array(
            [angular_proximity[sid, t] for t in other_sectors], dtype=np.float64
        )
        prox_sum = float(np.sum(prox_weights))
        if prox_sum < 1e-8:
            # No angularly meaningful neighbours: fall through to self-check
            # only rather than skipping this sector's refinement entirely.
            _apply_self_check(ranked, items, F, own_vec, self_weight)
            continue

        density = prox_sum / len(other_sectors)
        lam_base = context_weight * (1.0 + density)

        nbr_pharm = np.array(
            [sector_coefficients[t]["vector"][PHARM_DIMS] for t in other_sectors],
            dtype=np.float64,
        )
        nbr_shape = np.array(
            [sector_coefficients[t]["vector"][SHAPE_DIMS] for t in other_sectors],
            dtype=np.float64,
        )
        nbr_polar = np.array(
            [[sector_coefficients[t]["vector"][idx] for idx in POLAR_DIMS]
             for t in other_sectors],
            dtype=np.float64,
        )

        own_pharm = own_vec[PHARM_DIMS][None, :]
        own_shape = own_vec[SHAPE_DIMS][None, :]
        own_polar = own_vec[POLAR_DIMS][None, :]

        # Per-head scores against the k neighbours: (n_frag, k).
        pharm_scores = _pharm_complementarity_matrix(F[:, PHARM_DIMS], nbr_pharm)
        shape_scores = _cosine_matrix(F[:, SHAPE_DIMS], nbr_shape)
        polar_scores = _cosine_matrix(F[:, POLAR_DIMS], nbr_polar)

        # 2. Per-fragment lambda modulation by charge/polarity (smooth saturation).
        donor = F[:, 1]
        acceptor = F[:, 2]
        pos = F[:, 3]
        neg = F[:, 4]
        polarity = F[:, 11]
        interactivity = np.maximum(donor, acceptor) + np.maximum(pos, neg) + polarity
        charge_mod = 1.0 + CHARGE_MOD_SCALE * np.tanh(CHARGE_MOD_GAIN * interactivity)

        # Weighted mean (neighbour context) and std (confidence) per head.
        def _wmean_std(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            mean = scores @ prox_weights / prox_sum
            centered = scores - mean[:, None]
            var = (centered ** 2) @ prox_weights / prox_sum
            return mean, np.sqrt(var)

        ctx_pharm, std_pharm = _wmean_std(pharm_scores)
        ctx_shape, std_shape = _wmean_std(shape_scores)
        ctx_polar, std_polar = _wmean_std(polar_scores)

        # 3. Confidence gating: interpolate mean→min as heads disagree.
        conf_pharm = np.maximum(0.0, 1.0 - std_pharm / 0.40)
        conf_shape = np.maximum(0.0, 1.0 - std_shape / 0.40)
        conf_polar = np.maximum(0.0, 1.0 - std_polar / 0.40)
        conf_mean = (conf_pharm + conf_shape + conf_polar) / 3.0
        conf_min = np.minimum(np.minimum(conf_pharm, conf_shape), conf_polar)
        gap = conf_mean - conf_min
        w_min = np.minimum(1.0, gap / 0.667)
        confidence = (1.0 - w_min) * conf_mean + w_min * conf_min

        lam_neighbor = lam_base * charge_mod * confidence

        neighbor_ctx = (ctx_pharm + ctx_shape + ctx_polar) / 3.0

        # 4. Self-sector check (similarity, not complementarity).
        self_pharm = _cosine_matrix(F[:, PHARM_DIMS], own_pharm)[:, 0]
        self_shape = _cosine_matrix(F[:, SHAPE_DIMS], own_shape)[:, 0]
        self_polar = _cosine_matrix(F[:, POLAR_DIMS], own_polar)[:, 0]
        self_ctx = (self_pharm + self_shape + self_polar) / 3.0

        for i, item in enumerate(items):
            item["_ca_neighbor_ctx"] = float(neighbor_ctx[i])
            item["_ca_self_ctx"] = float(self_ctx[i])
            item["_ca_pharm"] = float(ctx_pharm[i])
            item["_ca_shape"] = float(ctx_shape[i])
            item["_ca_polar"] = float(ctx_polar[i])
            item["_ca_lambda_base"] = lam_base
            item["_ca_charge_mod"] = float(charge_mod[i])
            item["_ca_confidence"] = float(confidence[i])
            item["_ca_lam_eff"] = float(lam_neighbor[i])
            item["_ca_top_k"] = other_sectors

        # Mean-center contexts per sector so attention can penalise, not only bonus.
        mean_nc = float(np.mean(neighbor_ctx))
        mean_sc = float(np.mean(self_ctx))

        for i, item in enumerate(items):
            item["score"] = (
                item["score"]
                - float(lam_neighbor[i]) * (float(neighbor_ctx[i]) - mean_nc)
                - self_weight * float(charge_mod[i]) * (float(self_ctx[i]) - mean_sc)
            )
            item["_ca_mean_nc"] = mean_nc
            item["_ca_mean_sc"] = mean_sc

        ranked.sort(key=lambda x: x["score"])

    return fragment_rankings
