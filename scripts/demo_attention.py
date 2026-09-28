#!/usr/bin/env python3
"""Demonstrate the production cross-attention re-ranking in ``pocketfield grow``.

Loads the sector coefficients, fragment features, and dummy-sector assignments
from a pocketfield grow run, reconstructs the per-sector fragment rankings the
same way ``grow`` does, and applies the real multi-head cross-attention
(``pocketfield.attention.cross_attention_refine``) used when ``--cross-attention-weight``
is non-zero.

This imports the production helpers from ``pocketfield`` on purpose, so
the demo can never drift from what the pipeline actually runs.

Usage:
  python scripts/demo_attention.py \
      --grow-dir tests/6duk_grow \
      --field tests/6duk_field/field.npz
"""

import sys, os, json, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pocketfield.rdkit_grow import _score_vector_match
from pocketfield.attention import cross_attention_refine


# Feature vector layout (mirrors VECTOR_KEYS in rdkit_grow):
#   [0] hydrophobic  [1] hbond_donor  [2] hbond_acceptor  [3] positive
#   [4] negative     [5] aromatic     [6] steric_openness [7] small_size
#   [8] medium_size  [9] large_size   [10] rigidity       [11] polarity
#   [12] depth       [13] water_displacement  [14] buried_polar_support
PHARM = slice(0, 6)
SHAPE = slice(7, 10)
POLAR = 11


def _load(grow_dir, field_path):
    sector_data = json.load(open(os.path.join(grow_dir, "sector_coefficients.json")))
    fragment_data = json.load(open(os.path.join(grow_dir, "fragment_features.json")))
    dummy_data = json.load(open(os.path.join(grow_dir, "dummy_sector_assignments.json")))
    field = np.load(field_path)
    sector_centers = field["sector_centers"]  # (n_sectors, 3)

    sector_coefficients = {}
    for item in sector_data["items"]:
        sector_coefficients[item["sector_id"]] = {
            "sector_id": item["sector_id"],
            "vector": np.asarray(item["vector"], dtype=np.float64),
            "score": item.get("score", 0.0),
            "best_probe": item.get("best_probe", ""),
            "features": item.get("features", {}),
        }

    fragment_features = {}
    for item in fragment_data["items"]:
        fragment_features[item["name"]] = {
            "name": item["name"],
            "vector": np.asarray(item["vector"], dtype=np.float64),
            "features": item.get("features", {}),
            "smiles": item.get("smiles", ""),
            "probes": item.get("probes", []),
            "heavy_atoms": item.get("heavy_atoms", 0),
        }

    # Co-assignment map + active sectors, exactly as grow builds them.
    co_assignment = {}
    assigned_sids = set()
    for asgn in dummy_data.get("assignments", []):
        sids = [s["sector_id"] for s in asgn.get("sectors", [])]
        assigned_sids.update(sids)
        for sid in sids:
            co_assignment.setdefault(sid, set()).update(sids)

    return sector_coefficients, fragment_features, sector_centers, co_assignment, assigned_sids


def _rank_fragments(sector_coefficients, fragment_features, assigned_sids):
    """Rebuild per-sector fragment rankings with the production scoring fn."""
    rankings = {}
    for sid in assigned_sids:
        sector = sector_coefficients[sid]
        ranked = []
        for feature in fragment_features.values():
            match = _score_vector_match(sector, feature)
            ranked.append({"fragment_name": feature["name"], **match})
        ranked.sort(key=lambda x: x["score"])
        rankings[sid] = ranked
    return rankings


def _pharm_summary(vec):
    pharm = vec[PHARM]
    size = ["S", "M", "L"][int(np.argmax(vec[SHAPE]))]
    return f"pharm=[{' '.join(f'{v:.1f}' for v in pharm)}] size={size} polar={vec[POLAR]:.1f}"


def main():
    p = argparse.ArgumentParser(description="Demo the production cross-attention re-ranking")
    p.add_argument("--grow-dir", default="tests/6duk_grow")
    p.add_argument("--field", default="tests/6duk_field/field.npz")
    p.add_argument("--context-weight", type=float, default=0.25,
                   help="λ ∈ [0,1]: 0=ignore neighbors, 1=neighbors equally important")
    args = p.parse_args()

    (sector_coefficients, fragment_features, sector_centers,
     co_assignment, assigned_sids) = _load(args.grow_dir, args.field)

    sector_ids = sorted(assigned_sids)
    print(f"Active sectors (dummy assignments): {sector_ids}\n")

    rankings = _rank_fragments(sector_coefficients, fragment_features, assigned_sids)

    # ── Before ──
    print("=" * 80)
    print("BEFORE cross-attention (top 3 per sector)")
    print("=" * 80)
    for sid in sector_ids:
        probe = sector_coefficients[sid]["best_probe"]
        print(f"\nSector {sid} (probe={probe})")
        for rank, frag in enumerate(rankings[sid][:3], 1):
            print(f"  {rank}. {frag['fragment_name']:<20} score={frag['score']:.3f}  "
                  f"{_pharm_summary(fragment_features[frag['fragment_name']]['vector'])}")

    # Capture pre-refine scores for the delta display (refine mutates in place).
    orig_scores = {sid: [f["score"] for f in rankings[sid]] for sid in sector_ids}

    # ── Cross-attention (production path) ──
    cross_attention_refine(
        rankings,
        fragment_features,
        {s: sector_coefficients[s] for s in assigned_sids},
        sector_centers,
        context_weight=args.context_weight,
        co_assignment=co_assignment,
    )

    # ── After ──
    print("\n" + "=" * 80)
    print("AFTER cross-attention (top 3 per sector, re-ranked)")
    print("=" * 80)
    for sid in sector_ids:
        print(f"\nSector {sid}")
        for rank, frag in enumerate(rankings[sid][:3], 1):
            orig = orig_scores[sid][rank - 1]
            new = frag["score"]
            nctx = frag.get("_ca_neighbor_ctx", 0.0)
            sctx = frag.get("_ca_self_ctx", 0.0)
            chmod = frag.get("_ca_charge_mod", 1.0)
            conf = frag.get("_ca_confidence", 0.0)
            top_k = frag.get("_ca_top_k", [])
            print(f"  {rank}. {frag['fragment_name']:<20} orig={orig:.3f} → {new:.3f} "
                  f"(Δ={new - orig:+.3f})  nbr_ctx={nctx:.3f}  self_ctx={sctx:.3f}  "
                  f"charge_mod={chmod:.2f}  conf={conf:.2f}  top_k={top_k}")


if __name__ == "__main__":
    main()
