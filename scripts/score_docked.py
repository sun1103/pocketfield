#!/usr/bin/env python3
"""Combine PocketField bound-state scoring with solvation free energy.

For each docked pose in an SDF, computes:
  1. PocketField score — how well the molecule fits the pocket field
     (field_score, clash_score, combined score)
  2. Solvation free energy — OBC-GBSA implicit solvent (am1bcc charges,
     MMFF94 geometry) or explicit TIP3P

Outputs a ranking table and an annotated SDF with scores in SD tags.

Usage:
  python scripts/score_docked.py \
      --field tests/6duk_field/field.npz \
      --metadata tests/6duk_field/metadata.json \
      --sdf tests/6duk_grow/candidates.sdf \
      --output ranked.sdf
"""

import sys, os, argparse, json
import numpy as np

# Ensure AmberTools executables (antechamber, sqm) are on PATH for am1bcc.
# conda-forge ambertools installs them under the env's bin/ but they aren't
# visible unless the env is activated or PATH is set explicitly.
_conda_prefix = os.path.dirname(os.path.dirname(sys.executable))
_conda_bin = os.path.join(_conda_prefix, "bin")
if os.path.isdir(_conda_bin) and _conda_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _conda_bin + os.pathsep + os.environ.get("PATH", "")

# ── Paths ────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rdkit import Chem
from rdkit.Chem import AllChem
from openff.toolkit import Molecule, ForceField as OFF_ForceField


# ═══════════════════════════════════════════════════════════════════════════════
#  PocketField scoring (imported from rdkit_grow)
# ═══════════════════════════════════════════════════════════════════════════════

from pocketfield.rdkit_grow import _score_candidate


def _load_field(field_path, metadata_path):
    """Load field data and compute per-(probe, shell) min/max for normalisation."""
    field = np.load(field_path)
    with open(metadata_path) as f:
        metadata = json.load(f)

    energies = field["energies"]  # (n_probes, n_shells, n_directions)
    field_mins = energies.min(axis=2)  # (n_probes, n_shells)
    field_ranges = np.full_like(field_mins, 50.0)  # fixed cap: 50 kcal/mol above min

    return {
        "center": field["center"],
        "shells": field["shells"],
        "directions": field["directions"],
        "energies": energies,
        "probe_names": [str(n) for n in field["probe_names"]],
        "pocket_atoms": metadata["pocket_atoms"],
        "field_mins": field_mins,
        "field_ranges": field_ranges,
    }


def pocketfield_score(mol, field_data):
    """Compute PocketField score for one molecule. Returns dict of scores."""
    return _score_candidate(
        mol,
        center=field_data["center"],
        shells=field_data["shells"],
        directions=field_data["directions"],
        energies=field_data["energies"],
        probe_names=field_data["probe_names"],
        pocket_atoms=field_data["pocket_atoms"],
        field_mins=field_data["field_mins"],
        field_ranges=field_data["field_ranges"],
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Solvation free energy: OBC-GBSA (imported from pocketfield.solvation)
# ═══════════════════════════════════════════════════════════════════════════════

from pocketfield.solvation import gbsa_obc


def solvation_obc(smiles):
    """Compute OBC-GBSA solvation free energy for a SMILES string."""
    off_mol = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
    off_mol.assign_partial_charges("am1bcc")
    charges = [float(c.magnitude) for c in off_mol.partial_charges]

    mol2 = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol2, randomSeed=42)
    try:
        AllChem.MMFFOptimizeMolecule(mol2)
    except Exception:
        pass
    conf = mol2.GetConformer()
    positions = np.array([list(conf.GetAtomPosition(a))
                          for a in range(mol2.GetNumAtoms())])
    atomic_nums = [a.GetAtomicNum() for a in mol2.GetAtoms()]

    dG, dG_el, dG_np = gbsa_obc(positions, atomic_nums, charges)
    return dG


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="Score docked poses: PocketField + solvation")
    p.add_argument("--field", required=True, help="field.npz")
    p.add_argument("--metadata", required=True, help="metadata.json")
    p.add_argument("--sdf", required=True, help="Docked poses SDF")
    p.add_argument("--output", "-o", default="ranked.sdf",
                   help="Annotated output SDF")
    p.add_argument("--solvation", choices=["obc", "explicit"], default="obc",
                   help="Solvation method (default: obc)")
    p.add_argument("--solvation-weight", "-w", type=float, default=0.15,
                   help="Weight of solvation penalty in combined score "
                        "(default: 0.15)")
    p.add_argument("--skip-solvation", action="store_true",
                   help="Skip solvation, PocketField only")
    args = p.parse_args()

    # ── Load field ──
    print(f"Loading field from {args.field}")
    field_data = _load_field(args.field, args.metadata)
    print(f"  {len(field_data['probe_names'])} probes, "
          f"{field_data['shells'].shape[0]} shells, "
          f"{field_data['directions'].shape[0]} directions")
    print(f"  {len(field_data['pocket_atoms'])} pocket atoms")
    center = field_data["center"]
    print(f"  Center: ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})")

    # ── Load molecules ──
    supp = Chem.SDMolSupplier(args.sdf, sanitize=False)
    mols = []
    for mol in supp:
        if mol is None:
            continue
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            pass
        if mol.GetNumConformers() == 0:
            print("WARNING: molecule has no conformer, skipping")
            continue
        mols.append(mol)

    print(f"\nLoaded {len(mols)} molecules from {args.sdf}")

    # ── Score each molecule ──
    results = []
    for i, mol in enumerate(mols):
        smi = Chem.MolToSmiles(mol)
        heavy = mol.GetNumHeavyAtoms()

        # PocketField score
        pf = pocketfield_score(mol, field_data)

        result = {
            "idx": i + 1,
            "smi": smi,
            "heavy": heavy,
            "field_score": pf["field_score"],
            "clash_score": pf["clash_score"],
            "pf_score": pf["score"],
        }

        # Solvation
        if not args.skip_solvation:
            try:
                dG_solv = solvation_obc(smi)
                result["dG_solv"] = dG_solv
            except Exception as e:
                print(f"  Candidate {i+1}: solvation failed — {e}")
                result["dG_solv"] = None

        results.append(result)
        status = (f"  Candidate {i+1}: pf={pf['score']:.3f} "
                  f"(field={pf['field_score']:.3f}, clash={pf['clash_score']:.3f})")
        if result["dG_solv"] is not None:
            status += f", dG_solv={result['dG_solv']:.1f}"
        print(status)

    # ── Combined ranking ──
    dg_vals = [r["dG_solv"] for r in results if r.get("dG_solv") is not None]
    if dg_vals and not args.skip_solvation:
        dg_min, dg_max = min(dg_vals), max(dg_vals)
        for r in results:
            if r.get("dG_solv") is not None:
                if dg_max > dg_min:
                    # 0 = least soluble (best for binding), 1 = most soluble
                    solv_penalty = (dg_max - r["dG_solv"]) / (dg_max - dg_min)
                else:
                    solv_penalty = 0.0
                r["solv_penalty"] = solv_penalty
                r["score_combined"] = (r["pf_score"] +
                                       args.solvation_weight * solv_penalty)
            else:
                r["solv_penalty"] = None
                r["score_combined"] = r["pf_score"]
    else:
        for r in results:
            r["score_combined"] = r["pf_score"]

    results.sort(key=lambda r: r["score_combined"])

    # ── Print table ──
    header = (f"{'Rank':<5} {'Cand':<6} {'Field':>7} {'Clash':>7} "
              f"{'PF':>7}")
    if not args.skip_solvation:
        header += f" {'dG_solv':>9} {'SolvPen':>8} {'Combined':>9}"
    header += f"  {'Heavy':>5}  SMILES"
    print(f"\n{header}")
    print("-" * len(header))

    for rank, r in enumerate(results, 1):
        smi_short = r["smi"][:60]
        line = (f"{rank:<5} {r['idx']:<6} {r['field_score']:>7.3f} "
                f"{r['clash_score']:>7.3f} {r['pf_score']:>7.3f}")
        if not args.skip_solvation:
            dg_str = f"{r.get('dG_solv'):>9.1f}" if r.get('dG_solv') is not None else f"{'N/A':>9}"
            sp_str = f"{r.get('solv_penalty'):>8.3f}" if r.get('solv_penalty') is not None else f"{'N/A':>8}"
            line += f" {dg_str} {sp_str} {r['score_combined']:>9.4f}"
        line += f" {r['heavy']:>5}  {smi_short}"
        print(line)

    # ── Annotate SDF ──
    writer = Chem.SDWriter(args.output)
    for mol in mols:
        smi = Chem.MolToSmiles(mol)
        for r in results:
            if r["smi"] == smi:  # match by SMILES
                mol.SetProp("PF_field_score", f"{r['field_score']:.4f}")
                mol.SetProp("PF_clash_score", f"{r['clash_score']:.4f}")
                mol.SetProp("PF_score", f"{r['pf_score']:.4f}")
                mol.SetProp("PF_combined", f"{r['score_combined']:.4f}")
                if r.get("dG_solv") is not None:
                    mol.SetProp("dG_solv_obc", f"{r['dG_solv']:.2f}")
                    mol.SetProp("solv_penalty", f"{r.get('solv_penalty', 0):.4f}")
                break
        writer.write(mol)
    writer.close()

    print(f"\nWrote {len(mols)} molecules with scores to {args.output}")
    print(f"Best candidate: #{results[0]['idx']} "
          f"(combined={results[0]['score_combined']:.4f})")


if __name__ == "__main__":
    main()
