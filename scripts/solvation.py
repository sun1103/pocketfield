#!/usr/bin/env python3
"""Solvation free energy for analogue comparison.

Two methods:
  1. GBSA-OBC (corrected): Onufriev-Bashford-Case implicit solvent with
     pairwise-descreened effective Born radii and am1bcc charges.
  2. Explicit TIP3P: place water box, minimise, compute E(compound+water)
     − E(water only) − E(compound vacuum).

Usage:
  python scripts/solvation.py --sdf tests/6duk_grow/candidates.sdf
"""

import sys, os, argparse
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from openff.toolkit import Molecule, ForceField
from openmm import unit, app
import openmm as mm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pocketfield.solvation import gbsa_obc


# ═══════════════════════════════════════════════════════════════════════════════
#  Method 2: Explicit TIP3P water
# ═══════════════════════════════════════════════════════════════════════════════

WATER_PADDING = 8.0  # Å padding around solute


def explicit_water_energy(smiles):
    """Compute E(compound+water) − E(water) − E(compound) via OpenMM.

    Uses SMIRNOFF (via openmmforcefields) + TIP3P with short minimisation.
    """
    from openmmforcefields.generators import SMIRNOFFTemplateGenerator

    off_mol = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
    off_mol.assign_partial_charges("am1bcc")
    off_mol.generate_conformers(n_conformers=1)
    off_mol.name = "UNL"

    # Vacuum system (compound only)
    ff_solute = ForceField("openff-2.1.0.offxml")
    top_solute_off = off_mol.to_topology()
    system_vac = ff_solute.create_openmm_system(
        top_solute_off, charge_from_molecules=[off_mol],
        allow_nonintegral_charges=True,
    )

    conf = off_mol.conformers[0]
    pos_solute = unit.Quantity(
        np.array(conf.to("angstrom").magnitude, dtype=np.float64), unit.angstrom,
    )

    # Minimise in vacuum
    e_vac = _minimize_openmm(system_vac, pos_solute)

    # ── Solvated system ──
    top_solute_omm = top_solute_off.to_openmm()
    water_ff = app.ForceField("amber14/tip3p.xml")

    # Register SMIRNOFF template generator so the Amber force field can
    # parameterise the drug-like solute while handling water natively.
    smirnoff = SMIRNOFFTemplateGenerator(
        molecules=[off_mol], forcefield="openff-2.1.0.offxml",
    )
    water_ff.registerTemplateGenerator(smirnoff.generator)

    modeller = app.Modeller(top_solute_omm, pos_solute)
    modeller.addSolvent(
        water_ff, model="tip3p",
        padding=unit.Quantity(WATER_PADDING, unit.angstrom),
        neutralize=False,
    )

    system_solv = water_ff.createSystem(modeller.topology)

    # Replace solute charges with am1bcc values (SMIRNOFFTemplateGenerator may
    # assign its own charges; we want am1bcc for consistency with OBC).
    charges_am1bcc = [float(c.magnitude) for c in off_mol.partial_charges]
    for force in system_solv.getForces():
        if isinstance(force, mm.NonbondedForce):
            for i, q in enumerate(charges_am1bcc):
                _, sigma, epsilon = force.getParticleParameters(i)
                force.setParticleParameters(i, q * unit.elementary_charge, sigma, epsilon)
            break

    # Pad positions if system has virtual sites (SMIRNOFF adds lone-pair sites)
    n_particles = system_solv.getNumParticles()
    n_positions = len(modeller.positions)
    if n_particles > n_positions:
        extra = np.zeros((n_particles - n_positions, 3))
        modeller.positions = unit.Quantity(
            np.vstack([modeller.positions.value_in_unit(unit.angstrom), extra]),
            unit.angstrom,
        )

    e_solv = _minimize_openmm(system_solv, modeller.positions)

    # ── Water-only reference ──
    # Build topology and positions from HOH residues only (skip solute + counterions)
    water_indices = []
    for res in modeller.topology.residues():
        if res.name == "HOH":
            for atom in res.atoms():
                water_indices.append(atom.index)
    n_water = len(water_indices)

    water_top = app.Topology()
    water_chain = water_top.addChain()
    for i in range(n_water // 3):
        res = water_top.addResidue("HOH", water_chain)
        o = water_top.addAtom("O", app.Element.getBySymbol("O"), res)
        h1 = water_top.addAtom("H1", app.Element.getBySymbol("H"), res)
        h2 = water_top.addAtom("H2", app.Element.getBySymbol("H"), res)
        water_top.addBond(o, h1)
        water_top.addBond(o, h2)

    water_positions = unit.Quantity(
        np.array([list(modeller.positions[idx].value_in_unit(unit.angstrom))
                  for idx in water_indices]),
        unit.angstrom,
    )
    system_water = water_ff.createSystem(water_top)
    e_water = _minimize_openmm(system_water, water_positions)

    dG_solv = e_solv - e_water - e_vac
    return dG_solv, e_solv, e_water, e_vac


def _minimize_openmm(system, positions):
    """Minimise and return potential energy in kcal/mol."""
    integrator = mm.VerletIntegrator(0.001)
    platform = mm.Platform.getPlatformByName("Reference")
    topology = app.Topology()
    simulation = app.Simulation(topology, system, integrator, platform)
    context = simulation.context
    context.setPositions(positions)
    mm.LocalEnergyMinimizer.minimize(context, 10.0, 1000) #10 kJ/mol, 1000 steps
    state = context.getState(getEnergy=True)
    return state.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Solvation free energy: OBC + explicit TIP3P")
    p.add_argument("--sdf", required=True, help="Candidate SDF file")
    p.add_argument("--skip-explicit", action="store_true",
                   help="Skip explicit water (slow)")
    args = p.parse_args()

    supp = Chem.SDMolSupplier(args.sdf, sanitize=False)
    results = []

    for i, mol in enumerate(supp):
        if mol is None:
            continue
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            pass
        smi = Chem.MolToSmiles(mol)

        # ── Common prep: am1bcc charges + MMFF94 vacuum geometry ──
        off_mol = Molecule.from_smiles(smi, allow_undefined_stereo=True)
        off_mol.assign_partial_charges("am1bcc")
        charges = [float(c.magnitude) for c in off_mol.partial_charges]

        mol2 = Chem.AddHs(Chem.MolFromSmiles(smi))
        AllChem.EmbedMolecule(mol2, randomSeed=42)
        try:
            AllChem.MMFFOptimizeMolecule(mol2)
        except Exception:
            pass
        conf = mol2.GetConformer()
        positions = np.array([list(conf.GetAtomPosition(a))
                              for a in range(mol2.GetNumAtoms())])
        atomic_nums = [a.GetAtomicNum() for a in mol2.GetAtoms()]

        # ── Method 1: OBC-GBSA ──
        dG_obc, dG_el, dG_np = gbsa_obc(positions, atomic_nums, charges)

        result = {
            "candidate": i + 1,
            "smiles": smi[:80],
            "charge": Chem.GetFormalCharge(mol),
            "heavy_atoms": mol.GetNumHeavyAtoms(),
            "dG_obc": round(dG_obc, 2),
            "dG_el": round(dG_el, 2),
            "dG_np": round(dG_np, 2),
        }

        # ── Method 2: Explicit TIP3P ──
        if not args.skip_explicit:
            try:
                dG_explicit, e_solv, e_water, e_vac = explicit_water_energy(smi)
                result["dG_explicit"] = round(dG_explicit, 2)
                result["E_solv"] = round(e_solv, 1)
                result["E_water"] = round(e_water, 1)
                result["E_vac_mm"] = round(e_vac, 1)
                print(f"Candidate {i+1}: "
                      f"OBC={dG_obc:.1f}  "
                      f"TIP3P={dG_explicit:.1f}  "
                      f"(E_solv={e_solv:.0f} − E_water={e_water:.0f} − E_vac={e_vac:.0f}) "
                      f"kcal/mol")
            except Exception as e:
                print(f"Candidate {i+1}: OBC={dG_obc:.1f}, TIP3P FAILED — {e}")
                result["dG_explicit"] = None
        else:
            print(f"Candidate {i+1}: OBC={dG_obc:.1f} kcal/mol "
                  f"(ΔG_el={dG_el:.1f}, ΔG_np={dG_np:.1f})")

        results.append(result)

    if not results:
        print("No results.")
        return

    # Summary table
    header = f"{'Cand':<5} {'OBC-GBSA':>10} {'ΔG_el':>10} {'ΔG_np':>10}"
    if not args.skip_explicit:
        header += f" {'TIP3P':>10} {'ΔΔG_OBC':>10} {'ΔΔG_TIP3P':>10}"
    print(f"\n{header}")
    print("-" * len(header))
    ref_obc = results[0]["dG_obc"]
    ref_tip3p = results[0].get("dG_explicit") or 0
    for r in results:
        line = (f"{r['candidate']:<5} {r['dG_obc']:>10.1f} "
                f"{r['dG_el']:>10.1f} {r['dG_np']:>10.1f}")
        if not args.skip_explicit:
            dg_tip = r.get("dG_explicit")
            tip_str = f"{dg_tip:>10.1f}" if dg_tip is not None else f"{'FAILED':>10}"
            line += f" {tip_str}"
            line += f" {r['dG_obc'] - ref_obc:>+10.1f}"
            if dg_tip is not None and ref_tip3p:
                line += f" {dg_tip - ref_tip3p:>+10.1f}"
        print(line)


if __name__ == "__main__":
    main()
