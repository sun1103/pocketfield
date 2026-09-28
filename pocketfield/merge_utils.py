"""Merge utilities: assemble fragments, validate 3D, retrosynthesis check.

Adapted from reaction-generation_forlinker.py.
"""

from __future__ import annotations

import gzip
import json
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, rdForceFieldHelpers
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

# rdchiral is a heavy dependency — import it lazily in RetrosynthesisValidator


# ── Fragment assembly ────────────────────────────────────────────────────────

_TYPICAL_VALENCE = {"C": 4, "N": 3, "O": 2}


def _snapshot_arom_bonds(mol):
    """Return set of (i,j) pairs for all AROMATIC bonds in mol."""
    arom = set()
    for b in mol.GetBonds():
        if b.GetBondType() == Chem.BondType.AROMATIC:
            arom.add((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
    return arom


def _restore_arom_bonds(mol, arom_bonds):
    """Restore AROMATIC bond type + atom flags for bonds in arom_bonds."""
    for (i, j) in arom_bonds:
        bond = mol.GetBondBetweenAtoms(i, j)
        if bond is not None and bond.GetBondType() != Chem.BondType.AROMATIC:
            bond.SetBondType(Chem.BondType.AROMATIC)
            mol.GetAtomWithIdx(i).SetIsAromatic(True)
            mol.GetAtomWithIdx(j).SetIsAromatic(True)


def _derive_bond_type(atom1: Chem.Atom, atom2: Chem.Atom) -> Chem.BondType:
    """Derive bond type from anchor atom chemistry.

    Ring-to-ring or mixed ring/acyclic → SINGLE.
    Both acyclic C/N/O → DOUBLE if both have exactly 2 valence available, else SINGLE.
    S/P/other elements → SINGLE (variable oxidation states invisible to fragment-side check).
    """
    if atom1.IsInRing() or atom2.IsInRing():
        return Chem.BondType.SINGLE

    sym1 = atom1.GetSymbol()
    sym2 = atom2.GetSymbol()
    if sym1 not in _TYPICAL_VALENCE or sym2 not in _TYPICAL_VALENCE:
        return Chem.BondType.SINGLE

    # Current total valence (star bond is SINGLE → subtract 1)
    total1 = sum(b.GetBondTypeAsDouble() for b in atom1.GetBonds())
    total2 = sum(b.GetBondTypeAsDouble() for b in atom2.GetBonds())
    avail1 = _TYPICAL_VALENCE[sym1] - (total1 - 1.0)
    avail2 = _TYPICAL_VALENCE[sym2] - (total2 - 1.0)

    # DOUBLE only when both anchors need exactly a double bond to complete
    # their valence (avail == 2): e.g. two sp² carbons each with one = bond.
    if avail1 == 2.0 and avail2 == 2.0:
        return Chem.BondType.DOUBLE
    return Chem.BondType.SINGLE


def assemble_fragments(core_mol: Chem.Mol, frag_mol: Chem.Mol) -> Chem.Mol | None:
    """Merge two *-labeled fragments by bonding their anchor atoms.

    Both molecules must have the same number of [*] dummy atoms (1 or 2).
    Each [*] serves as an attachment point: its sole neighbor is the anchor
    atom that should be bonded to the corresponding anchor on the other side.

    Returns the assembled Mol with [*] atoms removed, or None on failure.
    """
    combined = Chem.CombineMols(core_mol, frag_mol)
    rw = Chem.RWMol(combined)

    core_n = core_mol.GetNumAtoms()
    core_stars = [a for a in rw.GetAtoms()
                  if a.GetSymbol() == '*' and a.GetIdx() < core_n]
    frag_stars = [a for a in rw.GetAtoms()
                  if a.GetSymbol() == '*' and a.GetIdx() >= core_n]

    if len(core_stars) != len(frag_stars) or len(core_stars) == 0:
        return None
    if len(core_stars) > 2:
        return None

    # Match stars by proximity (simplest: canonical order)
    core_anchors = []
    for s in core_stars:
        nbrs = s.GetNeighbors()
        if len(nbrs) != 1:
            return None
        core_anchors.append(nbrs[0].GetIdx())

    frag_anchors = []
    for s in frag_stars:
        nbrs = s.GetNeighbors()
        if len(nbrs) != 1:
            return None
        frag_anchors.append(nbrs[0].GetIdx())

    # Bond anchors pairwise
    for ca, fa in zip(core_anchors, frag_anchors):
        bt = _derive_bond_type(rw.GetAtomWithIdx(ca), rw.GetAtomWithIdx(fa))
        rw.AddBond(ca, fa, bt)
        rw.GetAtomWithIdx(ca).SetIsotope(88)
        rw.GetAtomWithIdx(fa).SetIsotope(99)

    # Remove all [*] atoms (in reverse index order)
    star_indices = sorted(
        [a.GetIdx() for a in rw.GetAtoms() if a.GetSymbol() == '*'],
        reverse=True,
    )
    for idx in star_indices:
        rw.RemoveAtom(idx)

    final = rw.GetMol()
    try:
        final.UpdatePropertyCache(strict=False)
        arom = _snapshot_arom_bonds(final)
        Chem.SanitizeMol(final)
        _restore_arom_bonds(final, arom)
    except Exception:
        return None
    return final


def assemble_at_star(
    core_mol: Chem.Mol, frag_mol: Chem.Mol, core_star_idx: int,
) -> Chem.Mol | None:
    """Bond one specific core [*] to a single-star fragment.

    The core may have multiple [*] atoms; only the one at core_star_idx
    is consumed. The fragment must have exactly one [*]. Other [*] atoms
    in the core remain intact for subsequent assembly steps.

    Returns the assembled Mol (N-1 stars remaining), or None on failure.
    """
    combined = Chem.CombineMols(core_mol, frag_mol)
    rw = Chem.RWMol(combined)

    core_n = core_mol.GetNumAtoms()
    core_star = None
    for a in rw.GetAtoms():
        if a.GetSymbol() == "*" and a.GetIdx() == core_star_idx:
            core_star = a
            break
    if core_star is None:
        return None

    frag_stars = [a for a in rw.GetAtoms()
                  if a.GetSymbol() == "*" and a.GetIdx() >= core_n]
    if len(frag_stars) != 1:
        return None

    # Core star anchor
    core_nbrs = core_star.GetNeighbors()
    if len(core_nbrs) != 1:
        return None
    core_anchor = core_nbrs[0].GetIdx()

    # Fragment star anchor
    frag_nbrs = frag_stars[0].GetNeighbors()
    if len(frag_nbrs) != 1:
        return None
    frag_anchor = frag_nbrs[0].GetIdx()

    bt = _derive_bond_type(rw.GetAtomWithIdx(core_anchor),
                           rw.GetAtomWithIdx(frag_anchor))
    rw.AddBond(core_anchor, frag_anchor, bt)
    rw.GetAtomWithIdx(core_anchor).SetIsotope(88)
    rw.GetAtomWithIdx(frag_anchor).SetIsotope(99)

    # Remove only the two consumed stars
    for idx in sorted([core_star_idx, frag_stars[0].GetIdx()], reverse=True):
        rw.RemoveAtom(idx)

    final = rw.GetMol()
    try:
        final.UpdatePropertyCache(strict=False)
        arom = _snapshot_arom_bonds(final)
        Chem.SanitizeMol(final)
        _restore_arom_bonds(final, arom)
    except Exception:
        return None
    return final


# ── 3D validation ────────────────────────────────────────────────────────────

def check_3d_clash(mol: Chem.Mol, energy_cutoff: float = 500.0) -> tuple[bool, float]:
    """ETKDGv3 conformer generation + MMFF optimization.

    If energy < cutoff, the 3D coordinates are written back to mol (in-place).
    Returns (is_valid, energy_kcal).
    """
    try:
        mol_3d = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol_3d, AllChem.ETKDGv3()) < 0:
            return False, 1e6
        # UFF — avoids MMFF94 segfaults on sulfonamide S / charged N
        # mp = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol_3d)
        # ff = None
        # if mp is not None:
        #     ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(mol_3d, mp)
        # if ff is None:
        #     ff = AllChem.UFFGetMoleculeForceField(mol_3d)
        ff = AllChem.UFFGetMoleculeForceField(mol_3d)
        if ff is None:
            return False, 1e6
        energy = ff.CalcEnergy()
        if energy < energy_cutoff:
            mol_3d = Chem.RemoveHs(mol_3d)
            mol.RemoveAllConformers()
            mol.AddConformer(mol_3d.GetConformer(), assignId=True)
            return True, energy
        return False, energy
    except Exception:
        return False, 1e6


# ── Protein-ligand clash ──────────────────────────────────────────────────────

BACKBONE_NAMES = {"N", "CA", "C", "O"}
SIDECHAIN_WEIGHT = 0.4


def check_protein_clash(
    mol: Chem.Mol,
    protein_atoms: list,
    clash_cutoff: float = 2.5,
    conf_id: int = 0,
) -> tuple[int, float, float]:
    """Check steric clashes between mol and protein atoms.

    The mol must already be in the protein coordinate frame (e.g., after
    scaffold alignment to a co-crystallized target ligand).

    Clash scoring uses fractional overlap depth (not binary contact count)
    and discounts side-chain clashes (0.4x) vs backbone (1.0x) since the
    protein is treated as rigid.

    Returns (n_clashes, min_distance, clash_score).
    clash_score ∈ [0, 1]; 1.0 = no clashes.
    """
    conf = mol.GetConformer(conf_id)
    if conf is None:
        return 0, float("inf"), 1.0

    vdw = {
        "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47,
        "S": 1.80, "P": 1.80, "Cl": 1.75, "Br": 1.85, "I": 1.98,
    }

    clashes = 0
    penalty = 0.0
    min_dist = float("inf")

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        lig_el = atom.GetSymbol()
        lig_vdw = vdw.get(lig_el, 1.70)
        lig_pos = conf.GetAtomPosition(atom.GetIdx())

        for p in protein_atoms:
            p_vdw = vdw.get(p.element, 1.70)
            threshold = (lig_vdw + p_vdw) * 0.8

            dx = lig_pos.x - p.x
            dy = lig_pos.y - p.y
            dz = lig_pos.z - p.z
            dist = (dx * dx + dy * dy + dz * dz) ** 0.5

            min_dist = min(min_dist, dist)
            if dist < threshold:
                clashes += 1
                overlap = (threshold - dist) / threshold
                weight = SIDECHAIN_WEIGHT if p.name not in BACKBONE_NAMES else 1.0
                penalty += overlap * weight

    clash_score = 1.0 / (1.0 + penalty)
    return clashes, min_dist, clash_score


def compute_sa_score(mol: Chem.Mol) -> float:
    """Synthetic accessibility score (1=easy, ~10=hard).

    Uses RDKit contrib SA_Score (Ertl & Schuffenhauer 2009).
    """
    from rdkit.Contrib.SA_Score import sascorer  # lazy import

    return sascorer.calculateScore(mol)


# ── Retrosynthesis validation ────────────────────────────────────────────────

class RetrosynthesisValidator:
    """Validate that formed bonds can be cut by known reaction rules."""

    def __init__(self, rules_path: str | list[str], fp_size: int = 2048):
        # rules_path may be a single path or a list of paths. Multiple libraries
        # are merged (e.g. USPTO + a bioisostere-specific library), so one file
        # can extend another's coverage without editing it.
        self.fp_size = fp_size
        self.rules_path = rules_path
        self._rules: list[tuple] | None = None
        self._cache: dict[str, bool] = {}

    def _lazy_load_rules(self):
        if self._rules is not None:
            return
        from rdchiral.main import rdchiralReaction  # lazy import

        paths = self.rules_path
        if isinstance(paths, (str, bytes)):
            paths = [paths]

        self._rules = []
        for path in paths:
            with gzip.open(path, 'rt', encoding='utf-8') as f:
                rules_dict = json.load(f)
            for _, smarts_list in rules_dict.items():
                for smarts in smarts_list:
                    rxn = rdchiralReaction(smarts)
                    lhs = smarts.split('>>')[0]
                    pat = Chem.MolFromSmarts(lhs)
                    if pat:
                        fp = Chem.PatternFingerprint(pat, fpSize=self.fp_size)
                        self._rules.append((rxn, fp))

    def validate(self, mol: Chem.Mol, isotope_pair: tuple[int, int] = (88, 99)) -> bool:
        """Check if bonds marked with the given isotope pair are synthesizable.

        Bonds formed during assembly are tagged with isotopes 88 and 99 on
        the anchor atoms. Each formed bond is validated independently to
        avoid rdchiral map-number scrambling with multi-bond assemblies.

        Results are cached by isotope-free canonical SMILES — different
        junction hypotheses often produce identical merged structures.
        """
        self._lazy_load_rules()
        from rdchiral.main import rdchiralReactants, rdchiralRun  # lazy import

        # Cache key: canonical SMILES with all isotopes stripped.
        # Different junction hypotheses may yield the same underlying molecule.
        mol_stripped = Chem.Mol(mol)
        for a in mol_stripped.GetAtoms():
            a.SetIsotope(0)
        cache_key = Chem.MolToSmiles(mol_stripped)
        if cache_key in self._cache:
            return self._cache[cache_key]

        iso_a, iso_b = isotope_pair

        # Collect formed bonds (isotope 88–99 pairs)
        formed_bonds: list[tuple[int, int]] = []
        for bond in mol.GetBonds():
            u, v = bond.GetBeginAtom(), bond.GetEndAtom()
            if {u.GetIsotope(), v.GetIsotope()} == {iso_a, iso_b}:
                formed_bonds.append((u.GetIdx(), v.GetIdx()))

        if not formed_bonds:
            self._cache[cache_key] = False
            return False  # no formed bonds marked — can't validate

        mol_fp = Chem.PatternFingerprint(mol, fpSize=self.fp_size)

        for u_idx, v_idx in formed_bonds:
            # Isolate this bond with the isotope pair only: zero every isotope/
            # map, then re-tag just this bond's anchors. Isotopes (not atom-map
            # numbers) are used as the tracer because rdchiral RENUMBERS atom
            # maps in its outcomes — our 1/2 would be clobbered — but it
            # preserves isotope labels through the transform. Zeroing the other
            # anchors keeps multi-bond assemblies unambiguous.
            mol_copy = Chem.Mol(mol)
            for a in mol_copy.GetAtoms():
                a.SetAtomMapNum(0)
                a.SetIsotope(0)
            mol_copy.GetAtomWithIdx(u_idx).SetIsotope(iso_a)
            mol_copy.GetAtomWithIdx(v_idx).SetIsotope(iso_b)

            labeled_smi = Chem.MolToSmiles(mol_copy)
            try:
                reactants = rdchiralReactants(labeled_smi)
            except Exception:
                self._cache[cache_key] = False
                return False

            bond_is_legal = False
            for rxn, r_fp in self._rules:
                if not DataStructs.AllProbeBitsMatch(r_fp, mol_fp):
                    continue
                try:
                    _, mapped_outcomes = rdchiralRun(rxn, reactants, return_mapped=True)
                    for _, (combined_smi, _) in mapped_outcomes.items():
                        for comp_smi in combined_smi.split('.'):
                            comp_mol = Chem.MolFromSmiles(comp_smi)
                            if not comp_mol:
                                continue
                            comp_isos = {a.GetIsotope() for a in comp_mol.GetAtoms()
                                         if a.GetIsotope() in (iso_a, iso_b)}
                            # Bond is cut if the two anchors' isotopes land in
                            # different components (this component has one, not both).
                            if (iso_a in comp_isos) != (iso_b in comp_isos):
                                bond_is_legal = True
                                break
                        if bond_is_legal:
                            break
                except Exception:
                    continue
                if bond_is_legal:
                    break

            if not bond_is_legal:
                self._cache[cache_key] = False
                return False

        self._cache[cache_key] = True
        return True

    def get_failing_motifs(
        self, mol: Chem.Mol,
        isotope_pair: tuple[int, int] = (88, 99),
        radius: int = 2,
        fp_size: int = 2048,
    ) -> list:
        """Return Morgan fingerprints around fragment-side anchor atoms.

        When retrosynthesis fails, the local environment around each formed
        bond has no known synthetic precedent. This method fingerprints the
        fragment-side anchor (isotope 99) neighborhood on the assembled mol.

        Returns list of RDKit ExplicitBitVect, empty if no anchors found.
        """
        gen = GetMorganGenerator(radius=radius, fpSize=fp_size)
        fps = []
        iso_frag = isotope_pair[1]  # 99 = fragment side
        for a in mol.GetAtoms():
            if a.GetIsotope() == iso_frag:
                fp = gen.GetFingerprint(mol, fromAtoms=[a.GetIdx()])
                fps.append(fp)
        return fps
