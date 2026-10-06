# PocketField

Probe-based spherical interaction-energy fields for pocket-conditioned molecule generation. PocketField ranks growth sectors from a 3D chemical-probe field around a protein binding pocket, then enumerates fragments and linkers from mapped anchor dummy atoms and scores the resulting RDKit candidates.

It is a deterministic, non-learned pipeline — an MVP of a CMB-like workflow. Fragment and linker choices are ranked by explicit coefficient-vector matching plus field/clash/solvation scoring.

---

## Overview

```
receptor PDB ─┐
              ├─► build ──► field.npz
ligand PDB ───┘             metadata.json
                            growth_plan.json
                                   │
                                   ▼
                              grow ──► candidates.sdf
                                        candidates.json
                                        sector_coefficients.json
                                        fragment_features.json
                                        dummy_sector_assignments.json
                                   │
                ┌──────────────────┴───────────────────┐
                ▼                                      ▼
       scripts/score_docked.py                scripts/solvation.py
       PocketField fit + OBC-GBSA             OBC-GBSA / explicit TIP3P
       → ranked.sdf (one combined score)      → solvation energies only
```

```
pocketfield build ──► pocketfield grow ──► docked/candidate poses
      │                    │                        │
      ▼                    ▼                        ▼
   field.npz          candidates.sdf         two independent downstream tools:
   metadata.json      candidates.json        • score_docked.py (fit + solvation)
   growth_plan.json   sector/fragment vectors  • solvation.py (solvation only)
```

The two scripts are alternatives, not stages: `score_docked.py` folds OBC-GBSA solvation into a single PocketField ranking, while `solvation.py` reports solvation alone (and is the one that offers explicit TIP3P water).

The field is not a raw conformal ESP map. Each value is an interaction-energy well computed at the original 3D sample point before spherical indexing, which avoids treating the 2D spherical projection as the physical object.

---

## Installation

```bash
python3 -m pip install -e .
```

The core field builder needs only NumPy. Candidate growth requires RDKit; solvation scoring additionally requires the OpenFF Toolkit (and OpenMM for explicit water).

```bash
# Recommended: a conda env with RDKit + OpenFF + OpenMM
conda create -n pocketfield python=3.12
conda activate pocketfield
pip install -e .
pip install rdkit openff-toolkit openmm
```

Optional retrosynthesis validation needs `rdchiral` (see [Filtering](#filtering)).

---

## Quick Start

### 1. Build a field

```bash
pocketfield build \
  --protein receptor.pdb \
  --center 12.4,3.1,-8.0 \
  --out-dir out/pocket_001/field
```

Or use ligand atoms to define the pocket center:

```bash
pocketfield build \
  --protein receptor.pdb \
  --ligand ligand.pdb \
  --out-dir out/pocket_001/field
```

Outputs `field.npz`, `metadata.json`, and `growth_plan.json` into `out/pocket_001/field/`.

### 2. Grow candidates

```bash
python -m pocketfield.cli grow \
  --field out/pocket_001/field/field.npz \
  --plan out/pocket_001/field/growth_plan.json \
  --metadata out/pocket_001/field/metadata.json \
  --out-dir out/pocket_001/grow
```

The default anchor is `C([*:1])([*:2])([*:3])[*:4]`. Dummy atoms mark growth sites. The generator attaches built-in probe-compatible fragments, embeds and optimizes 3D conformers with RDKit, aligns them toward ranked sectors, penalizes pocket clashes, and writes `candidates.sdf`, `candidates.json`, and the sector/fragment feature vectors.

### 3. One-step design

```bash
pocketfield design \
  --protein receptor.pdb \
  --ligand ligand.pdb \
  --out-dir out/pocket_001 \
  --fragment-library examples/fragments.csv
```

Writes `out/pocket_001/field/` and `out/pocket_001/grow/`.

### 4. Inspect growth outputs

```bash
python -m pocketfield.cli inspect \
  --grow-dir out/pocket_001/grow \
  --top 10 \
  --json-out out/pocket_001/grow/inspection.json
```

Read-only (except `--json-out`); audits sector chemistry, dummy assignments, fragment/linker matches, desolvation penalties, and candidate score summaries.

### 5. Score docked poses (PocketField + solvation)

```bash
python scripts/score_docked.py \
  --field out/pocket_001/field/field.npz \
  --metadata out/pocket_001/field/metadata.json \
  --sdf docked_poses.sdf \
  --output ranked.sdf \
  --solvation obc \
  --solvation-weight 0.15
```

Computes a PocketField fit score per molecule plus an OBC-GBSA solvation free energy, then combines them into one ranking. See [Scoring](#scoring).

### 6. Solvation free energy (standalone)

```bash
python scripts/solvation.py --sdf out/pocket_001/grow/candidates.sdf
```

Reports OBC-GBSA and (optionally) explicit TIP3P solvation energies for each molecule.

---

## Important Options

```bash
# build
--shells 2.5,3.5,4.5,6.0
--samples 2048
--sectors 64
--pocket-radius 8.0
--probes hydrophobic,hbond_donor,hbond_acceptor,positive,negative

# grow / design
--anchor-smiles 'c1cc([*:1])ccc1[*:2]'
--anchor-structure examples/anchor_dummy.sdf
--reference-ligand path/to/reference.sdf
--sectors-per-dummy 4
--dummy-angle-cutoff 80
--bridge-anchor-dummies
--linker-library examples/linkers.csv
--fragment-library examples/fragments.csv
--growth-depth 3
--request-limit 5000
--max-heavy-atoms 45   # optional hard cap; off by default (soft size penalty ranks)
--diversity-threshold 0.5   # pre-step fragment-level ECFP4 diversity; 1.0 = off
--max-clash-score 0.5
--max-field-score 0.5
--sector-match-weight 0.25
--drop-2d            # default: grow every fragment (single mode) / every span-fitting linker (bridge), rank by 3D score
--no-drop-2d         # restore 2D vector-match prefilter selection
--cross-attention-weight 0.25
--linker-distance-tolerance 2.0
--linker-conformers 20
--linker-fit-rmsd-tolerance 3.0
--conformers 10
--pose-refine
--pose-refine-iters 60
```

---

## Pipeline Modules

| Module | Role |
|--------|------|
| `pocketfield/cli.py` | Command-line entry point — `build`, `grow`, `design`, `fragments`, `inspect` |
| `pocketfield/atoms.py` | PDB parsing, atom typing, pocket-atom selection by radius |
| `pocketfield/sphere.py` | Fibonacci sphere directions, shell sampling, numeric arg parsing |
| `pocketfield/probes.py` | Chemical probe definitions (hydrophobic, donor, acceptor, ±charge) |
| `pocketfield/field.py` | Spherical interaction-energy field (energy + Cartesian gradient) |
| `pocketfield/sectors.py` | Sector assignment and single/pair sector ranking |
| `pocketfield/plan.py` | Growth plan (ranked single sectors and adjacent pairs) |
| `pocketfield/library.py` | Fragment/linker library definitions, loading, and validation |
| `pocketfield/features.py` | 15-dim sector/fragment/linker vectors, match ranking, reference-occupancy screening |
| `pocketfield/geometry.py` | Anchor placement, exit vectors, dummy-sector assignment, reference occupancy |
| `pocketfield/growth.py` | Growth-request generation (sector cover, bridge linker span prefilter) |
| `pocketfield/assembly.py` | Bonds fragments/linkers onto the anchor through mapped dummies, 3D alignment |
| `pocketfield/rdkit_grow.py` | `pocketfield grow` orchestrator — embedding, scoring, candidate assembly |
| `pocketfield/scoring.py` | Candidate field/clash scoring against the pocket field |
| `pocketfield/attention.py` | Multi-head cross-attention fragment re-ranking (no learned params) |
| `pocketfield/solvation.py` | OBC-GBSA implicit-solvent free energy (pure NumPy) |
| `pocketfield/linker_export.py` | Linker SMILES normalization (`[*:1]`/`[*:2]`) and probe annotation |
| `pocketfield/merge_utils.py` | Fragment assembly + optional rdchiral retrosynthesis validation |
| `pocketfield/io.py` | JSON write helper |

### Scripts

| Script | Role |
|--------|------|
| `scripts/solvation.py` | Solvation free energy CLI — OBC-GBSA + explicit TIP3P water |
| `scripts/score_docked.py` | Score docked poses — PocketField fit + solvation, combined ranking |
| `scripts/demo_attention.py` | Demonstrates the production cross-attention re-ranking |
| `scripts/export_linkers.py` | Exports linker SMILES from an external database to a PocketField CSV |
| `scripts/prepare_arm_fragments.py` | Converts docked ARM fragments (plain SMILES) into a `[*:1]`-capped PocketField fragment library |

---

## Key Concepts

### Pocket field

For each sample point on several spherical shells, PocketField evaluates the interaction energy of chemical probes (hydrophobic, H-bond donor/acceptor, positive, negative — plus aromatic if requested). Energies are computed in 3D first, then indexed by sphere direction and shell radius.

`field.npz` contains:

- `directions`: `(samples, 3)` unit vectors.
- `points`: `(shells, samples, 3)` Cartesian sample points.
- `energies`: `(probes, shells, samples)` interaction energies (kcal/mol-like).
- `gradients`: `(probes, shells, samples, 3)` energy gradients.
- `sector_ids`: `(samples,)` sector assignment per direction.
- `sector_centers`: `(sectors, 3)` unit sector-center directions.

### Sectors

The sphere is partitioned into approximately equal sectors. Each sector gets a best probe, best shell, field score, direction vector, and gradient-derived growth vector. Sectors are not the same as growth sites: when an anchor is spatially known, sectors are clustered around each anchor dummy exit vector.

### Anchor

The anchor is the fixed starting scaffold. It must use mapped RDKit dummy atoms (`[*:1]`, `[*:2]`, `[*:3]`, …). Bare `*` atoms are rejected because they cannot be mapped to spatial dummy positions.

Valid:

```text
c1cc([*:1])ccc1[*:2]
[*:1]c1ccccc1.[*:2]CC
```

Invalid:

```text
*c1ccccc1.*CC
```

`--anchor-structure` (SDF/MOL/PDB) is optional but important when the anchor is already placed in the pocket. Dummy isotope labels `1`, `2`, `3`, … map to `[*:1]`, `[*:2]`, `[*:3]`, …. PocketField clusters and ranks sectors around each dummy exit vector instead of assigning global top sectors blindly.

### Fragments and linkers

Fragments are one-dummy substituents (`[*:1]C`, `[*:1]O`, `[*:1]C(=O)O`). Linkers are two-dummy bridges (`[*:1]CC[*:2]`, `[*:1]C(=O)N[*:2]`) used in bridge mode to connect two anchor dummy sites:

```text
anchor [*:1] -> linker [*:1]
anchor [*:2] -> linker [*:2]
```

Fragment and linker libraries may be `.csv`, `.json`, or `.jsonl`. Only `smiles` is required; optional `name` and `probes` columns are honored. Minimal CSV:

```csv
smiles
[*:1]CCO
[*:1]c1ccccc1
```

Linker CSV:

```csv
smiles
[*:1]CC[*:2]
[*:1]C(=O)N[*:2]
```

### Bridge mode

Use bridge mode when disconnected anchor components should be connected through an enumerated linker:

```bash
pocketfield design \
  --protein receptor.pdb \
  --ligand ligand.pdb \
  --anchor-smiles '[*:1]c1ccccc1.[*:2]CC' \
  --bridge-anchor-dummies \
  --linker-library examples/linkers.csv
```

In bridge mode PocketField uses true two-dummy linkers, bonds anchor site `[*:1]` to the linker `[*:1]` neighbor and `[*:2]` to the linker `[*:2]` neighbor, then removes all dummy atoms. Candidate records use `mode: bridge` and produce connected SMILES.

When `--anchor-structure` is supplied, the two anchor components are held at their docked coordinates while the linker is fit between them. The anchor atoms are kept fixed during a constrained UFF minimization, so the linker's torsions relax to close the gap rather than the anchor moving off its docked pose. A candidate whose linker attachment atoms land farther than `--linker-fit-rmsd-tolerance` (default 3.0 Å) from their anchor dummy targets is rejected as `poor_linker_closure`; the residual is recorded in each candidate's `linker_fit_rmsd`.

### Vector matching

PocketField computes a 15-dim coefficient vector for each sector and a matching feature vector for each fragment/linker:

```text
[0] hydrophobic       [1] hbond_donor   [2] hbond_acceptor
[3] positive          [4] negative      [5] aromatic
[6] steric_openness   [7] small_size    [8] medium_size
[9] large_size        [10] rigidity     [11] polarity
[12] depth            [13] water_displacement  [14] buried_polar_support
```

Sector coefficients come from probe-well strength, best-shell depth, steric openness, polarity/rigidity preference, and burial-dependent water-displacement and buried-polar support. Fragment/linker features come from atom types, donor/acceptor counts, formal charge, aromaticity, heavy-atom count, bond-rigidity proxy, and probe tags.

The match score is:

```text
sector_match_score =
  -dot(sector_vector, candidate_vector)
  + mismatch_penalty
  + unsupported_polar_desolvation_penalty
```

Lower is better. The desolvation penalty is applied **outside** the dot product so a desolvation risk penalizes unsupported buried polar atoms rather than rewarding polar fragments.

### Cross-attention re-ranking

When `--cross-attention-weight > 0`, fragment rankings are refined by multi-head cross-attention over nearby sector feature vectors — a fragment bound in one sub-pocket is influenced by what sits in neighbouring sub-pockets. There are no learned parameters.

| Mechanism | Role |
|-----------|------|
| Co-assignment exclusion | Sectors bound by the same fragment (shared dummy) are self, not neighbours |
| Sparse attention | Only top-k spatially closest active sectors matter |
| Multi-head attention | Pharm (complementarity), shape (similarity), polar (similarity) heads |
| Charge modulation | Charged/polar fragments are more sensitive to neighbour environment |
| Confidence gating | Neighbours agree → amplified; disagree → dampened |
| Self-check | Fragment similarity to its own sector reinforces the base score |
| Mean-centering | Approximately zero-sum re-ranking per sector; above-mean → bonus, below-mean → penalty |

The base score uses **similarity** (the fragment should be what the pocket wants), while neighbour attention uses **complementarity** (a neighbour that wants X will itself be X, so my fragment should complement X).

### Scoring

PocketField scores are normalized to `[0, 1]` (lower is better):

```text
per_atom:   norm = (E_raw - E_min) / 50        # clip [0,1], 50 kcal/mol cap
field_score = mean(norm over heavy atoms)

clash_score = min(1.0, Σ overlap² / (n_heavy × 3.0 Å²))

pf_score = 0.35 × field_score + 0.30 × clash_score + 0.15 × (n_heavy / 70)
```

`grow` ranks candidates by `pf_score` alone. The sector-match score is used only to *select* which fragment fills each sector, not in the final candidate ranking.

### OBC-GBSA solvation

`pocketfield/solvation.py` implements Onufriev-Bashford-Case generalized-Born with pairwise-descreened Born radii and a Shrake-Rupley nonpolar surface term:

```text
ΔG_el = -½ (1/ε_in - 1/ε_out) × 332 × ΣᵢΣⱼ qᵢqⱼ / f_GB
f_GB  = √(r² + RᵢRⱼ exp(-r² / 4RᵢRⱼ))

Born radii Rᵢ via HCT pairwise descreening (no bond exclusion)
ΔG_np = 0.005 × SASA           (Shrake-Rupley)
ΔG_solv = ΔG_el + ΔG_np
```

A more negative `ΔG_solv` means more soluble, hence harder to desolvate into the pocket.

### Combined ranking

`score_docked.py` folds the solvation penalty into the PocketField score:

```text
solv_penalty = (ΔG_max - ΔG) / (ΔG_max - ΔG_min)     # [0,1], most soluble → 1
combined = pf_score + w × solv_penalty
```

Default `w = 0.15` — PocketField dominates and solvation serves as a tiebreaker.

---

## Data Files

| Path | Description |
|------|-------------|
| `examples/tiny_receptor.pdb`, `examples/tiny_ligand.pdb` | Minimal smoke-test inputs |
| `examples/anchor_dummy.sdf` | Example 3D anchor with positioned dummy atoms |
| `examples/fragments.csv`, `examples/linkers.csv` | Built-in fragment/linker libraries |
| `examples/fragments_smiles_only.csv`, `examples/linkers_smiles_only.csv` | Minimal SMILES-only libraries |

`tests/6duk_*` (the EGFR reference protein, precomputed field, and grow outputs) are gitignored — they are large, per-experiment reference data supplied separately.

---

## Output Structure

```
out/pocket_001/
├── field/                            # `build` output (also written by `design`)
│   ├── field.npz                     # directions, shells, points, energies, gradients, sectors
│   ├── metadata.json                 # center, pocket atoms, probes, shell/sector settings
│   └── growth_plan.json              # ranked single sectors + adjacent sector pairs
└── grow/                             # `grow` output (also written by `design`)
    ├── candidates.sdf                # ranked 3D molecules with score properties
    ├── candidates.json               # score table, fragments, sector ids, SMILES
    ├── sector_coefficients.json      # per-sector 15-dim coefficient vectors
    ├── fragment_features.json        # per-fragment 15-dim feature vectors
    ├── linker_features.json          # per-linker 15-dim feature vectors (bridge mode only)
    ├── dummy_sector_assignments.json # dummy map -> nearby sectors
    ├── sector_stats.json             # per-sector score mean/std
    ├── reference_screening.json      # fragments vs reference-occupied sectors (only with --reference-ligand)
    └── reference_sector_coefficients.json  # 15-dim vectors for reference-occupied sectors (only with --reference-ligand)
```

---

## Filtering

Important filters:

```bash
--max-heavy-atoms 45   # optional; off by default — size is ranked via the soft penalty
--diversity-threshold 0.5   # pre-step fragment-level ECFP4 diversity; 1.0 = off
--max-clash-score 0.5
--max-field-score 0.5
```

Optional retrosynthesis validation:

```bash
pocketfield design \
  ... \
  --retro-rules path/to/uspto.templates.classified.json.gz
```

This uses `RetrosynthesisValidator` from the bundled `pocketfield/merge_utils.py` (override with `--retro-script`). PocketField marks newly formed anchor-fragment or anchor-linker bonds with isotope pair `88/99`, calls the validator, then strips isotopes before writing outputs.

Scores are normalised to `[0, 1]`, so to disable the field/clash filters set them to `1.0` (`--max-clash-score 1.0 --max-field-score 1.0`); for real pockets, keep them strict.

### Diversity selection

`--diversity-threshold` (default `1.0` = off) is a pre-step diversity filter in the spirit of STELLA's clustering-based selection: instead of spending 3D embedding on every enumerated candidate, PocketField computes an ECFP4 (radius 2) fingerprint on each *grown fragment/linker* (the variable substituent), not the whole molecule — the anchor is constant across candidates and would otherwise swamp the similarity signal. A candidate is dropped when its substituent *pattern* is similar to an already-kept candidate's at **every anchor map** (per-map Tanimoto > threshold); the same fragment on a different dummy stays distinct. This runs *before* `AddHs`/`EmbedMultipleConfs`/UFF, so the expensive 3D + scoring work is only paid for structurally diverse representatives. Requests are first sorted by their cheap sector-match score, so within each cluster the most promising member is the one that gets embedded. The cutoff is a max-Tanimoto on the substituent fingerprint — lower collapses more aggressively (e.g. `[*:1]CC` vs `[*:1]CCC` ≈ 0.56, while `[*:1]C` vs `[*:1]O` ≈ 0.2). A value around `0.5` collapses simple chain-length analogs but keeps distinct functional groups; `1.0` disables the filter. Tune it against the fragment library you want to treat as interchangeable.

### 2D vector-match prefilter (dropped)

`--drop-2d` (default on; `--no-drop-2d` restores the old path) removes the 15-dim vector-match score from candidate *selection*. The match was never part of the final ranking (`score = 0.35·field + 0.30·clash + 0.15·n_heavy/70`); it only pre-filtered which fragments/linkers got embedded. In both growth modes that prefilter was anti-predictive of the 3D field fit, so with it dropped `grow` enumerates every fragment (single mode) or every span-fitting linker (bridge mode — the geometric span check is feasibility, not chemistry, and is kept) and lets the 3D score rank the whole library. Pair/multi growth is skipped under `--drop-2d`, since enumerating all fragment combinations is intractable without a prefilter.

---

## Current Scope

PocketField builds the field, sector plan, and first-pass RDKit candidates with field/clash/vector scoring plus heuristic desolvation. It is not yet a production de novo design engine. Current limitations:

- No external docking (in-house rigid-body pose refinement is opt-in via `--pose-refine`).
- Desolvation inside `grow` is a heuristic coefficient/penalty; explicit thermodynamics is only in the external `scripts/` solvation stage.
- Retrosynthesis validation is optional and does not replace a full synthetic-accessibility model.

---

## License

[MIT](LICENSE)
