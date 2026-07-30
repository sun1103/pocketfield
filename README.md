# PocketField

PocketField builds probe-based spherical interaction-energy maps for a protein binding pocket. It is an MVP implementation of a CMB-like workflow for molecule generation:

1. Define a pocket center from coordinates or ligand atoms.
2. Sample several spherical shells around that center.
3. Evaluate chemical-probe interaction energies directly in 3D.
4. Store energy gradients and sector-level growth scores.
5. Use the ranked sectors as conditions for anchor-based fragment growing.

The field is not a raw conformal ESP map. Each value is an interaction energy well computed at the original 3D sample point before spherical indexing, which avoids making the 2D spherical projection the physical object.

For the detailed end-to-end workflow, data formats, anchor conventions, bridge mode, and scoring details, see [docs/pipeline.md](docs/pipeline.md).

## Install

```bash
python3 -m pip install -e .
```

## Build A Field

```bash
pocketfield build \
  --protein receptor.pdb \
  --center 12.4,3.1,-8.0 \
  --out-dir out/pocket_001
```

Or use ligand atoms to define the center:

```bash
pocketfield build \
  --protein receptor.pdb \
  --ligand ligand.pdb \
  --out-dir out/pocket_001
```

Outputs:

- `field.npz`: sample directions, shell radii, Cartesian points, energies, gradients, sector assignments.
- `metadata.json`: center, pocket atoms, probes, shell/sector settings.
- `growth_plan.json`: ranked single sectors and adjacent sector pairs for anchor-based growing.

## Important Options

```bash
--shells 2.5,3.5,4.5,6.0
--samples 2048
--sectors 64
--pocket-radius 8.0
--probes hydrophobic,hbond_donor,hbond_acceptor,positive,negative
```

## Data Format

`field.npz` contains:

- `directions`: `(samples, 3)` unit vectors.
- `points`: `(shells, samples, 3)` Cartesian sample points.
- `energies`: `(probes, shells, samples)` interaction energies in kcal/mol-like units.
- `gradients`: `(probes, shells, samples, 3)` energy gradients with respect to Cartesian position.
- `sector_ids`: `(samples,)` sector assignment per direction.
- `sector_centers`: `(sectors, 3)` unit sector center directions.

`growth_plan.json` ranks sectors by favorable wells. Lower scores are better.

## Grow RDKit Candidates

Run `grow` in an environment with RDKit installed:

```bash
/opt/anaconda3/envs/crem/bin/python -m pocketfield.cli grow \
  --field out/pocket_001/field.npz \
  --plan out/pocket_001/growth_plan.json \
  --metadata out/pocket_001/metadata.json \
  --out-dir out/pocket_001/grow
```

The default anchor is `C([*:1])([*:2])([*:3])[*:4]`. Dummy atoms mark sector growth points. The generator attaches built-in probe-compatible fragments, embeds and optimizes 3D conformers with RDKit, aligns them toward ranked sectors, penalizes pocket clashes, and writes:

- `candidates.sdf`: ranked 3D molecules with score properties.
- `candidates.json`: score table, fragments, sector ids, and SMILES.
- `sector_coefficients.json`: per-sector coefficient vectors derived from probe wells, depth, openness, and polarity.
- `fragment_features.json`: per-fragment feature vectors derived from RDKit atom/bond features and probe tags.
- `dummy_sector_assignments.json`: sectors clustered around each positioned anchor dummy when `--anchor-structure` is provided.

Fragment selection is deterministic. For each sector, PocketField computes:

```text
sector_match_score =
  -dot(sector_coefficients, fragment_features)
  + mismatch_penalty
  + unsupported_polar_desolvation_penalty
```

Lower sector-match scores are better. Sector vectors include burial-aware water-displacement and buried-polar-support channels; unsupported polar burial is penalized outside the dot product so desolvation risk is not accidentally rewarded. The final candidate rank combines field score, clash score, and the sector-fragment match score.

If the anchor fragment is already spatially placed in the pocket, provide a 3D anchor structure with positioned dummy atoms:

```bash
--anchor-smiles 'C([*:1])([*:2])([*:3])[*:4]' \
--anchor-structure examples/anchor_dummy.sdf \
--sectors-per-dummy 4 \
--dummy-angle-cutoff 80
```

For SDF/MOL anchors, dummy isotopes `1`, `2`, `3`, ... map to `[*:1]`, `[*:2]`, `[*:3]`, ... in the anchor SMILES. PocketField then clusters/ranks sectors around each dummy exit vector instead of assigning global top sectors blindly.

Disconnected anchors are allowed if every dummy atom is mapped:

```bash
--anchor-smiles '[*:1]c1ccccc1.[*:2]CC'
```

Bare unmapped dummies such as `*C.*N` are rejected because the tool cannot determine which spatial dummy corresponds to which growth site.

Use bridge mode when the disconnected components should be connected through an enumerated linker fragment:

```bash
--anchor-smiles '[*:1]SMI1.[*:2]SMI2' \
--bridge-anchor-dummies \
--linker-library examples/linkers.csv
```

In bridge mode, PocketField uses true two-dummy linkers such as `[*:1]CC[*:2]` or `[*:1]C(=O)N[*:2]`. It bonds anchor dummy site `[*:1]` to the linker `[*:1]` neighbor, anchor dummy site `[*:2]` to the linker `[*:2]` neighbor, then removes all dummy atoms. Candidate records use `mode: bridge` and should produce connected SMILES.

Linker libraries use the same `.csv`, `.json`, or `.jsonl` format as fragments, but each linker SMILES must contain exactly `[*:1]` and `[*:2]`.

Optional retrosynthesis validation can be run under `rxn-env`:

```bash
--retro-rules /Users/mai/Software/fragenv/data/uspto.templates.classified.json.gz
```

## Inspect Growth Outputs

After `grow` or `design`, inspect sector chemistry, dummy assignments, fragment/linker matches, desolvation penalties, and candidate score summaries:

```bash
/opt/anaconda3/envs/crem/bin/python -m pocketfield.cli inspect \
  --grow-dir out/pocket_001/grow \
  --top 10 \
  --json-out out/pocket_001/grow/inspection.json
```

This is read-only except for `--json-out`.

Useful generation controls:

```bash
--growth-depth 3
--request-limit 5000
--fragment-library examples/fragments.csv
--max-heavy-atoms 45
--max-clash-score 10.0
--max-field-score 100.0
--sector-match-weight 10.0
--linker-distance-tolerance 2.0
--linker-conformers 20
```

Fragment libraries can be `.csv`, `.json`, or `.jsonl`. Only `smiles` is required; other vendor columns are ignored.

Minimal CSV:

```csv
smiles
[*:1]CCO
[*:1]c1ccccc1
```

Each fragment SMILES must have:

- one attachment dummy, for example `[*:1]O`.

Optional columns:

- `name`: fragment label.
- `probes`: probe labels such as `hydrophobic` or `hbond_acceptor`.

## One-Step Design

`design` runs field building and candidate growth in one command:

```bash
/opt/anaconda3/envs/crem/bin/python -m pocketfield.cli design \
  --protein receptor.pdb \
  --ligand ligand.pdb \
  --out-dir out/pocket_001 \
  --growth-depth 3 \
  --fragment-library examples/fragments.csv
```

Outputs are split into:

- `out/pocket_001/field/`
- `out/pocket_001/grow/`

## Current Scope

This package creates the pocket field, sector plan, and first-pass RDKit candidates. It is not yet a production de novo design engine: scoring is field/clash/vector based with heuristic desolvation, and synthetic accessibility is only available through the optional retrosynthesis-template filter.
