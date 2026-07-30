# PocketField Pipeline

PocketField is a deterministic pocket-conditioned molecule generation pipeline. It uses probe-based spherical interaction fields to rank growth sectors, then enumerates fragments or linkers from mapped anchor dummy atoms.

The central idea is:

```text
protein pocket -> spherical probe field -> sector coefficients -> anchor dummy sectors -> fragment/linker enumeration -> RDKit candidates
```

It does not use a learned fragment proposer. Fragment and linker choices are ranked by explicit coefficient-vector matching plus field/clash scoring.

## Core Objects

### Pocket Field

The pocket field is a multi-shell spherical sampling of the binding pocket.

For each sample point, PocketField evaluates interaction energy for chemical probes:

```text
hydrophobic
hbond_donor
hbond_acceptor
positive
negative
aromatic, if requested
```

The field is computed in 3D first, then indexed by sphere direction and shell radius. This avoids treating a distorted 2D conformal map as the physical source of truth.

### Sectors

The sphere is partitioned into approximately equal sectors. Each sector receives:

```text
best probe
best shell
field score
direction vector
gradient-derived growth vector
```

Sectors are not the same as anchor growth sites. If an anchor is spatially known, sectors are clustered around each anchor dummy exit vector.

### Anchor

The anchor is the fixed starting scaffold or set of scaffolds. It must use mapped RDKit dummy atoms:

```text
[*:1]
[*:2]
[*:3]
```

Valid:

```text
c1cc([*:1])ccc1[*:2]
[*:1]c1ccccc1.[*:2]CC
```

Invalid:

```text
*c1ccccc1.*CC
```

Bare `*` atoms are rejected because the tool cannot map them to spatial dummy positions.

### Anchor Structure

`--anchor-structure` is optional but important when the anchor is already placed in the pocket.

It should be a 3D SDF/MOL/PDB containing dummy atoms positioned at the anchor exit vectors. In SDF/MOL files, dummy isotope labels map to anchor SMILES dummy labels:

```text
dummy isotope 1 -> [*:1]
dummy isotope 2 -> [*:2]
dummy isotope 3 -> [*:3]
```

When this file is provided, PocketField assigns sectors around each dummy vector:

```text
anchor dummy position - dummy neighbor position = exit vector
```

Then sectors are filtered/ranked by angular distance from that exit vector.

### Fragments

Fragments are one-dummy substituents used for decoration/growth:

```text
[*:1]C
[*:1]O
[*:1]C(=O)O
[*:1]c1ccccc1
```

They attach to one anchor dummy site.

### Linkers

Linkers are two-dummy bridge fragments:

```text
[*:1]CC[*:2]
[*:1]C(=O)N[*:2]
[*:1]c1ccc([*:2])cc1
```

They connect two anchor dummy sites in bridge mode:

```text
anchor [*:1] -> linker [*:1]
anchor [*:2] -> linker [*:2]
```

This is directional two-ended linker growth, not a one-atom bridge-center shortcut.

## CLI Overview

### Build A Field

```bash
/opt/anaconda3/envs/crem/bin/python -m pocketfield.cli build \
  --protein receptor.pdb \
  --ligand ligand.pdb \
  --out-dir out/pocket/field
```

`--ligand` defines the pocket center by ligand centroid. It does not define growth points.

Alternative:

```bash
--center 12.4,3.1,-8.0
```

Important options:

```bash
--shells 2.5,3.5,4.5,6.0
--samples 2048
--sectors 64
--pocket-radius 8.0
--top-k 20
```

Outputs:

```text
field.npz
metadata.json
growth_plan.json
```

### Grow From An Existing Field

```bash
/opt/anaconda3/envs/crem/bin/python -m pocketfield.cli grow \
  --field out/pocket/field/field.npz \
  --plan out/pocket/field/growth_plan.json \
  --metadata out/pocket/field/metadata.json \
  --out-dir out/pocket/grow \
  --anchor-smiles 'c1cc([*:1])ccc1[*:2]'
```

### One-Step Design

```bash
/opt/anaconda3/envs/crem/bin/python -m pocketfield.cli design \
  --protein receptor.pdb \
  --ligand ligand.pdb \
  --out-dir out/pocket \
  --anchor-smiles 'c1cc([*:1])ccc1[*:2]'
```

This writes:

```text
out/pocket/field/
out/pocket/grow/
```

## Spatial Anchor Workflow

Use this when the anchor position and dummy directions are known.

```bash
/opt/anaconda3/envs/crem/bin/python -m pocketfield.cli design \
  --protein receptor.pdb \
  --ligand ligand.pdb \
  --out-dir out/anchored \
  --anchor-smiles 'C([*:1])([*:2])([*:3])[*:4]' \
  --anchor-structure anchor_dummy.sdf \
  --sectors-per-dummy 4 \
  --dummy-angle-cutoff 80
```

The tool will:

```text
1. Read anchor dummy coordinates.
2. Compute one exit vector per dummy.
3. Assign nearby sectors to each dummy.
4. Rank fragments for each dummy-specific sector.
5. Enumerate and score candidates.
6. Fit generated conformers back onto the placed anchor heavy atoms.
```

Output file:

```text
dummy_sector_assignments.json
```

Example assignment:

```json
{
  "anchor_map": 1,
  "sectors": [
    {"sector_id": 13, "angle_deg": 44.8},
    {"sector_id": 0, "angle_deg": 69.6}
  ]
}
```

## Disconnected Anchor Decoration

Disconnected mapped anchors are allowed:

```bash
--anchor-smiles '[*:1]c1ccccc1.[*:2]CC'
```

Without bridge mode, each dummy can be decorated independently. The output molecule may remain disconnected if only one component is grown.

Use this only if disconnected components are intended, or if you plan to postprocess them.

## Bridge Mode

Bridge mode is for cases like:

```text
[*:1]SMI1.[*:2]SMI2
```

where the two components should be connected through a linker.

Example:

```bash
/opt/anaconda3/envs/crem/bin/python -m pocketfield.cli design \
  --protein receptor.pdb \
  --ligand ligand.pdb \
  --out-dir out/bridge \
  --anchor-smiles '[*:1]c1ccccc1.[*:2]CC' \
  --bridge-anchor-dummies \
  --linker-library examples/linkers.csv
```

With linker:

```text
[*:1]C(=O)N[*:2]
```

PocketField performs:

```text
anchor component 1 neighbor -- C(=O)N -- anchor component 2 neighbor
```

Example output:

```text
CCNC(=O)c1ccccc1
```

Bridge mode requires linkers with exactly two mapped dummies:

```text
[*:1]...[*:2]
```

One-dummy fragments are not used as bridge linkers.

When `--anchor-structure` is provided, bridge mode also applies a coarse linker geometry prefilter. PocketField samples linker conformers and compares the linker dummy-to-dummy span against the placed anchor dummy-to-dummy distance.

Controls:

```bash
--linker-distance-tolerance 2.0
--linker-conformers 20
```

This is a fast prefilter. It is not a full endpoint-constrained linker conformer fit.

## Fragment And Linker Libraries

CSV format can be minimal. PocketField only requires a `smiles` column and ignores other vendor columns:

```csv
smiles
[*:1]CC
[*:1]C(=O)N
[*:1]c1ccccc1
```

Optional `name` and `probes` columns are supported but not required:

```csv
name,smiles,probes
ethyl,[*:1]CC,hydrophobic
amide,[*:1]C(=O)N,"hbond_donor,hbond_acceptor"
```

Minimal linker CSV:

```csv
smiles
[*:1]CC[*:2]
[*:1]C(=O)N[*:2]
```

Optional linker CSV:

```csv
name,smiles,probes
ethylene,[*:1]CC[*:2],hydrophobic
amide,[*:1]C(=O)N[*:2],"hbond_donor,hbond_acceptor"
phenylene,[*:1]c1ccc([*:2])cc1,"hydrophobic,aromatic"
```

JSON format:

```json
{
  "fragments": [
    {"name": "ethyl", "smiles": "[*:1]CC", "probes": ["hydrophobic"]}
  ]
}
```

Linker JSON:

```json
{
  "linkers": [
    {"name": "ethylene", "smiles": "[*:1]CC[*:2]", "probes": ["hydrophobic"]}
  ]
}
```

CSV and JSONL libraries are streamed row-by-row during parsing, so large DBeaver exports do not create an extra full raw-row copy in memory. PocketField still stores the valid loaded fragment/linker objects and their feature vectors for ranking.

## Vector Matching

PocketField computes coefficient vectors for sectors and feature vectors for fragments/linkers.

Vector keys:

```text
hydrophobic
hbond_donor
hbond_acceptor
positive
negative
aromatic
steric_openness
small_size
medium_size
large_size
rigidity
polarity
depth
water_displacement
buried_polar_support
```

Sector coefficients come from:

```text
probe well strength
best shell depth
steric openness along sector direction
polarity and rigidity preference
burial-dependent water displacement and buried polar support
```

Fragment/linker features come from:

```text
atom types
donor/acceptor counts
formal charge
aromaticity
heavy atom count
bond rigidity proxy
probe tags
hydrophobic/aromatic water-displacement capacity
polar desolvation cost
```

The match score is:

```text
sector_match_score =
  -dot(sector_vector, candidate_vector)
  + mismatch_penalty
  + unsupported_polar_desolvation_penalty
```

Lower is better.

The desolvation terms are deterministic heuristics:

```text
burial = 1 - steric_openness
sector water_displacement = buried hydrophobic/aromatic or unsupported buried volume
sector buried_polar_support = buried donor/acceptor/charge field support
candidate water_displacement = hydrophobic/aromatic capacity
candidate polar_desolvation_cost = exposed polar character that is costly to bury
```

The explicit penalty is applied outside the dot product. This is important because a desolvation risk should penalize unsupported buried polar atoms; it should not become a positive coefficient that rewards matching a polar fragment.

Final score:

```text
final_score =
  raw_field_clash_score
  + sector_match_weight * sector_match_score
```

where:

```text
raw_field_clash_score = field_score + 25 * clash_score + 0.02 * heavy_atoms
```

## Output Files

`field.npz`:

```text
center
shells
directions
points
energies
gradients
sector_ids
sector_centers
probe_names
```

`growth_plan.json`:

```text
ranked single sectors
ranked paired sectors
anchor metadata
```

`sector_coefficients.json`:

```text
sector vectors used for dot-product matching
```

`fragment_features.json`:

```text
fragment vectors
```

`linker_features.json`:

```text
linker vectors
```

`dummy_sector_assignments.json`:

```text
dummy map -> nearby sectors
```

`candidates.json`:

```text
rank
SMILES
mode
fragments
linkers
sector ids
attachment maps
field score
clash score
raw score
sector match score
final score
rejection diagnostics
```

`candidates.sdf`:

```text
3D RDKit molecules with score properties
```

## Filtering

Important filters:

```bash
--max-heavy-atoms 45
--max-clash-score 10.0
--max-field-score 100.0
--sector-match-weight 10.0
```

Optional retrosynthesis validation:

```bash
/opt/anaconda3/envs/rxn-env/bin/python -m pocketfield.cli design \
  ... \
  --retro-rules /Users/mai/Software/fragenv/data/uspto.templates.classified.json.gz
```

This uses `RetrosynthesisValidator` from:

```text
/Users/mai/Software/fragenv/scripts/merge_utils.py
```

PocketField marks newly formed anchor-fragment or anchor-linker bonds with isotope pair `88/99`, calls the validator, then strips isotopes before writing outputs. This filter is optional because rdchiral/template validation is slower than field and clash scoring.

For artificial examples, filters may need to be relaxed:

```bash
--max-clash-score 1000
--max-field-score 10000
```

For real pockets, strict filters are preferable.

## Current Limitations

PocketField is still an early deterministic design tool.

Current limitations:

```text
No docking refinement.
Optional retrosynthesis validation is available, but no full synthetic accessibility model.
No conformer ensemble scoring.
No explicit-water thermodynamics; desolvation is currently a heuristic sector coefficient and ranking penalty.
No pharmacophore constraints beyond probe fields.
No torsion-aware linker fitting.
No learned fragment proposal.
```

Bridge mode currently has a coarse linker span prefilter and then embeds/aligns the full molecule. A production linker workflow should add:

```text
stricter distance/angle compatibility between anchor dummy positions
endpoint-to-endpoint linker conformer fitting
strain penalty
torsion constraints
post-assembly local optimization in the pocket
```

Explicit water/desolvation policy:

```text
Do not start with full explicit-water thermodynamics inside PocketField.
```

A practical staged approach is:

```text
1. Use the current desolvation coefficient/penalty as a fast first-pass filter.
2. Allow optional conserved water points as pseudo-atoms/probes.
3. Add user-supplied hydration maps for water displacement scoring.
4. Only later integrate external water analysis, such as GIST/WaterMap-style maps.
```

For this deterministic enumerator, conserved-water-aware probe fields are likely more useful than a heavy explicit-water model.

## Recommended Real Workflow

1. Prepare receptor PDB.
2. Prepare reference ligand or known pocket center.
3. Prepare anchor SMILES with mapped dummy atoms.
4. If anchor is placed, prepare 3D anchor SDF with dummy isotope labels.
5. Build field with sufficient shells/sectors.
6. Use custom fragment/linker libraries.
7. Inspect `dummy_sector_assignments.json`.
8. Inspect `sector_coefficients.json`.
9. Generate candidates.
10. Filter by clash, field score, and properties.
11. Dock/refine externally.
