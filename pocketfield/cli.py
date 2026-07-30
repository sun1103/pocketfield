from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from pocketfield import __version__
from pocketfield.atoms import read_pdb_atoms, read_pdb_coordinates, select_pocket_atoms
from pocketfield.field import compute_field
from pocketfield.io import write_json
from pocketfield.plan import build_growth_plan
from pocketfield.probes import get_probes
from pocketfield.rdkit_grow import grow_candidates, load_fragments
from pocketfield.sectors import assign_sectors, rank_sector_pairs, rank_single_sectors
from pocketfield.sphere import parse_center, parse_float_list, shell_points, fibonacci_directions


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            build(args)
        elif args.command == "grow":
            grow(args)
        elif args.command == "design":
            design(args)
        elif args.command == "fragments":
            fragments(args)
        else:
            parser.error("missing command")
    except Exception as exc:
        print(f"pocketfield: error: {exc}", file=sys.stderr)
        return 2
    return 0


def build(args: argparse.Namespace) -> None:
    protein_path = Path(args.protein)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    center = _resolve_center(args)
    shells = parse_float_list(args.shells)
    probes = get_probes(args.probes)

    atoms = read_pdb_atoms(protein_path, include_hetatm=args.include_hetatm)
    if not atoms:
        raise ValueError(f"No protein atoms found in {protein_path}")
    pocket_atoms = select_pocket_atoms(atoms, center, args.pocket_radius)

    directions = fibonacci_directions(args.samples)
    points = shell_points(center, shells, directions)
    sector_ids, sector_centers = assign_sectors(directions, args.sectors)
    energies, gradients = compute_field(
        pocket_atoms=pocket_atoms,
        points=points,
        probes=probes,
        dielectric=args.dielectric,
    )

    probe_names = [probe.name for probe in probes]
    single_sectors = rank_single_sectors(
        energies,
        gradients,
        probe_names,
        shells,
        sector_ids,
        sector_centers,
        top_k=args.top_k,
    )
    sector_pairs = rank_sector_pairs(
        energies,
        probe_names,
        shells,
        sector_ids,
        sector_centers,
        top_k=args.top_k,
        neighbors_per_sector=args.neighbors_per_sector,
    )
    growth_plan = build_growth_plan(
        center=center.tolist(),
        single_sectors=single_sectors,
        sector_pairs=sector_pairs,
        anchor_radius=args.anchor_radius,
    )

    np.savez_compressed(
        out_dir / "field.npz",
        center=center,
        shells=np.asarray(shells, dtype=np.float64),
        directions=directions,
        points=points,
        energies=energies,
        gradients=gradients,
        sector_ids=sector_ids,
        sector_centers=sector_centers,
        probe_names=np.asarray(probe_names),
    )
    write_json(out_dir / "growth_plan.json", growth_plan)
    write_json(
        out_dir / "metadata.json",
        {
            "schema": "pocketfield.metadata.v1",
            "version": __version__,
            "protein": str(protein_path),
            "ligand": str(args.ligand) if args.ligand else None,
            "center": center.tolist(),
            "pocket_radius": args.pocket_radius,
            "shells": shells,
            "samples": args.samples,
            "sectors": args.sectors,
            "dielectric": args.dielectric,
            "probes": [probe.to_json() for probe in probes],
            "pocket_atom_count": len(pocket_atoms),
            "pocket_atoms": [atom.to_json() for atom in pocket_atoms],
            "outputs": {
                "field": "field.npz",
                "growth_plan": "growth_plan.json",
            },
        },
    )

    print(f"Wrote {out_dir / 'field.npz'}")
    print(f"Wrote {out_dir / 'growth_plan.json'}")
    print(f"Wrote {out_dir / 'metadata.json'}")


def grow(args: argparse.Namespace) -> None:
    result = grow_candidates(
        field_path=args.field,
        plan_path=args.plan,
        metadata_path=args.metadata,
        out_dir=args.out_dir,
        anchor_smiles=args.anchor_smiles,
        anchor_structure=args.anchor_structure,
        sectors_per_dummy=args.sectors_per_dummy,
        dummy_angle_cutoff=args.dummy_angle_cutoff,
        connect_anchor_dummies=args.bridge_anchor_dummies,
        linker_distance_tolerance=args.linker_distance_tolerance,
        linker_conformers=args.linker_conformers,
        max_candidates=args.max_candidates,
        top_sectors=args.top_sectors,
        fragments_per_sector=args.fragments_per_sector,
        growth_depth=args.growth_depth,
        request_limit=args.request_limit,
        include_pairs=args.include_pairs,
        fragment_library=args.fragment_library,
        linker_library=args.linker_library,
        max_heavy_atoms=args.max_heavy_atoms,
        max_clash_score=args.max_clash_score,
        max_field_score=args.max_field_score,
        sector_match_weight=args.sector_match_weight,
        retro_rules=args.retro_rules,
        retro_script=args.retro_script,
        random_seed=args.random_seed,
    )
    out_dir = Path(args.out_dir)
    print(f"Wrote {out_dir / 'candidates.sdf'}")
    print(f"Wrote {out_dir / 'candidates.json'}")
    print(f"Generated {result['candidate_count']} candidates")


def design(args: argparse.Namespace) -> None:
    field_dir = Path(args.out_dir) / "field"
    grow_dir = Path(args.out_dir) / "grow"
    build(
        argparse.Namespace(
            protein=args.protein,
            ligand=args.ligand,
            center=args.center,
            out_dir=str(field_dir),
            shells=args.shells,
            samples=args.samples,
            sectors=args.sectors,
            pocket_radius=args.pocket_radius,
            dielectric=args.dielectric,
            probes=args.probes,
            include_hetatm=args.include_hetatm,
            top_k=args.top_k,
            neighbors_per_sector=args.neighbors_per_sector,
            anchor_radius=args.anchor_radius,
        )
    )
    grow(
        argparse.Namespace(
            field=str(field_dir / "field.npz"),
            plan=str(field_dir / "growth_plan.json"),
            metadata=str(field_dir / "metadata.json"),
            out_dir=str(grow_dir),
            anchor_smiles=args.anchor_smiles,
            anchor_structure=args.anchor_structure,
            sectors_per_dummy=args.sectors_per_dummy,
            dummy_angle_cutoff=args.dummy_angle_cutoff,
            bridge_anchor_dummies=args.bridge_anchor_dummies,
            linker_distance_tolerance=args.linker_distance_tolerance,
            linker_conformers=args.linker_conformers,
            max_candidates=args.max_candidates,
            top_sectors=args.top_sectors,
            fragments_per_sector=args.fragments_per_sector,
            growth_depth=args.growth_depth,
            request_limit=args.request_limit,
            include_pairs=args.include_pairs,
            fragment_library=args.fragment_library,
            linker_library=args.linker_library,
            max_heavy_atoms=args.max_heavy_atoms,
            max_clash_score=args.max_clash_score,
            max_field_score=args.max_field_score,
            sector_match_weight=args.sector_match_weight,
            retro_rules=args.retro_rules,
            retro_script=args.retro_script,
            random_seed=args.random_seed,
        )
    )


def fragments(args: argparse.Namespace) -> None:
    for fragment in load_fragments(args.fragment_library):
        print(f"{fragment.name}\t{fragment.smiles}\t{','.join(fragment.probes)}")


def _resolve_center(args: argparse.Namespace) -> np.ndarray:
    if args.center and args.ligand:
        raise ValueError("Use either --center or --ligand, not both.")
    if args.center:
        return parse_center(args.center)
    if args.ligand:
        coords = read_pdb_coordinates(args.ligand)
        return np.mean(coords, axis=0)
    raise ValueError("Provide --center x,y,z or --ligand ligand.pdb")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pocketfield",
        description="Build probe-based spherical pocket fields for molecule generation.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    build_parser = subparsers.add_parser("build", help="build a spherical pocket field")
    _add_build_arguments(build_parser)

    grow_parser = subparsers.add_parser("grow", help="grow RDKit candidates from a field plan")
    _add_grow_arguments(grow_parser)

    design_parser = subparsers.add_parser("design", help="build a field and grow candidates")
    _add_build_arguments(design_parser)
    _add_grow_arguments(design_parser, include_inputs=False)

    fragments_parser = subparsers.add_parser("fragments", help="list built-in or custom fragments")
    fragments_parser.add_argument(
        "--fragment-library",
        help="Optional .json, .jsonl, or .csv fragment library. Only smiles is required.",
    )
    return parser


def _add_build_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--protein", required=True, help="Protein/receptor PDB file.")
    parser.add_argument("--ligand", help="Ligand PDB file used to compute the field center.")
    parser.add_argument("--center", help="Field center as x,y,z.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument("--shells", default="2.5,3.5,4.5,6.0", help="Shell radii in Angstrom.")
    parser.add_argument("--samples", type=int, default=2048, help="Spherical samples per shell.")
    parser.add_argument("--sectors", type=int, default=64, help="Number of equal-ish sectors.")
    parser.add_argument("--pocket-radius", type=float, default=8.0, help="Pocket atom cutoff radius.")
    parser.add_argument("--dielectric", type=float, default=20.0, help="Effective dielectric.")
    parser.add_argument(
        "--probes",
        default="hydrophobic,hbond_donor,hbond_acceptor,positive,negative",
        help="Comma-separated probe names.",
    )
    parser.add_argument(
        "--include-hetatm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include HETATM records from the protein file as pocket atoms.",
    )
    parser.add_argument("--top-k", type=int, default=20, help="Number of ranked sectors to output.")
    parser.add_argument(
        "--neighbors-per-sector",
        type=int,
        default=6,
        help="Adjacent sector count used for paired-sector ranking.",
    )
    parser.add_argument(
        "--anchor-radius",
        type=float,
        default=1.5,
        help="Suggested central anchor radius recorded in growth_plan.json.",
    )


def _add_grow_arguments(parser: argparse.ArgumentParser, include_inputs: bool = True) -> None:
    if include_inputs:
        parser.add_argument("--field", required=True, help="Input field.npz from pocketfield build.")
        parser.add_argument(
            "--plan", required=True, help="Input growth_plan.json from pocketfield build."
        )
        parser.add_argument(
            "--metadata", required=True, help="Input metadata.json from pocketfield build."
        )
        parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument(
        "--anchor-smiles",
        default="C([*:1])([*:2])([*:3])[*:4]",
        help="Anchor SMILES with dummy atoms [*:1], [*:2], ... as growth points.",
    )
    parser.add_argument(
        "--anchor-structure",
        help=(
            "Optional 3D anchor SDF/MOL/PDB with dummy atoms positioned in pocket coordinates. "
            "Dummy isotopes 1,2,... map to [*:1], [*:2], ..."
        ),
    )
    parser.add_argument(
        "--sectors-per-dummy",
        type=int,
        default=4,
        help="Number of nearby sectors assigned to each positioned dummy atom.",
    )
    parser.add_argument(
        "--dummy-angle-cutoff",
        type=float,
        default=80.0,
        help="Maximum angle in degrees from dummy exit vector to assigned sectors.",
    )
    parser.add_argument(
        "--bridge-anchor-dummies",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Connect pairs of anchor dummy sites through one linker fragment instead of decorating sites independently.",
    )
    parser.add_argument(
        "--linker-distance-tolerance",
        type=float,
        default=2.0,
        help="Allowed Angstrom mismatch between placed anchor dummy distance and sampled linker dummy span.",
    )
    parser.add_argument(
        "--linker-conformers",
        type=int,
        default=20,
        help="Number of linker conformers sampled for bridge geometry prefiltering.",
    )
    parser.add_argument(
        "--max-candidates", type=int, default=50, help="Maximum generated candidates."
    )
    parser.add_argument(
        "--top-sectors", type=int, default=12, help="Number of top sectors/pairs to expand."
    )
    parser.add_argument(
        "--fragments-per-sector",
        type=int,
        default=4,
        help="Probe-compatible fragments tried for each sector.",
    )
    parser.add_argument(
        "--growth-depth",
        type=int,
        default=2,
        help="Maximum number of anchor attachment points to fill from sectors.",
    )
    parser.add_argument(
        "--request-limit",
        type=int,
        default=1000,
        help="Maximum fragment/sector assembly requests to try before filtering.",
    )
    parser.add_argument(
        "--include-pairs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also generate candidates that span adjacent sector pairs.",
    )
    parser.add_argument(
        "--fragment-library",
        help="Optional .json, .jsonl, or .csv fragment library. Only smiles is required.",
    )
    parser.add_argument(
        "--linker-library",
        help="Optional .json, .jsonl, or .csv linker library. Only smiles is required.",
    )
    parser.add_argument(
        "--max-heavy-atoms",
        type=int,
        default=45,
        help="Reject candidates above this heavy atom count.",
    )
    parser.add_argument(
        "--max-clash-score",
        type=float,
        default=10.0,
        help="Reject candidates above this protein clash score.",
    )
    parser.add_argument(
        "--max-field-score",
        type=float,
        default=100.0,
        help="Reject candidates with field_score above this threshold. Lower is better.",
    )
    parser.add_argument(
        "--sector-match-weight",
        type=float,
        default=10.0,
        help="Weight for the sector-fragment dot-product score in final ranking.",
    )
    parser.add_argument(
        "--retro-rules",
        help="Optional retrosynthesis rule file, for example uspto.templates.classified.json.gz.",
    )
    parser.add_argument(
        "--retro-script",
        default="/Users/mai/Software/fragenv/scripts/merge_utils.py",
        help="Path to merge_utils.py containing RetrosynthesisValidator.",
    )
    parser.add_argument("--random-seed", type=int, default=13, help="RDKit embedding seed.")


if __name__ == "__main__":
    raise SystemExit(main())
