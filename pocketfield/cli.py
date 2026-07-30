from __future__ import annotations

import argparse
import json
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
        elif args.command == "inspect":
            inspect_outputs(args)
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


def inspect_outputs(args: argparse.Namespace) -> None:
    paths = _inspect_paths(args)
    report = _build_inspection_report(paths=paths, top=args.top)
    _print_inspection_report(report)
    if args.json_out:
        write_json(args.json_out, report)
        print(f"Wrote {args.json_out}")


def _inspect_paths(args: argparse.Namespace) -> dict[str, Path]:
    base = Path(args.grow_dir) if args.grow_dir else None
    paths = {
        "sector_coefficients": Path(args.sector_coefficients)
        if args.sector_coefficients
        else (base / "sector_coefficients.json" if base else None),
        "fragment_features": Path(args.fragment_features)
        if args.fragment_features
        else (base / "fragment_features.json" if base else None),
        "linker_features": Path(args.linker_features)
        if args.linker_features
        else (base / "linker_features.json" if base else None),
        "dummy_assignments": Path(args.dummy_assignments)
        if args.dummy_assignments
        else (base / "dummy_sector_assignments.json" if base else None),
        "candidates": Path(args.candidates)
        if args.candidates
        else (base / "candidates.json" if base else None),
    }
    if paths["sector_coefficients"] is None:
        raise ValueError("Provide --grow-dir or --sector-coefficients.")
    return {key: value for key, value in paths.items() if value is not None}


def _build_inspection_report(paths: dict[str, Path], top: int) -> dict[str, object]:
    top = max(1, top)
    sectors = _read_required_json(paths["sector_coefficients"])
    fragments_payload = _read_optional_json(paths.get("fragment_features"))
    linkers_payload = _read_optional_json(paths.get("linker_features"))
    assignments_payload = _read_optional_json(paths.get("dummy_assignments"))
    candidates_payload = _read_optional_json(paths.get("candidates"))

    sector_items = list(sectors.get("items", []))
    sector_items.sort(key=lambda item: float(item.get("score", 0.0)))
    report: dict[str, object] = {
        "schema": "pocketfield.inspection_report.v1",
        "inputs": {key: str(path) for key, path in paths.items()},
        "top": top,
        "vector_keys": sectors.get("vector_keys", []),
        "sector_count": len(sector_items),
        "top_sectors": [_sector_summary(item) for item in sector_items[:top]],
    }

    if assignments_payload:
        report["dummy_assignments"] = _assignment_summary(assignments_payload, top)
    if fragments_payload:
        report["top_fragment_matches"] = _top_matches_by_sector(
            sectors=sector_items[:top],
            features=list(fragments_payload.get("items", [])),
            label_key="fragment_name",
            top=top,
        )
    if linkers_payload:
        report["top_linker_matches"] = _top_matches_by_sector(
            sectors=sector_items[:top],
            features=list(linkers_payload.get("items", [])),
            label_key="linker_name",
            top=top,
        )
    if candidates_payload:
        report["candidate_summary"] = _candidate_summary(candidates_payload, top)
    return report


def _read_required_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise ValueError(f"Inspection input does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_optional_json(path: Path | None) -> dict[str, object] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _sector_summary(item: dict[str, object]) -> dict[str, object]:
    features = item.get("features", {})
    if not isinstance(features, dict):
        features = {}
    return {
        "sector_id": item.get("sector_id"),
        "score": item.get("score"),
        "best_probe": item.get("best_probe"),
        "depth": _round_float(features.get("depth")),
        "steric_openness": _round_float(features.get("steric_openness")),
        "polarity": _round_float(features.get("polarity")),
        "water_displacement": _round_float(features.get("water_displacement")),
        "buried_polar_support": _round_float(features.get("buried_polar_support")),
        "desolvation_risk": _round_float(features.get("desolvation_risk")),
    }


def _assignment_summary(payload: dict[str, object], top: int) -> list[dict[str, object]]:
    assignments = payload.get("assignments", [])
    if not isinstance(assignments, list):
        return []
    summary = []
    for assignment in assignments:
        if not isinstance(assignment, dict):
            continue
        sectors = assignment.get("sectors", [])
        if not isinstance(sectors, list):
            sectors = []
        summary.append(
            {
                "anchor_map": assignment.get("anchor_map"),
                "sector_count": len(sectors),
                "sectors": sectors[:top],
            }
        )
    return summary


def _top_matches_by_sector(
    sectors: list[dict[str, object]],
    features: list[dict[str, object]],
    label_key: str,
    top: int,
) -> list[dict[str, object]]:
    results = []
    for sector in sectors:
        ranked = []
        for feature in features:
            match = _score_inspection_match(sector, feature)
            ranked.append(
                {
                    label_key: feature.get("name"),
                    "smiles": feature.get("smiles"),
                    **match,
                }
            )
        ranked.sort(key=lambda item: float(item["score"]))
        results.append({"sector_id": sector.get("sector_id"), "matches": ranked[:top]})
    return results


def _score_inspection_match(
    sector: dict[str, object],
    feature: dict[str, object],
) -> dict[str, float]:
    sector_vector = np.asarray(sector.get("vector", []), dtype=np.float64)
    candidate_vector = np.asarray(feature.get("vector", []), dtype=np.float64)
    if sector_vector.shape != candidate_vector.shape:
        raise ValueError("Sector and candidate vectors use incompatible dimensions.")
    dot = float(np.dot(sector_vector, candidate_vector))
    mismatch = float(np.linalg.norm(sector_vector - candidate_vector))
    vector_match_score = -dot + 0.35 * mismatch
    desolvation_penalty = _inspection_desolvation_penalty(
        _dict_feature(sector),
        _dict_feature(feature),
    )
    return {
        "score": _round_float(vector_match_score + desolvation_penalty),
        "vector_match_score": _round_float(vector_match_score),
        "desolvation_penalty": _round_float(desolvation_penalty),
        "dot": _round_float(dot),
        "mismatch": _round_float(mismatch),
    }


def _inspection_desolvation_penalty(
    sector_features: dict[str, object],
    candidate_features: dict[str, object],
) -> float:
    sector_risk = float(sector_features.get("desolvation_risk", 0.0))
    candidate_cost = float(candidate_features.get("polar_desolvation_cost", 0.0))
    polar_support_match = min(
        float(sector_features.get("buried_polar_support", 0.0)),
        float(candidate_features.get("buried_polar_support", 0.0)),
    )
    return 0.55 * sector_risk * candidate_cost * (1.0 - polar_support_match)


def _dict_feature(item: dict[str, object]) -> dict[str, object]:
    features = item.get("features", {})
    return features if isinstance(features, dict) else {}


def _candidate_summary(payload: dict[str, object], top: int) -> dict[str, object]:
    candidates = payload.get("candidates", [])
    if not isinstance(candidates, list):
        candidates = []
    best = sorted(candidates, key=lambda item: float(item.get("score", 0.0)))[:top]
    return {
        "candidate_count": payload.get("candidate_count", len(candidates)),
        "attempted_requests": payload.get("attempted_requests"),
        "total_requests": payload.get("total_requests"),
        "rejections": payload.get("rejections", {}),
        "best_candidates": [
            {
                "rank": candidate.get("rank"),
                "smiles": candidate.get("smiles"),
                "mode": candidate.get("mode"),
                "score": _round_float(candidate.get("score")),
                "field_score": _round_float(candidate.get("field_score")),
                "clash_score": _round_float(candidate.get("clash_score")),
                "sector_match_score": _round_float(candidate.get("sector_match_score")),
            }
            for candidate in best
            if isinstance(candidate, dict)
        ],
    }


def _print_inspection_report(report: dict[str, object]) -> None:
    print("PocketField Inspection")
    print(f"Sectors: {report['sector_count']}  Top: {report['top']}")
    print()
    print("Top sectors")
    for sector in report.get("top_sectors", []):
        print(
            "  sector={sector_id} score={score} probe={best_probe} depth={depth} "
            "open={steric_openness} water={water_displacement} polar={buried_polar_support} "
            "risk={desolvation_risk}".format(**sector)
        )

    assignments = report.get("dummy_assignments", [])
    if assignments:
        print()
        print("Dummy assignments")
        for assignment in assignments:
            sector_ids = [
                str(sector.get("sector_id"))
                for sector in assignment["sectors"]
                if isinstance(sector, dict)
            ]
            print(
                f"  map={assignment['anchor_map']} sectors={assignment['sector_count']} "
                f"top={','.join(sector_ids)}"
            )

    _print_matches("Top fragment matches", report.get("top_fragment_matches", []), "fragment_name")
    _print_matches("Top linker matches", report.get("top_linker_matches", []), "linker_name")

    candidate_summary = report.get("candidate_summary")
    if isinstance(candidate_summary, dict):
        print()
        print(
            "Candidates: {candidate_count}  Attempted: {attempted_requests}  "
            "Requests: {total_requests}".format(**candidate_summary)
        )
        rejections = candidate_summary.get("rejections", {})
        if rejections:
            rejection_text = ", ".join(f"{key}={value}" for key, value in sorted(rejections.items()))
            print(f"Rejections: {rejection_text}")
        for candidate in candidate_summary.get("best_candidates", []):
            print(
                "  rank={rank} score={score} field={field_score} clash={clash_score} "
                "match={sector_match_score} mode={mode} smiles={smiles}".format(**candidate)
            )


def _print_matches(title: str, groups: object, label_key: str) -> None:
    if not groups:
        return
    print()
    print(title)
    for group in groups:
        if not isinstance(group, dict):
            continue
        print(f"  sector={group.get('sector_id')}")
        for match in group.get("matches", []):
            print(
                "    {label} score={score} vec={vector_match_score} desolv={desolvation_penalty} "
                "smiles={smiles}".format(label=match.get(label_key), **match)
            )


def _round_float(value: object) -> float | None:
    if value is None:
        return None
    return round(float(value), 4)


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

    inspect_parser = subparsers.add_parser("inspect", help="inspect growth outputs and rankings")
    inspect_parser.add_argument(
        "--grow-dir",
        help="Growth output directory containing sector/features/candidates JSON files.",
    )
    inspect_parser.add_argument(
        "--sector-coefficients",
        help="Explicit path to sector_coefficients.json. Required if --grow-dir is not provided.",
    )
    inspect_parser.add_argument("--fragment-features", help="Explicit path to fragment_features.json.")
    inspect_parser.add_argument("--linker-features", help="Explicit path to linker_features.json.")
    inspect_parser.add_argument(
        "--dummy-assignments",
        help="Explicit path to dummy_sector_assignments.json.",
    )
    inspect_parser.add_argument("--candidates", help="Explicit path to candidates.json.")
    inspect_parser.add_argument("--top", type=int, default=5, help="Number of rows per section.")
    inspect_parser.add_argument("--json-out", help="Optional path to write inspection report JSON.")
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
