from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pocketfield.cli import main


ROOT = Path(__file__).resolve().parents[1]


class PocketFieldSmokeTest(unittest.TestCase):
    def test_build_outputs_field_and_growth_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rc = main(
                [
                    "build",
                    "--protein",
                    str(ROOT / "examples" / "tiny_receptor.pdb"),
                    "--ligand",
                    str(ROOT / "examples" / "tiny_ligand.pdb"),
                    "--out-dir",
                    tmp,
                    "--samples",
                    "64",
                    "--sectors",
                    "12",
                    "--shells",
                    "2.5,3.5",
                    "--top-k",
                    "3",
                ]
            )

            self.assertEqual(rc, 0)
            field = np.load(Path(tmp) / "field.npz")
            self.assertEqual(field["energies"].shape, (5, 2, 64))
            self.assertEqual(field["gradients"].shape, (5, 2, 64, 3))

            plan = json.loads((Path(tmp) / "growth_plan.json").read_text())
            self.assertEqual(plan["schema"], "pocketfield.growth_plan.v1")
            self.assertEqual(len(plan["single_sector_growing"]), 3)
            self.assertEqual(len(plan["paired_sector_growing"]), 3)

    def test_grow_outputs_rdkit_candidates_when_rdkit_available(self) -> None:
        try:
            import rdkit  # noqa: F401
        except ImportError:
            self.skipTest("RDKit is not available in this Python environment.")

        with tempfile.TemporaryDirectory() as tmp:
            build_dir = Path(tmp) / "build"
            grow_dir = Path(tmp) / "grow"
            build_rc = main(
                [
                    "build",
                    "--protein",
                    str(ROOT / "examples" / "tiny_receptor.pdb"),
                    "--ligand",
                    str(ROOT / "examples" / "tiny_ligand.pdb"),
                    "--out-dir",
                    str(build_dir),
                    "--samples",
                    "64",
                    "--sectors",
                    "12",
                    "--shells",
                    "2.5,3.5",
                    "--top-k",
                    "4",
                ]
            )
            self.assertEqual(build_rc, 0)

            grow_rc = main(
                [
                    "grow",
                    "--field",
                    str(build_dir / "field.npz"),
                    "--plan",
                    str(build_dir / "growth_plan.json"),
                    "--metadata",
                    str(build_dir / "metadata.json"),
                    "--out-dir",
                    str(grow_dir),
                    "--max-candidates",
                    "5",
                    "--top-sectors",
                    "3",
                    "--fragments-per-sector",
                    "2",
                    "--max-clash-score",
                    "1000",
                    "--max-field-score",
                    "10000",
                ]
            )
            self.assertEqual(grow_rc, 0)

            candidates = json.loads((grow_dir / "candidates.json").read_text())
            self.assertGreater(candidates["candidate_count"], 0)
            self.assertGreater(candidates["attempted_requests"], 0)
            self.assertTrue((grow_dir / "candidates.sdf").exists())
            self.assertTrue((grow_dir / "sector_coefficients.json").exists())
            self.assertTrue((grow_dir / "fragment_features.json").exists())
            self.assertIn("sector_match_score", candidates["candidates"][0])
            # Conformer ensemble scoring records per-candidate stats.
            self.assertIn("n_conformers", candidates["candidates"][0])
            self.assertIn("best_conf_id", candidates["candidates"][0])
            self.assertIn("score_mean", candidates["candidates"][0])
            self.assertGreaterEqual(candidates["candidates"][0]["n_conformers"], 1)
            self.assertGreaterEqual(candidates["candidates"][0]["best_conf_id"], 0)

            sector_coefficients = json.loads((grow_dir / "sector_coefficients.json").read_text())
            self.assertIn("water_displacement", sector_coefficients["vector_keys"])
            self.assertIn("buried_polar_support", sector_coefficients["vector_keys"])
            self.assertIn("desolvation_risk", sector_coefficients["items"][0]["features"])

            fragment_features = json.loads((grow_dir / "fragment_features.json").read_text())
            self.assertIn("water_displacement", fragment_features["vector_keys"])
            self.assertIn("polar_desolvation_cost", fragment_features["items"][0]["features"])

            inspect_rc = main(
                [
                    "inspect",
                    "--grow-dir",
                    str(grow_dir),
                    "--top",
                    "2",
                    "--json-out",
                    str(grow_dir / "inspection.json"),
                ]
            )
            self.assertEqual(inspect_rc, 0)
            inspection = json.loads((grow_dir / "inspection.json").read_text())
            self.assertEqual(inspection["schema"], "pocketfield.inspection_report.v1")
            self.assertEqual(len(inspection["top_sectors"]), 2)
            self.assertIn("top_fragment_matches", inspection)
            self.assertNotIn("top_linker_matches", inspection)
            self.assertIn("candidate_summary", inspection)

    def test_design_accepts_custom_fragment_library_when_rdkit_available(self) -> None:
        try:
            import rdkit  # noqa: F401
        except ImportError:
            self.skipTest("RDKit is not available in this Python environment.")

        with tempfile.TemporaryDirectory() as tmp:
            rc = main(
                [
                    "design",
                    "--protein",
                    str(ROOT / "examples" / "tiny_receptor.pdb"),
                    "--ligand",
                    str(ROOT / "examples" / "tiny_ligand.pdb"),
                    "--out-dir",
                    tmp,
                    "--samples",
                    "64",
                    "--sectors",
                    "12",
                    "--shells",
                    "2.5,3.5",
                    "--top-k",
                    "4",
                    "--max-candidates",
                    "5",
                    "--top-sectors",
                    "3",
                    "--fragments-per-sector",
                    "2",
                    "--growth-depth",
                    "3",
                    "--request-limit",
                    "80",
                    "--max-clash-score",
                    "1000",
                    "--max-field-score",
                    "10000",
                    "--fragment-library",
                    str(ROOT / "examples" / "fragments.csv"),
                ]
            )
            self.assertEqual(rc, 0)

            candidates = json.loads((Path(tmp) / "grow" / "candidates.json").read_text())
            self.assertEqual(candidates["fragment_library"], str(ROOT / "examples" / "fragments.csv"))
            self.assertGreater(candidates["candidate_count"], 0)
            self.assertTrue((Path(tmp) / "grow" / "sector_coefficients.json").exists())

    def test_design_accepts_smiles_only_vendor_csv_when_rdkit_available(self) -> None:
        try:
            import rdkit  # noqa: F401
        except ImportError:
            self.skipTest("RDKit is not available in this Python environment.")

        with tempfile.TemporaryDirectory() as tmp:
            rc = main(
                [
                    "design",
                    "--protein",
                    str(ROOT / "examples" / "tiny_receptor.pdb"),
                    "--ligand",
                    str(ROOT / "examples" / "tiny_ligand.pdb"),
                    "--out-dir",
                    tmp,
                    "--samples",
                    "64",
                    "--sectors",
                    "12",
                    "--shells",
                    "2.5,3.5",
                    "--top-k",
                    "4",
                    "--max-candidates",
                    "5",
                    "--top-sectors",
                    "3",
                    "--fragments-per-sector",
                    "2",
                    "--max-clash-score",
                    "1000",
                    "--max-field-score",
                    "10000",
                    "--fragment-library",
                    str(ROOT / "examples" / "fragments_smiles_only.csv"),
                ]
            )
            self.assertEqual(rc, 0)

            candidates = json.loads((Path(tmp) / "grow" / "candidates.json").read_text())
            self.assertEqual(
                candidates["fragment_library"],
                str(ROOT / "examples" / "fragments_smiles_only.csv"),
            )
            self.assertGreater(candidates["candidate_count"], 0)

    def test_grow_uses_anchor_structure_dummy_sector_assignments_when_rdkit_available(self) -> None:
        try:
            import rdkit  # noqa: F401
        except ImportError:
            self.skipTest("RDKit is not available in this Python environment.")

        with tempfile.TemporaryDirectory() as tmp:
            build_dir = Path(tmp) / "build"
            grow_dir = Path(tmp) / "grow"
            build_rc = main(
                [
                    "build",
                    "--protein",
                    str(ROOT / "examples" / "tiny_receptor.pdb"),
                    "--ligand",
                    str(ROOT / "examples" / "tiny_ligand.pdb"),
                    "--out-dir",
                    str(build_dir),
                    "--samples",
                    "64",
                    "--sectors",
                    "12",
                    "--shells",
                    "2.5,3.5",
                    "--top-k",
                    "6",
                ]
            )
            self.assertEqual(build_rc, 0)

            grow_rc = main(
                [
                    "grow",
                    "--field",
                    str(build_dir / "field.npz"),
                    "--plan",
                    str(build_dir / "growth_plan.json"),
                    "--metadata",
                    str(build_dir / "metadata.json"),
                    "--out-dir",
                    str(grow_dir),
                    "--anchor-structure",
                    str(ROOT / "examples" / "anchor_dummy.sdf"),
                    "--sectors-per-dummy",
                    "2",
                    "--max-candidates",
                    "5",
                    "--top-sectors",
                    "4",
                    "--fragments-per-sector",
                    "2",
                    "--max-clash-score",
                    "1000",
                    "--max-field-score",
                    "10000",
                ]
            )
            self.assertEqual(grow_rc, 0)

            assignments = json.loads((grow_dir / "dummy_sector_assignments.json").read_text())
            self.assertEqual(assignments["schema"], "pocketfield.dummy_sector_assignments.v1")
            self.assertGreaterEqual(len(assignments["assignments"]), 4)
            candidates = json.loads((grow_dir / "candidates.json").read_text())
            self.assertGreater(candidates["candidate_count"], 0)
            self.assertIn("dummy_sector_angles", candidates["candidates"][0])

    def test_grow_accepts_disconnected_mapped_anchor_when_rdkit_available(self) -> None:
        try:
            import rdkit  # noqa: F401
        except ImportError:
            self.skipTest("RDKit is not available in this Python environment.")

        with tempfile.TemporaryDirectory() as tmp:
            build_dir = Path(tmp) / "build"
            grow_dir = Path(tmp) / "grow"
            self.assertEqual(
                main(
                    [
                        "build",
                        "--protein",
                        str(ROOT / "examples" / "tiny_receptor.pdb"),
                        "--ligand",
                        str(ROOT / "examples" / "tiny_ligand.pdb"),
                        "--out-dir",
                        str(build_dir),
                        "--samples",
                        "64",
                        "--sectors",
                        "12",
                        "--shells",
                        "2.5,3.5",
                        "--top-k",
                        "4",
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "grow",
                        "--field",
                        str(build_dir / "field.npz"),
                        "--plan",
                        str(build_dir / "growth_plan.json"),
                        "--metadata",
                        str(build_dir / "metadata.json"),
                        "--out-dir",
                        str(grow_dir),
                        "--anchor-smiles",
                        "[*:1]c1ccccc1.[*:2]CC",
                        "--max-candidates",
                        "4",
                        "--top-sectors",
                        "3",
                        "--fragments-per-sector",
                        "2",
                        "--max-clash-score",
                        "1000",
                        "--max-field-score",
                        "10000",
                    ]
                ),
                0,
            )

            candidates = json.loads((grow_dir / "candidates.json").read_text())
            self.assertGreater(candidates["candidate_count"], 0)
            self.assertIn(".", candidates["anchor_smiles"])

    def test_bridge_mode_connects_disconnected_mapped_anchor_when_rdkit_available(self) -> None:
        try:
            import rdkit  # noqa: F401
        except ImportError:
            self.skipTest("RDKit is not available in this Python environment.")

        with tempfile.TemporaryDirectory() as tmp:
            build_dir = Path(tmp) / "build"
            grow_dir = Path(tmp) / "grow"
            self.assertEqual(
                main(
                    [
                        "build",
                        "--protein",
                        str(ROOT / "examples" / "tiny_receptor.pdb"),
                        "--ligand",
                        str(ROOT / "examples" / "tiny_ligand.pdb"),
                        "--out-dir",
                        str(build_dir),
                        "--samples",
                        "64",
                        "--sectors",
                        "12",
                        "--shells",
                        "2.5,3.5",
                        "--top-k",
                        "4",
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "grow",
                        "--field",
                        str(build_dir / "field.npz"),
                        "--plan",
                        str(build_dir / "growth_plan.json"),
                        "--metadata",
                        str(build_dir / "metadata.json"),
                        "--out-dir",
                        str(grow_dir),
                        "--anchor-smiles",
                        "[*:1]c1ccccc1.[*:2]CC",
                        "--bridge-anchor-dummies",
                        "--linker-library",
                        str(ROOT / "examples" / "linkers_smiles_only.csv"),
                        "--max-candidates",
                        "4",
                        "--top-sectors",
                        "3",
                        "--fragments-per-sector",
                        "2",
                        "--max-clash-score",
                        "1000",
                        "--max-field-score",
                        "10000",
                    ]
                ),
                0,
            )

            candidates = json.loads((grow_dir / "candidates.json").read_text())
            self.assertGreater(candidates["candidate_count"], 0)
            self.assertEqual(candidates["candidates"][0]["mode"], "bridge")
            self.assertNotIn(".", candidates["candidates"][0]["smiles"])
            self.assertGreater(len(candidates["candidates"][0]["linkers"]), 0)

    def test_bridge_mode_records_linker_fit_rmsd_when_rdkit_available(self) -> None:
        try:
            import rdkit  # noqa: F401
        except ImportError:
            self.skipTest("RDKit is not available in this Python environment.")

        # Two disconnected one-heavy-atom anchors ([*:1]C.[*:2]C) 4.5 A apart
        # with their dummies pointing toward each other, so the linker closes
        # the gap between the fixed anchor points and its fit is measurable
        # against the anchored dummy coordinates.
        anchor_sdf = (
            "  bridge_anchor\n"
            "     RDKit          3D\n\n"
            "  4  2  0  0  0  0  0  0  0  0999 V2000\n"
            "    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
            "    1.5000    0.0000    0.0000 R   0  0  0  0  0  0  0  0  0  0  0  0\n"
            "    4.5000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
            "    3.0000    0.0000    0.0000 R   0  0  0  0  0  0  0  0  0  0  0  0\n"
            "  1  2  1  0\n"
            "  3  4  1  0\n"
            "M  ISO  2   2   1   4   2\n"
            "M  END\n"
            "$$$$\n"
        )

        with tempfile.TemporaryDirectory() as tmp:
            build_dir = Path(tmp) / "build"
            grow_dir = Path(tmp) / "grow"
            anchor_sdf_path = Path(tmp) / "bridge_anchor.sdf"
            anchor_sdf_path.write_text(anchor_sdf)
            self.assertEqual(
                main(
                    [
                        "build",
                        "--protein",
                        str(ROOT / "examples" / "tiny_receptor.pdb"),
                        "--ligand",
                        str(ROOT / "examples" / "tiny_ligand.pdb"),
                        "--out-dir",
                        str(build_dir),
                        "--samples",
                        "64",
                        "--sectors",
                        "12",
                        "--shells",
                        "2.5,3.5",
                        "--top-k",
                        "4",
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "grow",
                        "--field",
                        str(build_dir / "field.npz"),
                        "--plan",
                        str(build_dir / "growth_plan.json"),
                        "--metadata",
                        str(build_dir / "metadata.json"),
                        "--out-dir",
                        str(grow_dir),
                        "--anchor-smiles",
                        "[*:1]C.[*:2]C",
                        "--anchor-structure",
                        str(anchor_sdf_path),
                        "--bridge-anchor-dummies",
                        "--linker-library",
                        str(ROOT / "examples" / "linkers.csv"),
                        "--max-candidates",
                        "4",
                        "--top-sectors",
                        "3",
                        "--fragments-per-sector",
                        "2",
                        "--max-clash-score",
                        "1000",
                        "--max-field-score",
                        "10000",
                    ]
                ),
                0,
            )

            candidates = json.loads((grow_dir / "candidates.json").read_text())
            self.assertGreater(candidates["candidate_count"], 0)
            self.assertEqual(candidates["candidates"][0]["mode"], "bridge")
            linker_geometry = candidates["candidates"][0]["linker_geometry"]
            self.assertIsNotNone(linker_geometry)
            self.assertIn("fit_rmsd", linker_geometry)
            self.assertIsInstance(linker_geometry["fit_rmsd"], (int, float))

    def test_bridge_mode_flexible_linker_fit_when_rdkit_available(self) -> None:
        try:
            import rdkit  # noqa: F401
        except ImportError:
            self.skipTest("RDKit is not available in this Python environment.")

        # A 6 A anchor gap whose dummies point toward each other: short linkers
        # cannot close it under the rigid fit, so the flexible fit (which lets
        # the second component move) must recover them instead of rejecting them
        # at embedding.
        anchor_sdf = (
            "  bridge_anchor\n"
            "     RDKit          3D\n\n"
            "  4  2  0  0  0  0  0  0  0  0999 V2000\n"
            "    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
            "    1.5000    0.0000    0.0000 R   0  0  0  0  0  0  0  0  0  0  0  0\n"
            "    6.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
            "    4.5000    0.0000    0.0000 R   0  0  0  0  0  0  0  0  0  0  0  0\n"
            "  1  2  1  0\n"
            "  3  4  1  0\n"
            "M  ISO  2   2   1   4   2\n"
            "M  END\n"
            "$$$$\n"
        )

        with tempfile.TemporaryDirectory() as tmp:
            build_dir = Path(tmp) / "build"
            grow_dir = Path(tmp) / "grow"
            anchor_sdf_path = Path(tmp) / "bridge_anchor.sdf"
            anchor_sdf_path.write_text(anchor_sdf)
            self.assertEqual(
                main(
                    [
                        "build",
                        "--protein",
                        str(ROOT / "examples" / "tiny_receptor.pdb"),
                        "--ligand",
                        str(ROOT / "examples" / "tiny_ligand.pdb"),
                        "--out-dir",
                        str(build_dir),
                        "--samples",
                        "64",
                        "--sectors",
                        "12",
                        "--shells",
                        "2.5,3.5",
                        "--top-k",
                        "4",
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "grow",
                        "--field",
                        str(build_dir / "field.npz"),
                        "--plan",
                        str(build_dir / "growth_plan.json"),
                        "--metadata",
                        str(build_dir / "metadata.json"),
                        "--out-dir",
                        str(grow_dir),
                        "--anchor-smiles",
                        "[*:1]C.[*:2]C",
                        "--anchor-structure",
                        str(anchor_sdf_path),
                        "--bridge-anchor-dummies",
                        "--linker-library",
                        str(ROOT / "examples" / "linkers.csv"),
                        "--linker-fit-mode",
                        "flexible",
                        "--max-candidates",
                        "10",
                        "--top-sectors",
                        "3",
                        "--fragments-per-sector",
                        "2",
                        "--max-clash-score",
                        "1000",
                        "--max-field-score",
                        "10000",
                    ]
                ),
                0,
            )

            candidates = json.loads((grow_dir / "candidates.json").read_text())
            self.assertGreater(candidates["candidate_count"], 0)
            self.assertNotIn("embedding_failed", candidates["rejections"])

    def test_grow_pose_refine_completes_when_rdkit_available(self) -> None:
        try:
            import rdkit  # noqa: F401
        except ImportError:
            self.skipTest("RDKit is not available in this Python environment.")

        with tempfile.TemporaryDirectory() as tmp:
            build_dir = Path(tmp) / "build"
            grow_dir = Path(tmp) / "grow"
            build_rc = main(
                [
                    "build",
                    "--protein",
                    str(ROOT / "examples" / "tiny_receptor.pdb"),
                    "--ligand",
                    str(ROOT / "examples" / "tiny_ligand.pdb"),
                    "--out-dir",
                    str(build_dir),
                    "--samples",
                    "64",
                    "--sectors",
                    "12",
                    "--shells",
                    "2.5,3.5",
                    "--top-k",
                    "4",
                ]
            )
            self.assertEqual(build_rc, 0)

            grow_rc = main(
                [
                    "grow",
                    "--field",
                    str(build_dir / "field.npz"),
                    "--plan",
                    str(build_dir / "growth_plan.json"),
                    "--metadata",
                    str(build_dir / "metadata.json"),
                    "--out-dir",
                    str(grow_dir),
                    "--max-candidates",
                    "5",
                    "--top-sectors",
                    "3",
                    "--fragments-per-sector",
                    "2",
                    "--max-clash-score",
                    "1000",
                    "--max-field-score",
                    "10000",
                    "--pose-refine",
                ]
            )
            self.assertEqual(grow_rc, 0)

            candidates = json.loads((grow_dir / "candidates.json").read_text())
            self.assertGreater(candidates["candidate_count"], 0)
            scores = [c["score"] for c in candidates["candidates"]]
            self.assertEqual(scores, sorted(scores))

    def test_grow_diversity_filter_when_rdkit_available(self) -> None:
        try:
            import rdkit  # noqa: F401
        except ImportError:
            self.skipTest("RDKit is not available in this Python environment.")

        with tempfile.TemporaryDirectory() as tmp:
            build_dir = Path(tmp) / "build"
            grow_dir = Path(tmp) / "grow"
            build_rc = main(
                [
                    "build",
                    "--protein",
                    str(ROOT / "examples" / "tiny_receptor.pdb"),
                    "--ligand",
                    str(ROOT / "examples" / "tiny_ligand.pdb"),
                    "--out-dir",
                    str(build_dir),
                    "--samples",
                    "64",
                    "--sectors",
                    "12",
                    "--shells",
                    "2.5,3.5",
                    "--top-k",
                    "4",
                ]
            )
            self.assertEqual(build_rc, 0)

            grow_rc = main(
                [
                    "grow",
                    "--field",
                    str(build_dir / "field.npz"),
                    "--plan",
                    str(build_dir / "growth_plan.json"),
                    "--metadata",
                    str(build_dir / "metadata.json"),
                    "--out-dir",
                    str(grow_dir),
                    "--max-candidates",
                    "5",
                    "--top-sectors",
                    "3",
                    "--fragments-per-sector",
                    "2",
                    "--max-clash-score",
                    "1000",
                    "--max-field-score",
                    "10000",
                    "--diversity-threshold",
                    "0.7",
                ]
            )
            self.assertEqual(grow_rc, 0)

            candidates = json.loads((grow_dir / "candidates.json").read_text())
            self.assertGreater(candidates["candidate_count"], 0)
            self.assertEqual(candidates["filters"]["diversity_threshold"], 0.7)

    def test_grow_rejects_unmapped_disconnected_anchor_when_rdkit_available(self) -> None:
        try:
            import rdkit  # noqa: F401
        except ImportError:
            self.skipTest("RDKit is not available in this Python environment.")

        with tempfile.TemporaryDirectory() as tmp:
            build_dir = Path(tmp) / "build"
            grow_dir = Path(tmp) / "grow"
            self.assertEqual(
                main(
                    [
                        "build",
                        "--protein",
                        str(ROOT / "examples" / "tiny_receptor.pdb"),
                        "--ligand",
                        str(ROOT / "examples" / "tiny_ligand.pdb"),
                        "--out-dir",
                        str(build_dir),
                        "--samples",
                        "32",
                        "--sectors",
                        "8",
                        "--shells",
                        "2.5",
                        "--top-k",
                        "3",
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "grow",
                        "--field",
                        str(build_dir / "field.npz"),
                        "--plan",
                        str(build_dir / "growth_plan.json"),
                        "--metadata",
                        str(build_dir / "metadata.json"),
                        "--out-dir",
                        str(grow_dir),
                        "--anchor-smiles",
                        "*C.*N",
                    ]
                ),
                2,
            )


if __name__ == "__main__":
    unittest.main()
