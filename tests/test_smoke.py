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
            self.assertIn("top_linker_matches", inspection)
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
