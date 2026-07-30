from __future__ import annotations

from datetime import datetime, timezone


def build_growth_plan(
    center: list[float],
    single_sectors: list[dict],
    sector_pairs: list[dict],
    anchor_radius: float,
) -> dict:
    return {
        "schema": "pocketfield.growth_plan.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "center": center,
        "anchor": {
            "type": "central_fragment",
            "radius": anchor_radius,
            "description": (
                "Place or identify an anchor fragment near center, then grow substituents "
                "toward ranked sector directions. Use pair entries for fragments spanning "
                "adjacent sectors."
            ),
        },
        "single_sector_growing": single_sectors,
        "paired_sector_growing": sector_pairs,
    }
