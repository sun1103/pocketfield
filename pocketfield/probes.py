from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Probe:
    name: str
    radius: float
    epsilon: float
    charge: float = 0.0
    donor: bool = False
    acceptor: bool = False
    hydrophobic: bool = False
    hbond_strength: float = 2.5
    hydrophobic_strength: float = 1.0

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "radius": self.radius,
            "epsilon": self.epsilon,
            "charge": self.charge,
            "donor": self.donor,
            "acceptor": self.acceptor,
            "hydrophobic": self.hydrophobic,
            "hbond_strength": self.hbond_strength,
            "hydrophobic_strength": self.hydrophobic_strength,
        }


PROBES: dict[str, Probe] = {
    "hydrophobic": Probe(
        name="hydrophobic",
        radius=1.70,
        epsilon=0.10,
        hydrophobic=True,
        hydrophobic_strength=1.4,
    ),
    "hbond_donor": Probe(
        name="hbond_donor",
        radius=1.55,
        epsilon=0.08,
        charge=0.25,
        donor=True,
        hbond_strength=3.0,
    ),
    "hbond_acceptor": Probe(
        name="hbond_acceptor",
        radius=1.52,
        epsilon=0.08,
        charge=-0.25,
        acceptor=True,
        hbond_strength=3.0,
    ),
    "positive": Probe(name="positive", radius=1.55, epsilon=0.06, charge=0.6),
    "negative": Probe(name="negative", radius=1.52, epsilon=0.06, charge=-0.6),
    "aromatic": Probe(
        name="aromatic",
        radius=1.75,
        epsilon=0.12,
        hydrophobic=True,
        hydrophobic_strength=1.0,
    ),
}


def get_probes(names: str) -> list[Probe]:
    selected: list[Probe] = []
    for name in [item.strip() for item in names.split(",") if item.strip()]:
        if name not in PROBES:
            available = ", ".join(sorted(PROBES))
            raise ValueError(f"Unknown probe '{name}'. Available probes: {available}")
        selected.append(PROBES[name])
    if not selected:
        raise ValueError("At least one probe is required.")
    return selected

