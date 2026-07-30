from __future__ import annotations

import numpy as np

from pocketfield.atoms import Atom
from pocketfield.probes import Probe


def compute_field(
    pocket_atoms: list[Atom],
    points: np.ndarray,
    probes: list[Probe],
    dielectric: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return energies and Cartesian energy gradients for probes/shells/samples."""
    atom_coords = np.vstack([atom.coord for atom in pocket_atoms])
    atom_radii = np.array([atom.radius for atom in pocket_atoms], dtype=np.float64)
    atom_eps = np.array([atom.epsilon for atom in pocket_atoms], dtype=np.float64)
    atom_charges = np.array([atom.charge for atom in pocket_atoms], dtype=np.float64)
    atom_donor = np.array([atom.donor for atom in pocket_atoms], dtype=bool)
    atom_acceptor = np.array([atom.acceptor for atom in pocket_atoms], dtype=bool)
    atom_hydrophobic = np.array([atom.hydrophobic for atom in pocket_atoms], dtype=bool)

    flat_points = points.reshape(-1, 3)
    n_probes = len(probes)
    n_points = flat_points.shape[0]
    energies = np.zeros((n_probes, n_points), dtype=np.float64)
    gradients = np.zeros((n_probes, n_points, 3), dtype=np.float64)

    for probe_index, probe in enumerate(probes):
        energy, gradient = _probe_energy_gradient(
            flat_points,
            atom_coords,
            atom_radii,
            atom_eps,
            atom_charges,
            atom_donor,
            atom_acceptor,
            atom_hydrophobic,
            probe,
            dielectric,
        )
        energies[probe_index] = energy
        gradients[probe_index] = gradient

    shell_count, sample_count, _ = points.shape
    return (
        energies.reshape(n_probes, shell_count, sample_count),
        gradients.reshape(n_probes, shell_count, sample_count, 3),
    )


def _probe_energy_gradient(
    points: np.ndarray,
    atom_coords: np.ndarray,
    atom_radii: np.ndarray,
    atom_eps: np.ndarray,
    atom_charges: np.ndarray,
    atom_donor: np.ndarray,
    atom_acceptor: np.ndarray,
    atom_hydrophobic: np.ndarray,
    probe: Probe,
    dielectric: float,
) -> tuple[np.ndarray, np.ndarray]:
    energy = np.zeros(points.shape[0], dtype=np.float64)
    gradient = np.zeros((points.shape[0], 3), dtype=np.float64)
    chunk_size = 4096

    for start in range(0, points.shape[0], chunk_size):
        stop = min(start + chunk_size, points.shape[0])
        chunk = points[start:stop]
        diff = chunk[:, None, :] - atom_coords[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        safe_dist = np.maximum(dist, 0.75)
        unit = diff / safe_dist[:, :, None]

        sigma = probe.radius + atom_radii
        epsilon = np.sqrt(probe.epsilon * atom_eps)
        ratio = sigma[None, :] / safe_dist
        ratio6 = ratio**6
        ratio12 = ratio6**2

        lj_energy = epsilon[None, :] * (ratio12 - 2.0 * ratio6)
        d_lj_dr = epsilon[None, :] * (-12.0 * ratio12 / safe_dist + 12.0 * ratio6 / safe_dist)

        total_pair_energy = lj_energy
        total_d_dr = d_lj_dr

        if probe.charge != 0.0:
            coulomb = 332.0636 * probe.charge * atom_charges[None, :] / (dielectric * safe_dist)
            coulomb = np.clip(coulomb, -20.0, 20.0)
            d_coulomb_dr = -332.0636 * probe.charge * atom_charges[None, :] / (
                dielectric * safe_dist**2
            )
            active = atom_charges != 0.0
            total_pair_energy = total_pair_energy + np.where(active[None, :], coulomb, 0.0)
            total_d_dr = total_d_dr + np.where(active[None, :], d_coulomb_dr, 0.0)

        hbond_mask = np.zeros(atom_acceptor.shape, dtype=bool)
        if probe.donor:
            hbond_mask = np.logical_or(hbond_mask, atom_acceptor)
        if probe.acceptor:
            hbond_mask = np.logical_or(hbond_mask, atom_donor)
        if np.any(hbond_mask):
            hbond_energy, hbond_d_dr = _gaussian_well(
                safe_dist,
                center=sigma[None, :] + 0.15,
                width=0.45,
                strength=probe.hbond_strength,
            )
            total_pair_energy = total_pair_energy + np.where(hbond_mask[None, :], hbond_energy, 0.0)
            total_d_dr = total_d_dr + np.where(hbond_mask[None, :], hbond_d_dr, 0.0)

        if probe.hydrophobic and np.any(atom_hydrophobic):
            hydro_energy, hydro_d_dr = _gaussian_well(
                safe_dist,
                center=sigma[None, :] + 0.35,
                width=0.75,
                strength=probe.hydrophobic_strength,
            )
            total_pair_energy = total_pair_energy + np.where(
                atom_hydrophobic[None, :], hydro_energy, 0.0
            )
            total_d_dr = total_d_dr + np.where(atom_hydrophobic[None, :], hydro_d_dr, 0.0)

        energy[start:stop] = np.sum(total_pair_energy, axis=1)
        gradient[start:stop] = np.sum(total_d_dr[:, :, None] * unit, axis=1)

    return energy, gradient


def _gaussian_well(
    dist: np.ndarray,
    center: np.ndarray,
    width: float,
    strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    delta = dist - center
    exponent = -((delta / width) ** 2)
    well = -strength * np.exp(exponent)
    d_dr = well * (-2.0 * delta / (width**2))
    return well, d_dr
