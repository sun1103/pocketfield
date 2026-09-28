"""Implicit-solvent free energy (GBSA-OBC).

Onufriev-Bashford-Case generalized-Born with pairwise-descreened effective Born
radii, a Shrake-Rupley nonpolar surface term, and am1bcc (or any) partial
charges. Pure NumPy — no RDKit/OpenFF — so callers supply positions, atomic
numbers, and charges explicitly.

Example::

    from pocketfield.solvation import gbsa_obc
    dG, dG_el, dG_np = gbsa_obc(positions, atomic_nums, charges)

``dG`` is the total solvation free energy (kcal/mol); more negative = more
soluble.
"""

from __future__ import annotations

import numpy as np


# OBC VDW radii (Å).
OBC_VDW = {1: 1.20, 6: 1.70, 7: 1.55, 8: 1.50, 9: 1.50, 15: 1.80, 16: 1.80, 17: 1.70}
OBC_OFFSET = 0.09   # dielectric offset (Å)
OBC_ALPHA = 1.0     # OBC II parameters
OBC_BETA = 0.8
OBC_GAMMA = 4.85
EPS_SOLUTE = 1.0
EPS_SOLVENT = 78.5
SURFACE_TENSION = 0.005  # kcal/mol/Å²


def _obc_born_radii(positions, atomic_nums):
    """Effective Born radii via OBC-II (Onufriev-Bashford-Case) descreening.

    R_i^-1 = rho_tilde_i^-1 - rho_i^-1 * tanh(alpha*Psi - beta*Psi^2 + gamma*Psi^3)
    with rho_tilde = rho - 0.09 A, Psi = 0.5 * rho_tilde_i * sum_j term_ij and
    term_ij the pairwise HCT descreening integral. Matches OpenMM
    GBSAOBCForce: the offset radius is used for every atom in the descreening,
    the *full* radius is used in the tanh denominator, and all other atoms
    (no bond exclusion) contribute to the sum.
    """
    n = len(positions)
    radii = np.array([OBC_VDW.get(z, 1.70) for z in atomic_nums])
    rho = radii - OBC_OFFSET

    effective = np.zeros(n)
    for i in range(n):
        b = rho[i]
        s = 0.0
        for j in range(n):
            if i == j:
                continue
            a = rho[j]
            d = np.linalg.norm(positions[i] - positions[j])
            if d + a <= b:            # neighbour sphere inside atom i's sphere
                continue
            l_ij = 1.0 / max(b, abs(d - a))
            u_ij = 1.0 / (d + a)
            term = ((l_ij - u_ij)
                    + 0.25 * d * (u_ij * u_ij - l_ij * l_ij)
                    + 0.5 * (1.0 / d) * np.log(u_ij / l_ij)
                    + 0.25 * a * a * (1.0 / d) * (l_ij * l_ij - u_ij * u_ij))
            if b < a - d:             # atom i inside neighbour's sphere
                term += 2.0 * (1.0 / b - l_ij)
            s += term
        psi = 0.5 * b * s
        inv_R = (1.0 / b -
                 np.tanh(OBC_ALPHA * psi - OBC_BETA * psi**2 +
                         OBC_GAMMA * psi**3) / radii[i])
        effective[i] = 1.0 / max(inv_R, 1e-3)

    return effective


def _sasa(positions, atomic_nums, probe=1.4, n_points=512):
    """Solvent-accessible surface area (Å²) via Shrake-Rupley."""
    n = len(positions)
    radii = np.array([OBC_VDW.get(z, 1.70) for z in atomic_nums]) + probe
    idx = np.arange(n_points)
    z = 1.0 - 2.0 * (idx + 0.5) / n_points
    phi = idx * np.pi * (3.0 - np.sqrt(5.0))
    xy = np.sqrt(1.0 - z * z)
    dirs = np.stack([xy * np.cos(phi), xy * np.sin(phi), z], axis=1)
    area_per_point = 4.0 * np.pi / n_points

    total = 0.0
    for i in range(n):
        pts = positions[i] + radii[i] * dirs
        occluded = np.zeros(n_points, dtype=bool)
        for j in range(n):
            if i == j:
                continue
            occluded |= np.sum((pts - positions[j]) ** 2, axis=1) < radii[j] ** 2
        total += (n_points - occluded.sum()) * area_per_point * radii[i] ** 2
    return float(total)


def gbsa_obc(positions, atomic_nums, charges):
    """OBC-GBSA solvation free energy.

    Uses the Still formula with OBC effective Born radii:

        dG_el = -0.5 * (1/eps_in - 1/eps_out) * 332 * sum_i sum_j q_i q_j / f_GB

    Returns ``(dG_total, dG_el, dG_np)`` in kcal/mol.
    """
    n = len(positions)
    r_eff = _obc_born_radii(positions, atomic_nums)
    q = np.asarray(charges, dtype=np.float64)
    prefac = -0.5 * (1.0 / EPS_SOLUTE - 1.0 / EPS_SOLVENT) * 332.0

    g_el = 0.0
    for i in range(n):
        for j in range(i, n):
            d = np.linalg.norm(positions[i] - positions[j]) if i != j else 0.0
            f_gb = np.sqrt(d * d + r_eff[i] * r_eff[j] *
                           np.exp(-d * d / (4.0 * r_eff[i] * r_eff[j])))
            factor = 1.0 if i == j else 2.0  # self-term counted once, pairs twice
            g_el += prefac * factor * q[i] * q[j] / max(f_gb, 1e-6)

    g_np = SURFACE_TENSION * _sasa(positions, atomic_nums)
    return float(g_el + g_np), float(g_el), float(g_np)
