from __future__ import annotations

import numpy as np

from pocketfield.sphere import fibonacci_directions


def assign_sectors(directions: np.ndarray, n_sectors: int) -> tuple[np.ndarray, np.ndarray]:
    if n_sectors < 4:
        raise ValueError("At least 4 sectors are required.")
    centers = fibonacci_directions(n_sectors)
    dots = np.einsum("ij,kj->ik", directions, centers, optimize=False)
    sector_ids = np.argmax(dots, axis=1).astype(np.int32)
    return sector_ids, centers


def rank_single_sectors(
    energies: np.ndarray,
    gradients: np.ndarray,
    probe_names: list[str],
    shells: list[float],
    sector_ids: np.ndarray,
    sector_centers: np.ndarray,
    top_k: int = 20,
) -> list[dict]:
    ranked: list[dict] = []
    n_sectors = sector_centers.shape[0]
    for sector_id in range(n_sectors):
        mask = sector_ids == sector_id
        if not np.any(mask):
            continue
        sector_energies = energies[:, :, mask]
        probe_index, shell_index, local_index = np.unravel_index(
            int(np.argmin(sector_energies)),
            sector_energies.shape,
        )
        values = sector_energies[probe_index, shell_index]
        q = max(1, int(np.ceil(values.size * 0.25)))
        low_mean = float(np.mean(np.partition(values, q - 1)[:q]))
        sample_indices = np.flatnonzero(mask)
        sample_index = int(sample_indices[local_index])
        grad = gradients[probe_index, shell_index, sample_index]
        ranked.append(
            {
                "sector_id": int(sector_id),
                "score": low_mean,
                "min_energy": float(sector_energies[probe_index, shell_index, local_index]),
                "best_probe": probe_names[probe_index],
                "best_shell": float(shells[shell_index]),
                "direction": sector_centers[sector_id].tolist(),
                "gradient": grad.tolist(),
                "growth_vector": (-grad / max(np.linalg.norm(grad), 1e-8)).tolist(),
                "sample_count": int(mask.sum()),
            }
        )
    ranked.sort(key=lambda item: item["score"])
    return ranked[:top_k]


def rank_sector_pairs(
    energies: np.ndarray,
    probe_names: list[str],
    shells: list[float],
    sector_ids: np.ndarray,
    sector_centers: np.ndarray,
    top_k: int = 20,
    neighbors_per_sector: int = 6,
) -> list[dict]:
    del probe_names
    del shells
    pairs: list[dict] = []
    dots = np.einsum("ij,kj->ik", sector_centers, sector_centers, optimize=False)
    n_sectors = sector_centers.shape[0]
    seen: set[tuple[int, int]] = set()

    for sector_id in range(n_sectors):
        nearest = np.argsort(-dots[sector_id])[1 : neighbors_per_sector + 1]
        for other_id in nearest:
            pair = tuple(sorted((int(sector_id), int(other_id))))
            if pair in seen:
                continue
            seen.add(pair)
            mask = np.logical_or(sector_ids == pair[0], sector_ids == pair[1])
            if not np.any(mask):
                continue
            pair_energies = energies[:, :, mask]
            values = pair_energies.reshape(-1)
            q = max(1, int(np.ceil(values.size * 0.10)))
            score = float(np.mean(np.partition(values, q - 1)[:q]))
            direction = sector_centers[pair[0]] + sector_centers[pair[1]]
            direction = direction / max(np.linalg.norm(direction), 1e-8)
            pairs.append(
                {
                    "sector_ids": [pair[0], pair[1]],
                    "score": score,
                    "angle_deg": float(np.degrees(np.arccos(np.clip(dots[pair], -1.0, 1.0)))),
                    "direction": direction.tolist(),
                    "sample_count": int(mask.sum()),
                }
            )
    pairs.sort(key=lambda item: item["score"])
    return pairs[:top_k]
