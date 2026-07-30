from __future__ import annotations

import numpy as np


def parse_float_list(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected a comma-separated list of floats.")
    return values


def parse_center(raw: str) -> np.ndarray:
    values = parse_float_list(raw)
    if len(values) != 3:
        raise ValueError("--center must have exactly three values: x,y,z")
    return np.array(values, dtype=np.float64)


def fibonacci_directions(n: int) -> np.ndarray:
    if n < 4:
        raise ValueError("At least 4 samples are required.")
    indices = np.arange(n, dtype=np.float64)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    z = 1.0 - (2.0 * indices + 1.0) / n
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = golden_angle * indices
    directions = np.column_stack((np.cos(theta) * radius, np.sin(theta) * radius, z))
    return directions.astype(np.float64)


def shell_points(center: np.ndarray, shells: list[float], directions: np.ndarray) -> np.ndarray:
    shell_array = np.asarray(shells, dtype=np.float64)
    if np.any(shell_array <= 0):
        raise ValueError("Shell radii must be positive.")
    return center[None, None, :] + shell_array[:, None, None] * directions[None, :, :]

