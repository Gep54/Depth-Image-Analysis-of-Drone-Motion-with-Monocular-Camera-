"""Map planar (x, y) sync measurements onto an ordered frame list."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd


def sync_xy_weight_per_frame(
    frame_names: Sequence[str],
    sync_df: pd.DataFrame,
    *,
    image_col: str = "image",
    x_col: str = "x",
    y_col: str = "y",
    weight_col: str | None = None,
    sigma_col: str | None = None,
    default_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-frame planar prior targets and weights (z is not constrained).

    Rows with no match get weight 0 and NaN in ``xy_prior``.
    """
    lookup: dict[str, tuple[float, float, float]] = {}
    for _, row in sync_df.iterrows():
        key_raw = row[image_col]
        if pd.isna(key_raw):
            continue
        key = str(key_raw)
        x = float(row[x_col])
        y = float(row[y_col])
        if sigma_col is not None and sigma_col in row.index and pd.notna(row[sigma_col]):
            sigma = float(row[sigma_col])
            if sigma <= 0:
                raise ValueError(f"sigma must be positive, got {sigma}")
            w = 1.0 / (sigma**2)
        elif weight_col is not None and weight_col in row.index and pd.notna(row[weight_col]):
            w = float(row[weight_col])
        else:
            w = default_weight
        if w < 0:
            raise ValueError("Prior weight must be non-negative")
        lookup[key] = (x, y, w)
        lookup.setdefault(Path(key).name, (x, y, w))

    n = len(frame_names)
    xy_prior = np.full((n, 2), np.nan, dtype=np.float64)
    weight = np.zeros(n, dtype=np.float64)

    for i, name in enumerate(frame_names):
        hit = None
        for c in (name, str(Path(name)), Path(name).name):
            if c in lookup:
                hit = lookup[c]
                break
        if hit is None:
            continue
        xa, ya, wi = hit
        xy_prior[i, 0] = xa
        xy_prior[i, 1] = ya
        weight[i] = wi

    return xy_prior, weight
