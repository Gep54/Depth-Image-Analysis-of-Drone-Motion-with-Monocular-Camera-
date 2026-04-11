"""Multi-view bundle adjustment with weighted priors on camera center (x, y) only."""

from __future__ import annotations

from dataclasses import dataclass

import cv2 as cv
import numpy as np
from scipy.optimize import least_squares


def camera_center_world(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    R, _ = cv.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    t = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    return (-R.T @ t).reshape(3)


@dataclass
class BundleAdjustXYResult:
    success: bool
    message: str
    rvecs: np.ndarray
    tvecs: np.ndarray
    points_3d: np.ndarray
    cost: float


def bundle_adjust_multiview_xy_priors(
    observations: np.ndarray,
    K: np.ndarray,
    rvecs: np.ndarray,
    tvecs: np.ndarray,
    points_3d: np.ndarray,
    xy_prior: np.ndarray,
    prior_weight: np.ndarray,
    *,
    max_nfev: int | None = None,
    verbose: int = 0,
) -> BundleAdjustXYResult:
    """Least squares: reprojection + sqrt(w)*(C_x - x_prior), sqrt(w)*(C_y - y_prior).

    Camera 0 pose is **fixed** (identity) to fix gauge. **Z** of each camera center
    is not penalized — only the horizontal components of ``C = -R^T t`` vs sync.

    Parameters
    ----------
    observations
        Shape ``(n_obs, 4)``: ``frame_idx, point_idx, u, v`` (pixels, undistorted).
    rvecs, tvecs
        Shape ``(n_frames, 3)``. Frame 0 must be zeros; only frames ``1..n-1`` are optimized.
    xy_prior
        Shape ``(n_frames, 2)``. NaN rows / zero weight skip priors.
    prior_weight
        Shape ``(n_frames,)``.
    """
    observations = np.asarray(observations, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    rvecs = np.asarray(rvecs, dtype=np.float64).reshape(-1, 3).copy()
    tvecs = np.asarray(tvecs, dtype=np.float64).reshape(-1, 3).copy()
    points_3d = np.asarray(points_3d, dtype=np.float64).reshape(-1, 3).copy()
    n_frames = rvecs.shape[0]
    n_points = points_3d.shape[0]

    xy_prior = np.asarray(xy_prior, dtype=np.float64).reshape(n_frames, 2)
    prior_weight = np.asarray(prior_weight, dtype=np.float64).reshape(n_frames)

    if not np.allclose(rvecs[0], 0) or not np.allclose(tvecs[0], 0):
        raise ValueError("Frame 0 must be identity (rvec=0, tvec=0) for gauge fix")

    fi = observations[:, 0].astype(int)
    pi = observations[:, 1].astype(int)
    uv = observations[:, 2:4]
    if np.any(fi < 0) or np.any(fi >= n_frames):
        raise ValueError("observations frame_idx out of range")
    if np.any(pi < 0) or np.any(pi >= n_points):
        raise ValueError("observations point_idx out of range")

    prior_mask = (
        (prior_weight > 0)
        & np.isfinite(xy_prior[:, 0])
        & np.isfinite(xy_prior[:, 1])
    )

    n_other = n_frames - 1
    slice_r = slice(0, 3 * n_other)
    slice_t = slice(3 * n_other, 6 * n_other)
    slice_x = slice(6 * n_other, 6 * n_other + 3 * n_points)

    def pack() -> np.ndarray:
        return np.concatenate(
            [
                rvecs[1:].reshape(-1),
                tvecs[1:].reshape(-1),
                points_3d.reshape(-1),
            ]
        )

    def unpack(p: np.ndarray) -> None:
        rvecs[1:] = p[slice_r].reshape(n_other, 3)
        tvecs[1:] = p[slice_t].reshape(n_other, 3)
        points_3d[:] = p[slice_x].reshape(n_points, 3)

    def residuals(p: np.ndarray) -> np.ndarray:
        unpack(p)
        res: list[float] = []
        for k in range(len(observations)):
            f = int(fi[k])
            j = int(pi[k])
            rv = rvecs[f]
            tv = tvecs[f]
            pt = points_3d[j].reshape(1, 1, 3)
            proj, _ = cv.projectPoints(pt, rv, tv, K, None)
            u, v = proj.reshape(2)
            res.append(u - uv[k, 0])
            res.append(v - uv[k, 1])

        for i in range(n_frames):
            if not prior_mask[i]:
                continue
            rv = rvecs[i]
            tv = tvecs[i]
            C = camera_center_world(rv, tv)
            sw = float(np.sqrt(prior_weight[i]))
            res.append(sw * (C[0] - xy_prior[i, 0]))
            res.append(sw * (C[1] - xy_prior[i, 1]))

        return np.asarray(res, dtype=np.float64)

    x0 = pack()
    if max_nfev is None:
        max_nfev = max(200, 15 * len(x0))

    ls = least_squares(
        residuals,
        x0,
        method="trf",
        max_nfev=max_nfev,
        verbose=verbose,
    )
    unpack(ls.x)

    return BundleAdjustXYResult(
        success=ls.success,
        message=ls.message,
        rvecs=rvecs,
        tvecs=tvecs,
        points_3d=points_3d,
        cost=float(ls.cost),
    )
