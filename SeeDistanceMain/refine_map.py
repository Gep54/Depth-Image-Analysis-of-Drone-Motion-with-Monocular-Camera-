"""Step 5: global robust refinement — re-run bundle adjustment on a saved map."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bundle_adjust_xy import bundle_adjust_multiview_xy_priors
from incremental_sfm import IncrementalSfMResult
from sync_priors import sync_xy_weight_per_frame


def refine_incremental_bundle(
    result: IncrementalSfMResult,
    K: np.ndarray,
    *,
    sync_df: pd.DataFrame | None = None,
    sync_default_weight: float = 1.0,
    loss: str = "soft_l1",
    f_scale: float = 1.5,
    max_nfev: int | None = None,
    verbose: int = 0,
) -> IncrementalSfMResult:
    """Second-pass joint optimization with optional robust loss.

    Reuses the same observation list and **(x, y)-only** planar priors as step 3.
    Camera 0 must remain identity (as produced by ``run_incremental_sfm``).

    Parameters
    ----------
    loss, f_scale
        Forwarded to :func:`bundle_adjust_xy.bundle_adjust_multiview_xy_priors`.
        ``soft_l1`` or ``huber`` down-weight large residuals (mis-tracked points,
        bad sync rows, etc.). **All** residuals use the same loss in SciPy.
    """
    r0, t0 = result.rvecs[0], result.tvecs[0]
    if not np.allclose(r0, 0) or not np.allclose(t0, 0):
        raise ValueError(
            "refine_incremental_bundle expects camera 0 at identity (rvec=tvec=0). "
            "Reload an incremental-sfm NPZ or fix the gauge before refining."
        )

    n = len(result.frame_names)
    if sync_df is None:
        xy_prior = np.full((n, 2), np.nan, dtype=np.float64)
        prior_w = np.zeros(n, dtype=np.float64)
    else:
        xy_prior, prior_w = sync_xy_weight_per_frame(
            result.frame_names,
            sync_df,
            default_weight=sync_default_weight,
        )

    rvecs = np.asarray(result.rvecs, dtype=np.float64).copy()
    tvecs = np.asarray(result.tvecs, dtype=np.float64).copy()
    points_3d = np.asarray(result.points_3d, dtype=np.float64).copy()
    observations = np.asarray(result.observations, dtype=np.float64).copy()

    ba = bundle_adjust_multiview_xy_priors(
        observations,
        K,
        rvecs,
        tvecs,
        points_3d,
        xy_prior,
        prior_w,
        max_nfev=max_nfev,
        verbose=verbose,
        loss=loss,
        f_scale=f_scale,
    )

    return IncrementalSfMResult(
        frame_names=list(result.frame_names),
        rvecs=ba.rvecs,
        tvecs=ba.tvecs,
        points_3d=ba.points_3d,
        colors_bgr=np.asarray(result.colors_bgr).copy(),
        observations=observations,
        pnp_inliers_last_frame=-1,
        bundle_cost=ba.cost,
    )
