"""Step 3: incremental registration (PnP + triangulation) and optional BA with (x,y) sync priors."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd

from data import undistort_image_bgr
from sequence_match import match_descriptors
from sync_priors import sync_xy_weight_per_frame
from two_view import reconstruct_two_view


@dataclass
class IncrementalSfMResult:
    frame_names: list[str]
    rvecs: np.ndarray
    tvecs: np.ndarray
    points_3d: np.ndarray
    colors_bgr: np.ndarray
    observations: np.ndarray
    pnp_inliers_last_frame: int
    bundle_cost: float | None


def _prepare_image(
    bgr: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    undistort: bool,
) -> np.ndarray:
    if undistort:
        return undistort_image_bgr(bgr, K, dist)
    return bgr


def _projection_matrix(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return K @ np.hstack([R, t.reshape(3, 1)])


def _triangulate_stereo(
    K: np.ndarray,
    rvec1: np.ndarray,
    tvec1: np.ndarray,
    rvec2: np.ndarray,
    tvec2: np.ndarray,
    uv1_nx2: np.ndarray,
    uv2_nx2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate N correspondences; return Nx3 points and boolean keep mask."""
    R1, _ = cv.Rodrigues(np.asarray(rvec1, dtype=np.float64).reshape(3, 1))
    R2, _ = cv.Rodrigues(np.asarray(rvec2, dtype=np.float64).reshape(3, 1))
    t1 = np.asarray(tvec1, dtype=np.float64).reshape(3)
    t2 = np.asarray(tvec2, dtype=np.float64).reshape(3)
    P1 = _projection_matrix(K, R1, t1)
    P2 = _projection_matrix(K, R2, t2)

    if uv1_nx2.size == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros(0, dtype=bool)

    pts_h = cv.triangulatePoints(P1, P2, uv1_nx2.T, uv2_nx2.T)
    w = pts_h[3, :]
    ok_w = np.abs(w) > 1e-10
    xyz = np.zeros((pts_h.shape[1], 3), dtype=np.float64)
    xyz[ok_w] = (pts_h[:3, ok_w] / w[ok_w]).T

    keep = np.zeros(pts_h.shape[1], dtype=bool)
    for k in range(pts_h.shape[1]):
        if not ok_w[k]:
            continue
        X = xyz[k]
        x1 = R1 @ X + t1
        x2 = R2 @ X + t2
        keep[k] = x1[2] > 0 and x2[2] > 0
    return xyz, keep


def run_incremental_sfm(
    frames: Sequence[tuple[str, np.ndarray]],
    K: np.ndarray,
    dist: np.ndarray,
    *,
    feature_extractor: Callable[[np.ndarray], tuple[list, np.ndarray | None]],
    preprocess: Callable[[np.ndarray], np.ndarray],
    feature_name: str,
    undistort: bool = True,
    two_view_pair_index: int = 0,
    ransac_prob: float = 0.999,
    ransac_threshold_px: float = 1.0,
    pnp_reproj_threshold: float = 8.0,
    sync_df: pd.DataFrame | None = None,
    sync_default_weight: float = 1.0,
    run_bundle_adjustment: bool = True,
    ba_max_nfev: int | None = None,
    ba_verbose: int = 0,
) -> IncrementalSfMResult:
    """Seed with two-view on frames ``(k, k+1)``, then PnP + stereo for the rest.

    World frame is **camera k** (identity). Only **(x, y)** of each camera center
    can be pulled toward sync in bundle adjustment; **z** is never penalized.

    Parameters
    ----------
    two_view_pair_index
        Index ``k`` of the first seed frame; must be ``0`` (first camera is world origin).
    sync_df
        Optional table with columns ``image``, ``x``, ``y``, and optionally ``weight`` or ``sigma``.
    """
    if two_view_pair_index != 0:
        raise ValueError("two_view_pair_index must be 0 (world = first frame); other values are not supported yet")

    n = len(frames)
    if n < 2:
        raise ValueError("Need at least two frames")

    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    dist = np.asarray(dist, dtype=np.float64).reshape(-1, 1)

    names = [fn for fn, _ in frames]

    img0 = _prepare_image(frames[0][1], K, dist, undistort)
    img1 = _prepare_image(frames[1][1], K, dist, undistort)

    tv = reconstruct_two_view(
        img0,
        img1,
        K,
        dist,
        feature_extractor=feature_extractor,
        preprocess=preprocess,
        feature_name=feature_name,
        undistort=False,
        ransac_prob=ransac_prob,
        ransac_threshold_px=ransac_threshold_px,
    )

    rvec1, _ = cv.Rodrigues(tv.R.astype(np.float64))

    rvecs = np.zeros((n, 3), dtype=np.float64)
    tvecs = np.zeros((n, 3), dtype=np.float64)
    rvecs[1] = rvec1.ravel()
    tvecs[1] = tv.t.astype(np.float64).reshape(3)

    points_3d = tv.points_3d.copy()
    colors_bgr = tv.colors_bgr.copy()
    n_pts = points_3d.shape[0]

    kp_pid: list[dict[int, int]] = [{} for _ in range(n)]
    for k in range(n_pts):
        kp_pid[0][int(tv.inlier_kp_idx0[k])] = k
        kp_pid[1][int(tv.inlier_kp_idx1[k])] = k

    obs: list[list[float]] = []
    for k in range(n_pts):
        obs.append([0, k, float(tv.inlier_uv0[k, 0]), float(tv.inlier_uv0[k, 1])])
        obs.append([1, k, float(tv.inlier_uv1[k, 0]), float(tv.inlier_uv1[k, 1])])

    pnp_inliers_last = 0
    zero_dist = np.zeros(5, dtype=np.float64)

    for i in range(2, n):
        prev = i - 1
        img_prev = _prepare_image(frames[prev][1], K, dist, undistort)
        img_curr = _prepare_image(frames[i][1], K, dist, undistort)

        proc_p = preprocess(img_prev)
        proc_c = preprocess(img_curr)
        kp_p, d_p = feature_extractor(proc_p)
        kp_c, d_c = feature_extractor(proc_c)
        matches = match_descriptors(d_c, d_p, feature_name)

        obj_list: list[np.ndarray] = []
        img_list: list[tuple[float, float]] = []
        pid_list: list[int] = []

        for m in matches:
            pid = kp_pid[prev].get(int(m.trainIdx))
            if pid is None:
                continue
            obj_list.append(points_3d[pid])
            img_list.append(kp_c[int(m.queryIdx)].pt)
            pid_list.append(pid)

        if len(obj_list) < 6:
            raise RuntimeError(
                f"Frame {i} ({names[i]}): need ≥6 3D–2D correspondences for PnP; got {len(obj_list)}"
            )

        obj_pts = np.asarray(obj_list, dtype=np.float64).reshape(-1, 3)
        img_pts = np.asarray(img_list, dtype=np.float64).reshape(-1, 1, 2)

        ok, rvec_i, tvec_i, inliers = cv.solvePnPRansac(
            obj_pts.astype(np.float32),
            img_pts.astype(np.float32),
            K,
            zero_dist,
            reprojectionError=float(pnp_reproj_threshold),
            confidence=0.999,
        )
        if not ok or inliers is None or len(inliers) < 6:
            raise RuntimeError(f"PnP failed for frame {i} ({names[i]})")

        rvec_i = rvec_i.reshape(3)
        tvec_i = tvec_i.reshape(3)
        rvecs[i] = rvec_i
        tvecs[i] = tvec_i
        pnp_inliers_last = int(len(inliers))

        for row in inliers.ravel():
            pid = pid_list[int(row)]
            u, v = img_list[int(row)]
            obs.append([i, pid, float(u), float(v)])

        R_prev, _ = cv.Rodrigues(rvecs[prev].reshape(3, 1))
        t_prev = tvecs[prev].reshape(3)
        R_curr, _ = cv.Rodrigues(rvec_i.reshape(3, 1))
        t_curr = tvec_i.reshape(3)

        new_u_prev: list[tuple[float, float]] = []
        new_u_curr: list[tuple[float, float]] = []
        new_train: list[int] = []
        new_query: list[int] = []

        for m in matches:
            tid = int(m.trainIdx)
            qid = int(m.queryIdx)
            if tid in kp_pid[prev]:
                continue
            new_u_prev.append(kp_p[tid].pt)
            new_u_curr.append(kp_c[qid].pt)
            new_train.append(tid)
            new_query.append(qid)

        if new_u_prev:
            uv_p = np.asarray(new_u_prev, dtype=np.float64)
            uv_c = np.asarray(new_u_curr, dtype=np.float64)
            xyz_new, keep = _triangulate_stereo(
                K, rvecs[prev], tvecs[prev], rvec_i, tvec_i, uv_p, uv_c
            )
            h, wim = img_prev.shape[:2]
            for j in range(len(keep)):
                if not keep[j]:
                    continue
                new_id = points_3d.shape[0]
                points_3d = np.vstack([points_3d, xyz_new[j : j + 1]])
                yi = int(round(np.clip(new_u_prev[j][1], 0, h - 1)))
                xi = int(round(np.clip(new_u_prev[j][0], 0, wim - 1)))
                colors_bgr = np.vstack(
                    [colors_bgr, img_prev[yi, xi].astype(np.uint8).reshape(1, 3)]
                )
                kp_pid[prev][new_train[j]] = new_id
                kp_pid[i][new_query[j]] = new_id
                obs.append([prev, new_id, float(new_u_prev[j][0]), float(new_u_prev[j][1])])
                obs.append([i, new_id, float(new_u_curr[j][0]), float(new_u_curr[j][1])])

    observations = np.asarray(obs, dtype=np.float64)
    bundle_cost: float | None = None

    if run_bundle_adjustment:
        try:
            from bundle_adjust_xy import bundle_adjust_multiview_xy_priors
        except ImportError as e:
            raise ImportError(
                "Bundle adjustment requires scipy. Install with `pip install scipy` "
                "or pass run_bundle_adjustment=False / CLI --no-bundle."
            ) from e

        if sync_df is None:
            xy_prior = np.full((n, 2), np.nan, dtype=np.float64)
            prior_w = np.zeros(n, dtype=np.float64)
        else:
            xy_prior, prior_w = sync_xy_weight_per_frame(
                names,
                sync_df,
                default_weight=sync_default_weight,
            )

        ba = bundle_adjust_multiview_xy_priors(
            observations,
            K,
            rvecs,
            tvecs,
            points_3d,
            xy_prior,
            prior_w,
            max_nfev=ba_max_nfev,
            verbose=ba_verbose,
        )
        rvecs = ba.rvecs
        tvecs = ba.tvecs
        points_3d = ba.points_3d
        bundle_cost = ba.cost

    return IncrementalSfMResult(
        frame_names=names,
        rvecs=rvecs,
        tvecs=tvecs,
        points_3d=points_3d,
        colors_bgr=colors_bgr,
        observations=observations,
        pnp_inliers_last_frame=pnp_inliers_last,
        bundle_cost=bundle_cost,
    )


def save_incremental_npz(path: Path | str, result: IncrementalSfMResult, K: np.ndarray) -> None:
    np.savez(
        Path(path),
        frame_names=np.asarray(result.frame_names, dtype=object),
        rvecs=result.rvecs,
        tvecs=result.tvecs,
        points_3d=result.points_3d,
        colors_bgr=result.colors_bgr,
        observations=result.observations,
        K=np.asarray(K, dtype=np.float64),
        bundle_cost=-1.0 if result.bundle_cost is None else result.bundle_cost,
    )


def load_incremental_npz(path: Path | str) -> tuple[IncrementalSfMResult, np.ndarray]:
    """Reload a bundle written by :func:`save_incremental_npz`.

    ``pnp_inliers_last_frame`` is set to ``-1`` (unknown when loading from disk).
    """
    z = np.load(Path(path), allow_pickle=True)
    fn = z["frame_names"]
    names = [str(x) for x in np.atleast_1d(fn).tolist()]
    bc = float(z["bundle_cost"])
    result = IncrementalSfMResult(
        frame_names=names,
        rvecs=np.asarray(z["rvecs"], dtype=np.float64),
        tvecs=np.asarray(z["tvecs"], dtype=np.float64),
        points_3d=np.asarray(z["points_3d"], dtype=np.float64),
        colors_bgr=np.asarray(z["colors_bgr"], dtype=np.uint8),
        observations=np.asarray(z["observations"], dtype=np.float64),
        pnp_inliers_last_frame=-1,
        bundle_cost=None if bc < 0 else bc,
    )
    K = np.asarray(z["K"], dtype=np.float64).reshape(3, 3)
    return result, K
