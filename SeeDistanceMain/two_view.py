"""Two-view structure from motion: match, essential matrix, pose, triangulation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np

from data import undistort_image_bgr
from sequence_match import match_descriptors


@dataclass
class TwoViewReconstruction:
    """Sparse 3D points and relative pose (camera 2 w.r.t. camera 1)."""

    R: np.ndarray
    """Rotation 3×3: maps points from camera-1 frame to camera-2 frame (OpenCV ``recoverPose``)."""
    t: np.ndarray
    """Translation 3-vector, unit norm; reconstruction is up to this scale."""
    points_3d: np.ndarray
    """Inlier triangulated points in camera-1 world frame, shape ``(n, 3)``."""
    colors_bgr: np.ndarray
    """uint8 colors sampled from image 1, shape ``(n, 3)``."""
    inlier_mask: np.ndarray
    """Boolean mask over **all** matches; ``True`` where geometry + cheirality kept the point."""
    mean_reproj_error_px: float
    """Mean Euclidean reprojection error (pixels) over inliers."""
    n_inliers: int
    E: np.ndarray
    """Estimated essential matrix 3×3."""
    inlier_kp_idx0: np.ndarray
    """Keypoint index on image 1 for each triangulated point (same order as ``points_3d``)."""
    inlier_kp_idx1: np.ndarray
    """Keypoint index on image 2 for each triangulated point."""
    inlier_uv0: np.ndarray
    """Image-plane coordinates on view 1, shape ``(n_inliers, 2)``."""
    inlier_uv1: np.ndarray
    """Image-plane coordinates on view 2, shape ``(n_inliers, 2)``."""


def _matched_pixel_pairs(
    keypoints1: list,
    keypoints2: list,
    matches: list[cv.DMatch],
) -> tuple[np.ndarray, np.ndarray]:
    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    return pts1, pts2


def _reprojection_errors_px(
    points_3d_nx3: np.ndarray,
    pts1_nx2: np.ndarray,
    pts2_nx2: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Per-point combined reprojection RMSE across both views (pixels)."""
    rvec1 = np.zeros(3, dtype=np.float64)
    tvec1 = np.zeros(3, dtype=np.float64)
    rvec2, _ = cv.Rodrigues(R.astype(np.float64))
    tvec2 = t.astype(np.float64).reshape(3)

    X = points_3d_nx3.astype(np.float64).reshape(-1, 1, 3)
    proj1, _ = cv.projectPoints(X, rvec1, tvec1, K, None)
    proj2, _ = cv.projectPoints(X, rvec2, tvec2, K, None)
    e1 = np.linalg.norm(proj1.reshape(-1, 2) - pts1_nx2, axis=1)
    e2 = np.linalg.norm(proj2.reshape(-1, 2) - pts2_nx2, axis=1)
    return (e1 + e2) / 2.0


def reconstruct_two_view(
    image1_bgr: np.ndarray,
    image2_bgr: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    *,
    feature_extractor: Callable[[np.ndarray], tuple[list, np.ndarray | None]],
    preprocess: Callable[[np.ndarray], np.ndarray],
    feature_name: str,
    undistort: bool = True,
    ransac_prob: float = 0.999,
    ransac_threshold_px: float = 1.0,
) -> TwoViewReconstruction:
    """Estimate :math:`E`, relative pose, and triangulate inlier matches.

    Images are optionally undistorted with ``K``, ``dist`` before detection.
    All geometry uses the same pinhole ``K`` (undistorted coordinates).

    Returns
    -------
    TwoViewReconstruction
        Translation ``t`` is normalized; 3D points share the same arbitrary metric scale.
    """
    if image1_bgr.shape[:2] != image2_bgr.shape[:2]:
        raise ValueError("Images must have the same size for this two-view pipeline")

    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    dist = np.asarray(dist, dtype=np.float64).reshape(-1, 1)

    img1 = undistort_image_bgr(image1_bgr, K, dist) if undistort else image1_bgr
    img2 = undistort_image_bgr(image2_bgr, K, dist) if undistort else image2_bgr

    proc1 = preprocess(img1)
    proc2 = preprocess(img2)

    kp1, d1 = feature_extractor(proc1)
    kp2, d2 = feature_extractor(proc2)
    matches = match_descriptors(d1, d2, feature_name)
    if len(matches) < 8:
        raise RuntimeError(f"Need at least 8 matches for essential matrix; got {len(matches)}")

    pts1, pts2 = _matched_pixel_pairs(kp1, kp2, matches)

    E, mask_e = cv.findEssentialMat(
        pts1,
        pts2,
        K,
        method=cv.RANSAC,
        prob=ransac_prob,
        threshold=ransac_threshold_px,
    )
    if E is None or E.shape != (3, 3):
        raise RuntimeError("findEssentialMat failed to produce a 3×3 matrix")

    mask_e = mask_e.ravel().astype(bool)
    _, R, t, mask_pose = cv.recoverPose(E, pts1, pts2, K, mask=mask_e.astype(np.uint8))
    mask = mask_pose.ravel().astype(bool)

    if int(np.sum(mask)) < 1:
        raise RuntimeError("recoverPose left no inliers; try different features or thresholds")

    pts1_in = pts1[mask].reshape(-1, 2)
    pts2_in = pts2[mask].reshape(-1, 2)

    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t.reshape(3, 1)])

    points_h = cv.triangulatePoints(P1, P2, pts1_in.T, pts2_in.T)
    w = points_h[3, :]
    if np.any(np.abs(w) < 1e-10):
        raise RuntimeError("Triangulation produced points at infinity")
    xyz = (points_h[:3] / w).T

    # Cheirality: positive depth in both cameras (world Z for cam1; cam2 Z for cam2)
    z1 = xyz[:, 2]
    cam2_xyz = (R @ xyz.T + t.reshape(3, 1)).T
    z2 = cam2_xyz[:, 2]
    front = (z1 > 0) & (z2 > 0)
    if not np.any(front):
        raise RuntimeError("No points in front of both cameras after triangulation")

    pts1_in = pts1_in[front]
    pts2_in = pts2_in[front]
    xyz = xyz[front]

    reproj = _reprojection_errors_px(xyz, pts1_in, pts2_in, K, R, t)
    mean_err = float(np.mean(reproj))

    h, wim = img1.shape[:2]
    match_idx_kept = np.where(mask)[0][front]
    colors = []
    for mi in match_idx_kept:
        m = matches[int(mi)]
        x, y = kp1[m.queryIdx].pt
        xi = int(round(np.clip(x, 0, wim - 1)))
        yi = int(round(np.clip(y, 0, h - 1)))
        colors.append(img1[yi, xi].copy())
    colors_bgr = np.uint8(colors)

    final_inlier = np.zeros(len(matches), dtype=bool)
    final_inlier[match_idx_kept] = True

    idx0_list: list[int] = []
    idx1_list: list[int] = []
    uv0_rows: list[tuple[float, float]] = []
    uv1_rows: list[tuple[float, float]] = []
    for mi in match_idx_kept:
        m = matches[int(mi)]
        idx0_list.append(int(m.queryIdx))
        idx1_list.append(int(m.trainIdx))
        uv0_rows.append(kp1[m.queryIdx].pt)
        uv1_rows.append(kp2[m.trainIdx].pt)
    inlier_kp_idx0 = np.asarray(idx0_list, dtype=np.int32)
    inlier_kp_idx1 = np.asarray(idx1_list, dtype=np.int32)
    inlier_uv0 = np.asarray(uv0_rows, dtype=np.float64)
    inlier_uv1 = np.asarray(uv1_rows, dtype=np.float64)

    return TwoViewReconstruction(
        R=R,
        t=t.reshape(3),
        points_3d=xyz,
        colors_bgr=colors_bgr,
        inlier_mask=final_inlier,
        mean_reproj_error_px=mean_err,
        n_inliers=int(xyz.shape[0]),
        E=E,
        inlier_kp_idx0=inlier_kp_idx0,
        inlier_kp_idx1=inlier_kp_idx1,
        inlier_uv0=inlier_uv0,
        inlier_uv1=inlier_uv1,
    )


def write_ply_ascii(path: Path | str, points_xyz: np.ndarray, colors_bgr: np.ndarray | None = None) -> None:
    """Write a minimal ASCII PLY (vertex-only). Colors optional as RGB uint8."""
    path = Path(path)
    pts = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    n = pts.shape[0]
    has_color = colors_bgr is not None and len(colors_bgr) == n
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if has_color:
        lines.extend(
            [
                "property uchar red",
                "property uchar green",
                "property uchar blue",
            ]
        )
    lines.append("end_header")

    if has_color:
        c = np.asarray(colors_bgr, dtype=np.uint8).reshape(-1, 3)
        rgb = c[:, ::-1]  # BGR -> RGB for PLY
        for i in range(n):
            r, g, b = rgb[i]
            lines.append(f"{pts[i, 0]} {pts[i, 1]} {pts[i, 2]} {int(r)} {int(g)} {int(b)}")
    else:
        for i in range(n):
            lines.append(f"{pts[i, 0]} {pts[i, 1]} {pts[i, 2]}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_two_view_npz(
    path: Path | str,
    result: TwoViewReconstruction,
    K: np.ndarray,
) -> None:
    rvec, _ = cv.Rodrigues(result.R)
    np.savez(
        Path(path),
        points_3d=result.points_3d,
        colors_bgr=result.colors_bgr,
        R=result.R,
        t=result.t,
        rvec=rvec.reshape(3),
        E=result.E,
        K=np.asarray(K, dtype=np.float64),
        mean_reproj_error_px=result.mean_reproj_error_px,
        n_inliers=result.n_inliers,
        inlier_kp_idx0=result.inlier_kp_idx0,
        inlier_kp_idx1=result.inlier_kp_idx1,
        inlier_uv0=result.inlier_uv0,
        inlier_uv1=result.inlier_uv1,
    )
