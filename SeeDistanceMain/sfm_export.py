"""Step 4: export map artifacts and reprojection diagnostics from a saved incremental SfM bundle."""

from __future__ import annotations

from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd

from incremental_sfm import IncrementalSfMResult
from two_view import write_ply_ascii


def _camera_center_world(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    R, _ = cv.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    t = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    return (-R.T @ t).reshape(3)


def camera_centers_trajectory(rvecs: np.ndarray, tvecs: np.ndarray) -> np.ndarray:
    """World-frame camera centers ``C = -R^T t``, shape ``(n_frames, 3)``."""
    n = rvecs.shape[0]
    out = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        out[i] = _camera_center_world(rvecs[i], tvecs[i])
    return out


def per_observation_reprojection_errors(
    observations: np.ndarray,
    K: np.ndarray,
    rvecs: np.ndarray,
    tvecs: np.ndarray,
    points_3d: np.ndarray,
) -> np.ndarray:
    """Euclidean pixel error for each row of ``observations`` (n_obs,)."""
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    errs = np.zeros(len(observations), dtype=np.float64)
    for k in range(len(observations)):
        f = int(observations[k, 0])
        pid = int(observations[k, 1])
        u_obs, v_obs = observations[k, 2], observations[k, 3]
        X = points_3d[pid].reshape(1, 1, 3).astype(np.float64)
        proj, _ = cv.projectPoints(X, rvecs[f], tvecs[f], K, None)
        u, v = proj.reshape(2)
        errs[k] = float(np.hypot(u - u_obs, v - v_obs))
    return errs


def reprojection_summary_by_frame(
    observations: np.ndarray,
    K: np.ndarray,
    rvecs: np.ndarray,
    tvecs: np.ndarray,
    points_3d: np.ndarray,
    frame_names: list[str],
) -> pd.DataFrame:
    """One row per frame: observation count and error statistics (pixels)."""
    err = per_observation_reprojection_errors(observations, K, rvecs, tvecs, points_3d)
    fi = observations[:, 0].astype(int)
    rows = []
    n_frames = len(frame_names)
    for f in range(n_frames):
        m = fi == f
        if not np.any(m):
            rows.append(
                {
                    "frame_index": f,
                    "frame_name": frame_names[f],
                    "n_observations": 0,
                    "rms_reproj_px": float("nan"),
                    "mean_reproj_px": float("nan"),
                    "max_reproj_px": float("nan"),
                }
            )
            continue
        e = err[m]
        rows.append(
            {
                "frame_index": f,
                "frame_name": frame_names[f],
                "n_observations": int(np.sum(m)),
                "rms_reproj_px": float(np.sqrt(np.mean(e**2))),
                "mean_reproj_px": float(np.mean(e)),
                "max_reproj_px": float(np.max(e)),
            }
        )
    return pd.DataFrame(rows)


def export_incremental_map(
    result: IncrementalSfMResult,
    K: np.ndarray,
    output_dir: Path | str,
    *,
    write_ply: bool = True,
    write_observations_csv: bool = False,
) -> dict[str, Path]:
    """Write cameras CSV, point cloud CSV, reprojection summary; optional PLY and obs CSV.

    Returns paths keyed by ``cameras``, ``points``, ``reprojection``, optional ``ply``, ``observations``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    C = camera_centers_trajectory(result.rvecs, result.tvecs)
    cam_df = pd.DataFrame(
        {
            "frame_index": np.arange(len(result.frame_names), dtype=int),
            "frame_name": result.frame_names,
            "center_x": C[:, 0],
            "center_y": C[:, 1],
            "center_z": C[:, 2],
            "rvec_x": result.rvecs[:, 0],
            "rvec_y": result.rvecs[:, 1],
            "rvec_z": result.rvecs[:, 2],
            "tvec_x": result.tvecs[:, 0],
            "tvec_y": result.tvecs[:, 1],
            "tvec_z": result.tvecs[:, 2],
        }
    )
    cam_path = output_dir / "cameras.csv"
    cam_df.to_csv(cam_path, index=False)

    n_pt = result.points_3d.shape[0]
    bgr = np.asarray(result.colors_bgr, dtype=np.uint8).reshape(n_pt, 3)
    pts_df = pd.DataFrame(
        {
            "point_id": np.arange(n_pt, dtype=int),
            "x": result.points_3d[:, 0],
            "y": result.points_3d[:, 1],
            "z": result.points_3d[:, 2],
            "b": bgr[:, 0],
            "g": bgr[:, 1],
            "r": bgr[:, 2],
        }
    )
    pts_path = output_dir / "points.csv"
    pts_df.to_csv(pts_path, index=False)

    rep_df = reprojection_summary_by_frame(
        result.observations,
        K,
        result.rvecs,
        result.tvecs,
        result.points_3d,
        result.frame_names,
    )
    rep_path = output_dir / "reprojection_by_frame.csv"
    rep_df.to_csv(rep_path, index=False)

    out: dict[str, Path] = {
        "cameras": cam_path,
        "points": pts_path,
        "reprojection": rep_path,
    }

    if write_ply:
        ply_path = output_dir / "points.ply"
        write_ply_ascii(ply_path, result.points_3d, result.colors_bgr)
        out["ply"] = ply_path

    if write_observations_csv:
        err = per_observation_reprojection_errors(
            result.observations, K, result.rvecs, result.tvecs, result.points_3d
        )
        obs_df = pd.DataFrame(
            {
                "frame_index": result.observations[:, 0].astype(int),
                "point_id": result.observations[:, 1].astype(int),
                "u": result.observations[:, 2],
                "v": result.observations[:, 3],
                "reproj_error_px": err,
            }
        )
        obs_df["frame_name"] = obs_df["frame_index"].map(
            lambda i: result.frame_names[int(i)]
        )
        obs_path = output_dir / "observations.csv"
        obs_df.to_csv(obs_path, index=False)
        out["observations"] = obs_path

    return out
