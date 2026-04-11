"""Consecutive-frame feature matching and statistics for multi-image sequences."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import cv2 as cv
import numpy as np
import pandas as pd

from data import undistort_image_bgr


def _matcher_norm_for_feature(feature_name: str) -> int:
    return cv.NORM_HAMMING if feature_name.upper() == "ORB" else cv.NORM_L2


def match_descriptors(
    descriptors1: np.ndarray,
    descriptors2: np.ndarray,
    feature_name: str,
) -> list[cv.DMatch]:
    if descriptors1 is None or descriptors2 is None:
        return []
    norm = _matcher_norm_for_feature(feature_name)
    bf = cv.BFMatcher(norm, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    return sorted(matches, key=lambda m: m.distance)


def consecutive_pair_match_stats(
    frames: Sequence[tuple[str, np.ndarray]],
    *,
    feature_extractor: Callable[[np.ndarray], tuple[list, np.ndarray | None]],
    preprocess: Callable[[np.ndarray], np.ndarray],
    feature_name: str,
    K: np.ndarray | None = None,
    dist: np.ndarray | None = None,
    undistort: bool = True,
) -> pd.DataFrame:
    """Match each consecutive frame pair and collect summary statistics.

    When ``undistort`` is True and ``K`` and ``dist`` are provided, images are
    undistorted before detection (pinhole-consistent keypoint locations when
    distortion is non-negligible).

    Parameters
    ----------
    frames
        Ordered list of ``(filename, bgr_image)`` as from :func:`data.load_frames`.
    feature_extractor
        Function taking a single-channel or BGR array and returning
        ``(keypoints, descriptors)``.
    preprocess
        Applied to each image before detection (e.g. edges or identity).
    feature_name
        Used only to pick Hamming vs L2 matching (ORB vs SIFT/KAZE).
    K, dist
        Camera intrinsics and distortion; required for undistortion when
        ``undistort`` is True.
    undistort
        If False, use raw images regardless of ``K``/``dist``.

    Returns
    -------
    DataFrame with one row per consecutive pair.
    """
    if len(frames) < 2:
        raise ValueError("Need at least two frames for consecutive matching")

    if undistort:
        if K is None or dist is None:
            raise ValueError("K and dist are required when undistort=True")
        K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        dist = np.asarray(dist, dtype=np.float64).reshape(-1)

    rows: list[dict] = []
    for i in range(len(frames) - 1):
        name_a, img_a = frames[i]
        name_b, img_b = frames[i + 1]

        if undistort:
            img_a = undistort_image_bgr(img_a, K, dist)
            img_b = undistort_image_bgr(img_b, K, dist)

        proc_a = preprocess(img_a)
        proc_b = preprocess(img_b)

        kp_a, desc_a = feature_extractor(proc_a)
        kp_b, desc_b = feature_extractor(proc_b)

        n_kp_a = len(kp_a) if kp_a is not None else 0
        n_kp_b = len(kp_b) if kp_b is not None else 0

        matches = match_descriptors(desc_a, desc_b, feature_name)
        n_m = len(matches)
        if n_m:
            distances = np.array([m.distance for m in matches], dtype=np.float64)
            mean_d = float(np.mean(distances))
            median_d = float(np.median(distances))
        else:
            mean_d = float("nan")
            median_d = float("nan")

        rows.append(
            {
                "pair_index": i,
                "image_a": name_a,
                "image_b": name_b,
                "n_keypoints_a": n_kp_a,
                "n_keypoints_b": n_kp_b,
                "n_matches": n_m,
                "mean_match_distance": mean_d,
                "median_match_distance": median_d,
            }
        )

    return pd.DataFrame(rows)
