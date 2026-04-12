import argparse
from pathlib import Path

import cv2 as cv

from camera_calibration import calibrate_camera_chessboard
from data import load_distortion, load_frames, load_intrinsics, load_sync
from mymatch import (
    edge_preprocess,
    features_ORB,
    features_SIFT,
    features_KAZE,
    match,
    visualize_images_matched_features,
    save_images_matched_features,
    matches_to_df,
)
from sequence_match import consecutive_pair_match_stats
from incremental_sfm import load_incremental_npz, run_incremental_sfm, save_incremental_npz
from sfm_export import export_incremental_map
from two_view import reconstruct_two_view, save_two_view_npz, write_ply_ascii


def load_image(image_path: str):
    image = cv.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return image


def get_feature_extractor(name: str):
    name = name.upper()
    if name == "ORB":
        return features_ORB
    if name == "SIFT":
        return features_SIFT
    if name == "KAZE":
        return features_KAZE
    raise ValueError(f"Unsupported feature type: {name}")


def run_match(args):
    image1 = load_image(args.image1)
    image2 = load_image(args.image2)

    extractor = get_feature_extractor(args.feature)
    preprocess = edge_preprocess if args.preprocess == "edges" else lambda x: x

    processed1 = preprocess(image1)
    processed2 = preprocess(image2)

    keypoints1, descriptors1 = extractor(processed1)
    keypoints2, descriptors2 = extractor(processed2)

    if descriptors1 is None or descriptors2 is None:
        raise RuntimeError("No descriptors found in one or both images.")

    matches = match(descriptors1, descriptors2)
    df = matches_to_df(matches, keypoints1, keypoints2)

    if args.output:
        output_path = Path(args.output)
        if output_path.suffix.lower() == ".csv":
            df.to_csv(output_path, index=False)
        else:
            save_images_matched_features(
                image1,
                image2,
                keypoints1,
                keypoints2,
                matches,
                name=str(output_path),
            )

    if args.show:
        visualize_images_matched_features(image1, image2, keypoints1, keypoints2, matches)

    print(df.head())
    print(f"Total matches: {len(matches)}")


def run_calibrate(_args):
    ret, mtx, dist, rvecs, tvecs = calibrate_camera_chessboard()
    print("Calibration successful")
    print(f"Reprojection error: {ret}")
    print("Camera matrix:")
    print(mtx)
    print("Distortion coefficients:")
    print(dist)


def run_sequence_match(args):
    frames_dir = Path(args.frames_dir)
    frames = load_frames(frames_dir)
    K = load_intrinsics(Path(args.intrinsics))
    dist = load_distortion(Path(args.dist) if args.dist else None)

    extractor = get_feature_extractor(args.feature)
    preprocess = edge_preprocess if args.preprocess == "edges" else lambda x: x

    df = consecutive_pair_match_stats(
        frames,
        feature_extractor=extractor,
        preprocess=preprocess,
        feature_name=args.feature,
        K=K,
        dist=dist,
        undistort=not args.no_undistort,
    )

    out = Path(args.output) if args.output else None
    if out:
        df.to_csv(out, index=False)
        print(f"Wrote statistics to {out}")

    print(df.to_string(index=False))
    print(f"Pairs processed: {len(df)}")


def run_two_view(args):
    K = load_intrinsics(Path(args.intrinsics))
    dist = load_distortion(Path(args.dist) if args.dist else None)

    if args.frames_dir:
        frames = load_frames(Path(args.frames_dir))
        pair = args.pair_index
        if pair < 0 or pair + 1 >= len(frames):
            raise SystemExit(f"pair_index {pair} invalid for {len(frames)} frames (need 0 .. {len(frames) - 2})")
        _, img1 = frames[pair]
        _, img2 = frames[pair + 1]
    elif args.image1:
        if not args.image2:
            raise SystemExit("--image2 is required when using --image1")
        img1 = load_image(args.image1)
        img2 = load_image(args.image2)
    else:
        raise SystemExit("Provide --frames-dir or both --image1 and --image2")

    extractor = get_feature_extractor(args.feature)
    preprocess = edge_preprocess if args.preprocess == "edges" else lambda x: x

    result = reconstruct_two_view(
        img1,
        img2,
        K,
        dist,
        feature_extractor=extractor,
        preprocess=preprocess,
        feature_name=args.feature,
        undistort=not args.no_undistort,
        ransac_prob=args.ransac_prob,
        ransac_threshold_px=args.ransac_threshold,
    )

    if args.output_ply:
        write_ply_ascii(args.output_ply, result.points_3d, result.colors_bgr)
        print(f"Wrote PLY: {args.output_ply}")
    if args.output_npz:
        save_two_view_npz(args.output_npz, result, K)
        print(f"Wrote NPZ: {args.output_npz}")

    print(f"Inlier 3D points: {result.n_inliers}")
    print(f"Mean reprojection error (px): {result.mean_reproj_error_px:.4f}")


def run_incremental_cli(args):
    frames = load_frames(Path(args.frames_dir))
    K = load_intrinsics(Path(args.intrinsics))
    dist = load_distortion(Path(args.dist) if args.dist else None)

    sync_df = None
    if args.sync:
        sync_df = load_sync(Path(args.sync))

    extractor = get_feature_extractor(args.feature)
    preprocess = edge_preprocess if args.preprocess == "edges" else lambda x: x

    result = run_incremental_sfm(
        frames,
        K,
        dist,
        feature_extractor=extractor,
        preprocess=preprocess,
        feature_name=args.feature,
        undistort=not args.no_undistort,
        two_view_pair_index=args.pair_index,
        ransac_prob=args.ransac_prob,
        ransac_threshold_px=args.ransac_threshold,
        pnp_reproj_threshold=args.pnp_reproj_threshold,
        sync_df=sync_df,
        sync_default_weight=args.sync_default_weight,
        run_bundle_adjustment=not args.no_bundle,
        ba_verbose=args.ba_verbose,
    )

    if args.output_ply:
        write_ply_ascii(args.output_ply, result.points_3d, result.colors_bgr)
        print(f"Wrote PLY: {args.output_ply}")
    if args.output_npz:
        save_incremental_npz(args.output_npz, result, K)
        print(f"Wrote NPZ: {args.output_npz}")

    print(f"3D points: {result.points_3d.shape[0]}")
    print(f"Frames: {len(result.frame_names)}")
    if result.bundle_cost is not None:
        print(f"Bundle adjustment final cost: {result.bundle_cost:.6f}")
    else:
        print("Bundle adjustment skipped")
    print(f"PnP inliers (last frame): {result.pnp_inliers_last_frame}")


def run_export_map(args):
    result, K = load_incremental_npz(Path(args.input))
    paths = export_incremental_map(
        result,
        K,
        Path(args.output_dir),
        write_ply=not args.no_ply,
        write_observations_csv=args.observations_csv,
    )
    for key, p in paths.items():
        print(f"{key}: {p}")
    if result.bundle_cost is not None:
        print(f"bundle_cost (from NPZ): {result.bundle_cost:.6f}")


def run_reconstruct(args):
    """End-to-end: optional match stats → incremental SfM + BA → robust refine → PLY/NPZ/export."""
    frames = load_frames(Path(args.frames_dir))
    K = load_intrinsics(Path(args.intrinsics))
    dist = load_distortion(Path(args.dist) if args.dist else None)

    extractor = get_feature_extractor(args.feature)
    preprocess = edge_preprocess if args.preprocess == "edges" else lambda x: x

    if args.match_stats:
        df = consecutive_pair_match_stats(
            frames,
            feature_extractor=extractor,
            preprocess=preprocess,
            feature_name=args.feature,
            K=K,
            dist=dist,
            undistort=not args.no_undistort,
        )
        Path(args.match_stats).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.match_stats, index=False)
        print(f"Wrote sequence match stats: {args.match_stats}")

    sync_df = None
    if args.sync:
        sync_df = load_sync(Path(args.sync))

    result = run_incremental_sfm(
        frames,
        K,
        dist,
        feature_extractor=extractor,
        preprocess=preprocess,
        feature_name=args.feature,
        undistort=not args.no_undistort,
        two_view_pair_index=args.pair_index,
        ransac_prob=args.ransac_prob,
        ransac_threshold_px=args.ransac_threshold,
        pnp_reproj_threshold=args.pnp_reproj_threshold,
        sync_df=sync_df,
        sync_default_weight=args.sync_default_weight,
        run_bundle_adjustment=not args.no_bundle,
        ba_verbose=args.ba_verbose,
    )

    if not args.no_refine:
        from refine_map import refine_incremental_bundle

        rmax = args.refine_max_nfev if args.refine_max_nfev > 0 else None
        result = refine_incremental_bundle(
            result,
            K,
            sync_df=sync_df,
            sync_default_weight=args.sync_default_weight,
            loss=args.refine_loss,
            f_scale=args.refine_f_scale,
            max_nfev=rmax,
            verbose=args.ba_verbose,
        )

    out_ply = Path(args.output_ply)
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    write_ply_ascii(out_ply, result.points_3d, result.colors_bgr)
    print(f"Wrote point cloud (PLY): {out_ply}")

    if args.output_npz:
        save_incremental_npz(Path(args.output_npz), result, K)
        print(f"Wrote bundle (NPZ): {args.output_npz}")

    if args.export_dir:
        paths = export_incremental_map(
            result,
            K,
            Path(args.export_dir),
            write_ply=not args.export_no_ply,
            write_observations_csv=args.export_observations_csv,
        )
        for key, p in paths.items():
            print(f"export {key}: {p}")

    print(f"Frames: {len(result.frame_names)}  |  3D points: {result.points_3d.shape[0]}")
    if result.bundle_cost is not None:
        print(f"Final bundle cost: {result.bundle_cost:.6f}")


def run_refine_map(args):
    from refine_map import refine_incremental_bundle

    result, K = load_incremental_npz(Path(args.input))
    sync_df = load_sync(Path(args.sync)) if args.sync else None
    max_nfev = args.max_nfev if args.max_nfev > 0 else None
    refined = refine_incremental_bundle(
        result,
        K,
        sync_df=sync_df,
        sync_default_weight=args.sync_default_weight,
        loss=args.loss,
        f_scale=args.f_scale,
        max_nfev=max_nfev,
        verbose=args.ba_verbose,
    )
    save_incremental_npz(Path(args.output), refined, K)
    print(f"Wrote {args.output}")
    print(f"Refinement cost: {refined.bundle_cost:.6f}")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="see-distance",
        description="Image feature matching and camera calibration tool",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    calibrate_parser = subparsers.add_parser(
        "calibrate",
        help="Run chessboard camera calibration",
    )
    calibrate_parser.set_defaults(func=run_calibrate)

    match_parser = subparsers.add_parser(
        "match",
        help="Match features between two images",
    )
    match_parser.add_argument("--image1", required=True, help="Path to first image")
    match_parser.add_argument("--image2", required=True, help="Path to second image")
    match_parser.add_argument(
        "--feature",
        default="ORB",
        choices=["ORB", "SIFT", "KAZE"],
        help="Feature detector to use",
    )
    match_parser.add_argument(
        "--preprocess",
        default="edges",
        choices=["edges", "none"],
        help="Preprocessing mode",
    )
    match_parser.add_argument(
        "--output",
        help="Output file path (.csv for table, .jpg/.png for visualization)",
    )
    match_parser.add_argument(
        "--show",
        action="store_true",
        help="Show matched features in a window",
    )
    match_parser.set_defaults(func=run_match)

    seq_parser = subparsers.add_parser(
        "sequence-match",
        help="Step 1: load a frame folder + intrinsics, optionally undistort, "
        "match consecutive pairs, export statistics",
    )
    seq_parser.add_argument(
        "--frames-dir",
        required=True,
        type=str,
        help="Directory of images (sorted by filename)",
    )
    seq_parser.add_argument(
        "--intrinsics",
        required=True,
        type=str,
        help="Camera matrix K (.npy, .txt, or .csv)",
    )
    seq_parser.add_argument(
        "--dist",
        type=str,
        default=None,
        help="Distortion coefficients (.npy or .txt); omit for no distortion",
    )
    seq_parser.add_argument(
        "--no-undistort",
        action="store_true",
        help="Skip cv.undistort (ignore --dist for geometry; matching on raw images)",
    )
    seq_parser.add_argument(
        "--feature",
        default="ORB",
        choices=["ORB", "SIFT", "KAZE"],
        help="Feature detector",
    )
    seq_parser.add_argument(
        "--preprocess",
        default="edges",
        choices=["edges", "none"],
        help="Preprocessing before detection",
    )
    seq_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write match statistics CSV to this path",
    )
    seq_parser.set_defaults(func=run_sequence_match)

    tv_parser = subparsers.add_parser(
        "two-view",
        help="Step 2: two-view SfM — essential matrix, pose, triangulation, export cloud",
    )
    src = tv_parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image1", default=None, help="First image path (use with --image2)")
    src.add_argument(
        "--frames-dir",
        default=None,
        type=str,
        help="Directory of images; uses frames[pair_index] and frames[pair_index+1]",
    )
    tv_parser.add_argument(
        "--image2",
        default=None,
        help="Second image path (required with --image1)",
    )
    tv_parser.add_argument(
        "--pair-index",
        type=int,
        default=0,
        help="When using --frames-dir: index i matches frames i and i+1 (default 0)",
    )
    tv_parser.add_argument(
        "--intrinsics",
        required=True,
        type=str,
        help="Camera matrix K (.npy, .txt, or .csv)",
    )
    tv_parser.add_argument(
        "--dist",
        type=str,
        default=None,
        help="Distortion coefficients (.npy or .txt); omit if none",
    )
    tv_parser.add_argument(
        "--no-undistort",
        action="store_true",
        help="Detect/match on raw images (geometry still uses K)",
    )
    tv_parser.add_argument(
        "--feature",
        default="SIFT",
        choices=["ORB", "SIFT", "KAZE"],
        help="Feature detector (default SIFT for two-view stability)",
    )
    tv_parser.add_argument(
        "--preprocess",
        default="none",
        choices=["edges", "none"],
        help="Preprocessing before detection (default none for step 2)",
    )
    tv_parser.add_argument(
        "--output-ply",
        type=str,
        default=None,
        help="Write sparse point cloud as ASCII PLY",
    )
    tv_parser.add_argument(
        "--output-npz",
        type=str,
        default=None,
        help="Write points, pose, K, and diagnostics as NPZ",
    )
    tv_parser.add_argument(
        "--ransac-prob",
        type=float,
        default=0.999,
        help="RANSAC confidence for findEssentialMat",
    )
    tv_parser.add_argument(
        "--ransac-threshold",
        type=float,
        default=1.0,
        help="Max epipolar distance in pixels for essential matrix RANSAC",
    )
    tv_parser.set_defaults(func=run_two_view)

    inc_parser = subparsers.add_parser(
        "incremental-sfm",
        help="Step 3: incremental PnP + triangulation; optional BA with (x,y) sync priors only",
    )
    inc_parser.add_argument("--frames-dir", required=True, type=str, help="Ordered image directory")
    inc_parser.add_argument("--intrinsics", required=True, type=str, help="3×3 K (.npy, .txt, .csv)")
    inc_parser.add_argument(
        "--dist",
        type=str,
        default=None,
        help="Distortion .npy or .txt (optional)",
    )
    inc_parser.add_argument(
        "--no-undistort",
        action="store_true",
        help="Skip undistortion before features",
    )
    inc_parser.add_argument(
        "--feature",
        default="SIFT",
        choices=["ORB", "SIFT", "KAZE"],
        help="Feature type",
    )
    inc_parser.add_argument(
        "--preprocess",
        default="none",
        choices=["edges", "none"],
        help="Preprocessing",
    )
    inc_parser.add_argument(
        "--pair-index",
        type=int,
        default=0,
        help="Two-view seed is frames[i] and frames[i+1] (must be 0)",
    )
    inc_parser.add_argument(
        "--ransac-prob",
        type=float,
        default=0.999,
        help="Essential matrix RANSAC confidence",
    )
    inc_parser.add_argument(
        "--ransac-threshold",
        type=float,
        default=1.0,
        help="Essential matrix RANSAC threshold (px)",
    )
    inc_parser.add_argument(
        "--pnp-reproj-threshold",
        type=float,
        default=8.0,
        help="solvePnPRansac reprojection threshold (px)",
    )
    inc_parser.add_argument(
        "--sync",
        type=str,
        default=None,
        help="CSV/TSV with image,x,y[,weight|sigma] for planar camera-center priors",
    )
    inc_parser.add_argument(
        "--sync-default-weight",
        type=float,
        default=1.0,
        help="Prior weight if sync row has no weight/sigma column",
    )
    inc_parser.add_argument(
        "--no-bundle",
        action="store_true",
        help="Skip bundle adjustment (geometry = incremental only)",
    )
    inc_parser.add_argument(
        "--ba-verbose",
        type=int,
        default=0,
        help="Verbosity for scipy least_squares (0 or 1)",
    )
    inc_parser.add_argument("--output-ply", type=str, default=None, help="Sparse colored PLY")
    inc_parser.add_argument("--output-npz", type=str, default=None, help="NPZ with poses, points, obs")
    inc_parser.set_defaults(func=run_incremental_cli)

    rec_parser = subparsers.add_parser(
        "reconstruct",
        help="Full chain: frames + K + dist → match stats (opt.) → incremental SfM + BA → robust refine → PLY",
    )
    rec_parser.add_argument(
        "--frames-dir",
        required=True,
        type=str,
        help="Directory of ordered images (same as step 1)",
    )
    rec_parser.add_argument(
        "--intrinsics",
        required=True,
        type=str,
        help="Camera matrix K (.npy, .txt, .csv)",
    )
    rec_parser.add_argument(
        "--dist",
        type=str,
        default=None,
        help="Distortion coefficients (.npy or .txt); omit if none",
    )
    rec_parser.add_argument(
        "--output-ply",
        required=True,
        type=str,
        help="Output path for colored sparse point cloud (PLY)",
    )
    rec_parser.add_argument(
        "--output-npz",
        type=str,
        default=None,
        help="Also save full bundle for refine-map / export-map",
    )
    rec_parser.add_argument(
        "--export-dir",
        type=str,
        default=None,
        help="Also run step-4 export (CSV + optional PLY) into this directory",
    )
    rec_parser.add_argument(
        "--export-no-ply",
        action="store_true",
        help="With --export-dir: skip duplicate points.ply there",
    )
    rec_parser.add_argument(
        "--export-observations-csv",
        action="store_true",
        help="With --export-dir: write observations.csv",
    )
    rec_parser.add_argument(
        "--match-stats",
        type=str,
        default=None,
        help="Optional: write step-1 consecutive-pair match statistics CSV",
    )
    rec_parser.add_argument(
        "--no-undistort",
        action="store_true",
        help="Skip undistortion before features",
    )
    rec_parser.add_argument(
        "--feature",
        default="SIFT",
        choices=["ORB", "SIFT", "KAZE"],
        help="Feature detector",
    )
    rec_parser.add_argument(
        "--preprocess",
        default="none",
        choices=["edges", "none"],
        help="Preprocessing before detection",
    )
    rec_parser.add_argument(
        "--pair-index",
        type=int,
        default=0,
        help="Two-view seed frames i and i+1 (must be 0)",
    )
    rec_parser.add_argument(
        "--ransac-prob",
        type=float,
        default=0.999,
        help="Essential matrix RANSAC confidence",
    )
    rec_parser.add_argument(
        "--ransac-threshold",
        type=float,
        default=1.0,
        help="Essential matrix RANSAC threshold (px)",
    )
    rec_parser.add_argument(
        "--pnp-reproj-threshold",
        type=float,
        default=8.0,
        help="solvePnPRansac reprojection threshold (px)",
    )
    rec_parser.add_argument(
        "--sync",
        type=str,
        default=None,
        help="CSV/TSV with image,x,y for planar camera-center priors (steps 3–5)",
    )
    rec_parser.add_argument(
        "--sync-default-weight",
        type=float,
        default=1.0,
        help="Default prior weight when sync row has no weight/sigma",
    )
    rec_parser.add_argument(
        "--no-bundle",
        action="store_true",
        help="Skip step-3 bundle adjustment (incremental geometry only)",
    )
    rec_parser.add_argument(
        "--no-refine",
        action="store_true",
        help="Skip step-5 robust refinement",
    )
    rec_parser.add_argument(
        "--refine-loss",
        type=str,
        default="soft_l1",
        choices=["linear", "soft_l1", "huber", "cauchy", "arctan"],
        help="Robust loss for refinement pass",
    )
    rec_parser.add_argument(
        "--refine-f-scale",
        type=float,
        default=1.5,
        help="Robust f_scale for refinement",
    )
    rec_parser.add_argument(
        "--refine-max-nfev",
        type=int,
        default=0,
        help="Max refinement iterations (0 = automatic)",
    )
    rec_parser.add_argument(
        "--ba-verbose",
        type=int,
        default=0,
        help="Verbosity for scipy least_squares (bundle + refine)",
    )
    rec_parser.set_defaults(func=run_reconstruct)

    exp_parser = subparsers.add_parser(
        "export-map",
        help="Step 4: export cameras, points, reprojection stats from incremental NPZ",
    )
    exp_parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="NPZ from incremental-sfm (save_incremental_npz)",
    )
    exp_parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="Directory for CSV (and optional PLY) outputs",
    )
    exp_parser.add_argument(
        "--no-ply",
        action="store_true",
        help="Skip writing points.ply",
    )
    exp_parser.add_argument(
        "--observations-csv",
        action="store_true",
        help="Also write observations.csv (per keypoint reprojection errors; can be large)",
    )
    exp_parser.set_defaults(func=run_export_map)

    ref_parser = subparsers.add_parser(
        "refine-map",
        help="Step 5: robust global BA on a saved NPZ (soft_l1/huber loss, optional sync)",
    )
    ref_parser.add_argument("--input", required=True, type=str, help="NPZ from incremental-sfm or prior refine")
    ref_parser.add_argument("--output", required=True, type=str, help="Output NPZ path")
    ref_parser.add_argument(
        "--sync",
        type=str,
        default=None,
        help="Optional CSV/TSV for (x,y) camera-center priors during refinement",
    )
    ref_parser.add_argument(
        "--sync-default-weight",
        type=float,
        default=1.0,
        help="Default prior weight when sync row has no weight/sigma",
    )
    ref_parser.add_argument(
        "--loss",
        type=str,
        default="soft_l1",
        choices=["linear", "soft_l1", "huber", "cauchy", "arctan"],
        help="scipy least_squares robust loss (applies to all residuals)",
    )
    ref_parser.add_argument(
        "--f-scale",
        type=float,
        default=1.5,
        help="Robust scale (pixels-ish for reproj components); see SciPy docs",
    )
    ref_parser.add_argument(
        "--max-nfev",
        type=int,
        default=0,
        help="Max BA iterations (0 = automatic)",
    )
    ref_parser.add_argument(
        "--ba-verbose",
        type=int,
        default=0,
        help="scipy least_squares verbosity",
    )
    ref_parser.set_defaults(func=run_refine_map)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()