import argparse
from pathlib import Path

import cv2 as cv

from camera_calibration import calibrate_camera_chessboard
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

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()