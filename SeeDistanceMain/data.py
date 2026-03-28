from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd


@dataclass
class DatasetBundle:
    """Container for the dataset inputs required by the application."""

    frames: list
    intrinsics: np.ndarray
    sync: pd.DataFrame
    poses: pd.DataFrame


def load_frames(frames_dir: Path):
    """Load all supported image frames from a directory.

    Args:
        frames_dir: Directory containing image files.

    Returns:
        A list of tuples `(filename, image)` for each readable frame.

    Raises:
        FileNotFoundError: If the directory contains no supported images or a
            frame cannot be read.
    """
    image_paths = sorted(
        [
            p for p in frames_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        ]
    )
    if not image_paths:
        raise FileNotFoundError(f"No image frames found in {frames_dir}")

    frames = []
    for path in image_paths:
        image = cv.imread(str(path), cv.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        frames.append((path.name, image))
    return frames


def load_intrinsics(k_path: Path):
    """Load the camera intrinsic matrix from disk.

    Supported formats are `.npy`, `.txt`, and `.csv`.

    Args:
        k_path: Path to the intrinsic matrix file.

    Returns:
        A 3x3 NumPy array representing camera intrinsics.

    Raises:
        ValueError: If the file format is unsupported or the loaded matrix is
            not 3x3.
    """
    suffix = k_path.suffix.lower()

    if suffix == ".npy":
        K = np.load(k_path)
    elif suffix == ".txt":
        K = np.loadtxt(k_path)
    elif suffix == ".csv":
        K = np.loadtxt(k_path, delimiter=",")
    else:
        raise ValueError("Intrinsics must be stored as .npy, .txt, or .csv")

    if K.shape != (3, 3):
        raise ValueError(f"Camera intrinsics must be a 3x3 matrix, got shape {K.shape}")

    return K


def load_sync(sync_path: Path):
    """Load time synchronization information from a tabular file.

    Args:
        sync_path: Path to the synchronization file.

    Returns:
        A pandas DataFrame containing synchronization data.

    Raises:
        ValueError: If the file format is unsupported.
    """
    suffix = sync_path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(sync_path)
    if suffix == ".tsv":
        return pd.read_csv(sync_path, sep="\t")

    raise ValueError("Time synchronization file must be .csv or .tsv")


def load_poses(poses_path: Path):
    """Load relative pose estimates from a tabular file.

    Args:
        poses_path: Path to the relative pose file.

    Returns:
        A pandas DataFrame containing relative pose estimates.

    Raises:
        ValueError: If the file format is unsupported.
    """
    suffix = poses_path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(poses_path)
    if suffix == ".tsv":
        return pd.read_csv(poses_path, sep="\t")

    raise ValueError("Relative pose file must be .csv or .tsv")


def load_dataset(frames_dir: Path, intrinsics_path: Path, sync_path: Path, poses_path: Path):
    """Load all dataset inputs into a single bundle.

    Args:
        frames_dir: Directory containing image frames.
        intrinsics_path: Path to camera intrinsics.
        sync_path: Path to time synchronization data.
        poses_path: Path to relative pose estimates.

    Returns:
        A DatasetBundle containing frames, intrinsics, synchronization data,
        and pose estimates.
    """
    frames = load_frames(frames_dir)
    intrinsics = load_intrinsics(intrinsics_path)
    sync = load_sync(sync_path)
    poses = load_poses(poses_path)

    return DatasetBundle(
        frames=frames,
        intrinsics=intrinsics,
        sync=sync,
        poses=poses,
    )


def dataset_summary(bundle: DatasetBundle):
    """Build a small summary of the loaded dataset.

    Args:
        bundle: Loaded dataset bundle.

    Returns:
        A dictionary containing basic dataset statistics.
    """
    return {
        "frames": len(bundle.frames),
        "intrinsics_shape": bundle.intrinsics.shape,
        "sync_rows": len(bundle.sync),
        "pose_rows": len(bundle.poses),
        "first_frame": bundle.frames[0][0] if bundle.frames else None,
    }