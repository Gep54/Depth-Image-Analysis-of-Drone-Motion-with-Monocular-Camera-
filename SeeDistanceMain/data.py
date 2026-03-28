from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np
import pandas as pd


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SUPPORTED_TABLE_EXTENSIONS = {".csv", ".tsv"}
SUPPORTED_INTRINSICS_EXTENSIONS = {".npy", ".txt", ".csv"}


@dataclass(slots=True)
class DatasetBundle:
    """Container holding all dataset inputs used by the application."""

    frames: list[tuple[str, np.ndarray]]
    intrinsics: np.ndarray
    poses: pd.DataFrame


def dataset_summary(bundle: DatasetBundle) -> dict[str, Any]:
    """Return a compact summary of the loaded dataset."""
    return {
        "Nu. of frames": len(bundle.frames),
        "intrinsics shape": bundle.intrinsics.shape,
        "pose_rows": len(bundle.poses),
        "first_frame": bundle.frames[0][0] if bundle.frames else None,
    }


class DatasetLoader:
    """Loader object responsible for reading and validating all input data."""

    def __init__(
        self,
        frames_dir: str | Path,
        intrinsics_path: str | Path,
        poses_path: str | Path,
    ) -> None:
        """Store input paths for later loading.

        Args:
            frames_dir: Directory containing image frames.
            intrinsics_path: Path to camera intrinsics.
            poses_path: Path to relative pose estimates.
        """
        self.frames_dir = Path(frames_dir)
        self.intrinsics_path = Path(intrinsics_path)
        self.poses_path = Path(poses_path)

    def _load_frames(self) -> list[tuple[str, np.ndarray]]:
        """Load all supported image frames from the frames directory."""
        if not self.frames_dir.exists():
            raise FileNotFoundError(f"Frames directory does not exist: {self.frames_dir}")

        image_paths = sorted(
            path for path in self.frames_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        )

        if not image_paths:
            raise FileNotFoundError(f"No supported image frames found in {self.frames_dir}")

        frames: list[tuple[str, np.ndarray]] = []
        for path in image_paths:
            image = cv.imread(str(path), cv.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Could not read image: {path}")
            frames.append((path.name, image))

        return frames

    def _load_intrinsics(self) -> np.ndarray:
        """Load the camera intrinsic matrix from the disk."""
        if not self.intrinsics_path.exists():
            raise FileNotFoundError(f"Intrinsics file does not exist: {self.intrinsics_path}")

        suffix = self.intrinsics_path.suffix.lower()

        if suffix == ".npy":
            intrinsics = np.load(self.intrinsics_path)
        elif suffix == ".txt":
            intrinsics = np.loadtxt(self.intrinsics_path)
        elif suffix == ".csv":
            intrinsics = np.loadtxt(self.intrinsics_path, delimiter=",")
        else:
            raise ValueError("Intrinsics must be stored as .npy, .txt, or .csv")

        if intrinsics.shape != (3, 3):
            raise ValueError(
                f"Camera intrinsics must be a 3x3 matrix, got shape {intrinsics.shape}"
            )

        return intrinsics

    def _load_poses(self) -> pd.DataFrame:
        """Load relative pose estimates from disk."""
        if not self.poses_path.exists():
            raise FileNotFoundError(f"Pose file does not exist: {self.poses_path}")

        suffix = self.poses_path.suffix.lower()

        if suffix == ".csv":
            return pd.read_csv(self.poses_path)
        if suffix == ".tsv":
            return pd.read_csv(self.poses_path, sep="\t")

        raise ValueError("Relative pose file must be .csv or .tsv")

    def load_dataset(self) -> DatasetBundle:
        """Load all inputs and return them as a bundle."""
        return DatasetBundle(
            frames=self._load_frames(),
            intrinsics=self._load_intrinsics(),
            poses=self._load_poses(),
        )
