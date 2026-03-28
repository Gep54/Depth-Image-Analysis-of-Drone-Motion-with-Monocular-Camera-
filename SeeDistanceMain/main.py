from pathlib import Path

from data import load_dataset


class SeeDistanceApp:
    """Main application container for the SeeDistance pipeline.

    This class will hold the loaded dataset and orchestrate the processing
    steps. Individual processing stages will be implemented later.
    """

    def __init__(
            self,
            frames_dir: str | Path,
            intrinsics_path: str | Path,
            sync_path: str | Path,
            poses_path: str | Path,
    ) -> None:
        """Initialize the application and load the required input data.

        Args:
            frames_dir: Directory containing image frames.
            intrinsics_path: Path to the camera intrinsics file.
            sync_path: Path to the time synchronization file.
            poses_path: Path to the relative pose file.
        """
        self.frames_dir = Path(frames_dir)
        self.intrinsics_path = Path(intrinsics_path)
        self.sync_path = Path(sync_path)
        self.poses_path = Path(poses_path)

        self.dataset = None

    def run(self) -> None:
        """Execute the application pipeline.

        The individual processing steps will be added later. For now, this
        method loads the dataset and prepares the application state.
        """
        self.dataset = load_dataset(
            frames_dir=self.frames_dir,
            intrinsics_path=self.intrinsics_path,
            sync_path=self.sync_path,
            poses_path=self.poses_path,
        )

        print("Dataset loaded successfully")
        print(f"Frames: {len(self.dataset.frames)}")
        print(f"Intrinsics shape: {self.dataset.intrinsics.shape}")
        print(f"Synchronization rows: {len(self.dataset.sync)}")
        print(f"Pose rows: {len(self.dataset.poses)}")