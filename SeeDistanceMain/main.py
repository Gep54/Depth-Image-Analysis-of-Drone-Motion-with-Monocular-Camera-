from pathlib import Path

from data import DatasetLoader


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
        loader = DatasetLoader(
            frames_dir=self.frames_dir,
            intrinsics_path=self.intrinsics_path,
            poses_path=self.poses_path,)
        self.dataset = loader.load_dataset()

        log_msg = f"Dataset loaded successfully: {self.dataset.dataset_summary(self.dataset)}"


