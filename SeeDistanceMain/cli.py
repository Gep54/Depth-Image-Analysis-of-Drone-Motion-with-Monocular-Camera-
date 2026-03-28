from pathlib import Path

import click

from main import SeeDistanceApp


@click.command()
@click.option("--frames-dir", type=click.Path(path_type=Path), required=True, help="Directory with image frames")
@click.option("--intrinsics", type=click.Path(path_type=Path), required=True, help="Path to camera intrinsics file")
@click.option("--sync", type=click.Path(path_type=Path), required=True, help="Path to time synchronization file")
@click.option("--poses", type=click.Path(path_type=Path), required=True, help="Path to relative pose file")
def main(frames_dir: Path, intrinsics: Path, sync: Path, poses: Path) -> None:
    """Initialize the app and run the full pipeline."""
    app = SeeDistanceApp(
        frames_dir=frames_dir,
        intrinsics_path=intrinsics,
        sync_path=sync,
        poses_path=poses,
    )
    app.run()


if __name__ == "__main__":
    main()