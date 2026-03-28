from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class OdometryState:
    """Container for odometry-related state."""

    position: np.ndarray
    orientation: np.ndarray
    transform: np.ndarray


def _identity_transform() -> np.ndarray:
    """Return a 4x4 identity transform."""
    return np.eye(4, dtype=float)


def _compose_transform(t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    """Compose two homogeneous transforms."""
    if t1.shape != (4, 4) or t2.shape != (4, 4):
        raise ValueError("Both transforms must be 4x4 matrices")
    return t1 @ t2


def _invert_transform(transform: np.ndarray) -> np.ndarray:
    """Invert a homogeneous transform."""
    if transform.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 transform, got {transform.shape}")
    return np.linalg.inv(transform)


class OdometryGeometry:
    """Handle geometry operations for odometry processing."""

    def __init__(self) -> None:
        """Initialize the odometry geometry helper."""
        self.state: OdometryState | None = None

    def _make_state(self) -> OdometryState:
        """Create a default odometry state."""
        return OdometryState(
            position=np.zeros(3, dtype=float),
            orientation=np.zeros(3, dtype=float),
            transform=_identity_transform(),
        )

    def _update_state(self, transform: np.ndarray) -> None:
        """Update the internal odometry state with a new transform."""
        if transform.shape != (4, 4):
            raise ValueError(f"Expected a 4x4 transform, got {transform.shape}")

        if self.state is None:
            self.state = self._make_state()

        self.state.transform = transform

    def step(self, transform: np.ndarray | None = None) -> OdometryState:
        """Run the geometry handling step.
        Args:
            transform: Optional 4x4 pose transform to store as the current state.
        Returns:
            The current odometry state.
        """
        if self.state is None:
            self.state = self._make_state()

        if transform is not None:
            self._update_state(transform)

        return self.state