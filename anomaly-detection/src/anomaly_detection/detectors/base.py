"""Base anomaly detector interface and shared data structures."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class Severity(Enum):
    """Anomaly severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @classmethod
    def from_score(cls, score: float) -> "Severity":
        """Determine severity based on an anomaly score (0-1 scale).

        Args:
            score: Anomaly score between 0 and 1.

        Returns:
            Corresponding severity level.
        """
        if score >= 0.9:
            return cls.CRITICAL
        elif score >= 0.7:
            return cls.HIGH
        elif score >= 0.4:
            return cls.MEDIUM
        else:
            return cls.LOW


@dataclass
class AnomalyResult:
    """Result of anomaly detection for a single data point."""

    index: int
    value: float
    is_anomaly: bool
    score: float
    severity: Severity
    method: str
    metric_name: str = ""
    timestamp: Optional[float] = None
    details: dict[str, float] = field(default_factory=dict)


class BaseDetector(ABC):
    """Abstract base class for anomaly detectors."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the detection method."""

    @abstractmethod
    def detect(
        self,
        values: np.ndarray,
        metric_name: str = "",
        timestamps: Optional[np.ndarray] = None,
    ) -> list[AnomalyResult]:
        """Run anomaly detection on a series of values.

        Args:
            values: Array of numeric values to analyze.
            metric_name: Name of the metric being analyzed.
            timestamps: Optional array of timestamps corresponding to values.

        Returns:
            List of AnomalyResult objects for each data point.
        """

    def _validate_input(self, values: np.ndarray, min_points: int = 3) -> bool:
        """Validate input data meets minimum requirements.

        Args:
            values: Input data array.
            min_points: Minimum number of data points required.

        Returns:
            True if input is valid, False otherwise.
        """
        if len(values) < min_points:
            return False
        return not np.all(np.isnan(values))
