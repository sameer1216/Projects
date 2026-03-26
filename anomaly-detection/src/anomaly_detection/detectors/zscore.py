"""Z-Score based anomaly detection."""

import logging
from typing import Optional

import numpy as np

from anomaly_detection.detectors.base import AnomalyResult, BaseDetector, Severity

logger = logging.getLogger(__name__)


class ZScoreDetector(BaseDetector):
    """Detect anomalies using the Z-Score (standard deviation) method.

    A data point is flagged as anomalous if its Z-score exceeds the
    configured threshold. The Z-score measures how many standard
    deviations a point is from the mean.
    """

    def __init__(self, threshold: float = 3.0) -> None:
        """Initialize the Z-Score detector.

        Args:
            threshold: Number of standard deviations from the mean
                       to consider a point anomalous. Default is 3.0.
        """
        self.threshold = threshold

    @property
    def name(self) -> str:
        return "zscore"

    def detect(
        self,
        values: np.ndarray,
        metric_name: str = "",
        timestamps: Optional[np.ndarray] = None,
    ) -> list[AnomalyResult]:
        """Detect anomalies using Z-Score method.

        Args:
            values: Array of numeric values to analyze.
            metric_name: Name of the metric being analyzed.
            timestamps: Optional array of timestamps.

        Returns:
            List of AnomalyResult for each data point.
        """
        if not self._validate_input(values):
            logger.warning("Insufficient data for Z-Score detection (need >= 3 points)")
            return []

        clean_values = np.nan_to_num(values, nan=0.0)
        mean = np.mean(clean_values)
        std = np.std(clean_values)

        if std == 0:
            logger.info("Standard deviation is 0, no anomalies possible")
            return [
                AnomalyResult(
                    index=i,
                    value=float(clean_values[i]),
                    is_anomaly=False,
                    score=0.0,
                    severity=Severity.LOW,
                    method=self.name,
                    metric_name=metric_name,
                    timestamp=float(timestamps[i]) if timestamps is not None else None,
                    details={"mean": float(mean), "std": float(std), "zscore": 0.0},
                )
                for i in range(len(clean_values))
            ]

        z_scores = np.abs((clean_values - mean) / std)
        results: list[AnomalyResult] = []

        for i, (value, zscore) in enumerate(zip(clean_values, z_scores)):
            is_anomaly = bool(zscore > self.threshold)
            # Normalize score to 0-1 range (cap at 2x threshold)
            normalized_score = min(float(zscore) / (self.threshold * 2), 1.0)
            severity = Severity.from_score(normalized_score) if is_anomaly else Severity.LOW

            results.append(
                AnomalyResult(
                    index=i,
                    value=float(value),
                    is_anomaly=is_anomaly,
                    score=normalized_score,
                    severity=severity,
                    method=self.name,
                    metric_name=metric_name,
                    timestamp=float(timestamps[i]) if timestamps is not None else None,
                    details={
                        "mean": float(mean),
                        "std": float(std),
                        "zscore": float(zscore),
                        "threshold": self.threshold,
                    },
                )
            )

        anomaly_count = sum(1 for r in results if r.is_anomaly)
        logger.info(
            "Z-Score detection found %d anomalies in %d data points for %s",
            anomaly_count,
            len(results),
            metric_name,
        )
        return results
