"""Interquartile Range (IQR) based anomaly detection."""

import logging
from typing import Optional

import numpy as np

from anomaly_detection.detectors.base import AnomalyResult, BaseDetector, Severity

logger = logging.getLogger(__name__)


class IQRDetector(BaseDetector):
    """Detect anomalies using the Interquartile Range (IQR) method.

    A data point is flagged as anomalous if it falls outside the range
    [Q1 - multiplier * IQR, Q3 + multiplier * IQR], where Q1 and Q3
    are the first and third quartiles.
    """

    def __init__(self, multiplier: float = 1.5) -> None:
        """Initialize the IQR detector.

        Args:
            multiplier: Multiplier for the IQR to determine the
                        outlier threshold. Default is 1.5.
        """
        self.multiplier = multiplier

    @property
    def name(self) -> str:
        return "iqr"

    def detect(
        self,
        values: np.ndarray,
        metric_name: str = "",
        timestamps: Optional[np.ndarray] = None,
    ) -> list[AnomalyResult]:
        """Detect anomalies using IQR method.

        Args:
            values: Array of numeric values to analyze.
            metric_name: Name of the metric being analyzed.
            timestamps: Optional array of timestamps.

        Returns:
            List of AnomalyResult for each data point.
        """
        if not self._validate_input(values):
            logger.warning("Insufficient data for IQR detection (need >= 3 points)")
            return []

        clean_values = np.nan_to_num(values, nan=0.0)
        q1 = float(np.percentile(clean_values, 25))
        q3 = float(np.percentile(clean_values, 75))
        iqr = q3 - q1

        lower_bound = q1 - self.multiplier * iqr
        upper_bound = q3 + self.multiplier * iqr

        results: list[AnomalyResult] = []

        for i, value in enumerate(clean_values):
            val = float(value)
            is_anomaly = val < lower_bound or val > upper_bound

            # Calculate distance from bounds for scoring
            if is_anomaly:
                distance = lower_bound - val if val < lower_bound else val - upper_bound
                # Normalize: distance relative to IQR, capped at 1.0
                normalized_score = min(distance / (iqr + 1e-10), 1.0) if iqr > 0 else 0.5
            else:
                normalized_score = 0.0

            severity = Severity.from_score(normalized_score) if is_anomaly else Severity.LOW

            results.append(
                AnomalyResult(
                    index=i,
                    value=val,
                    is_anomaly=is_anomaly,
                    score=normalized_score,
                    severity=severity,
                    method=self.name,
                    metric_name=metric_name,
                    timestamp=float(timestamps[i]) if timestamps is not None else None,
                    details={
                        "q1": q1,
                        "q3": q3,
                        "iqr": iqr,
                        "lower_bound": lower_bound,
                        "upper_bound": upper_bound,
                        "multiplier": self.multiplier,
                    },
                )
            )

        anomaly_count = sum(1 for r in results if r.is_anomaly)
        logger.info(
            "IQR detection found %d anomalies in %d data points for %s",
            anomaly_count,
            len(results),
            metric_name,
        )
        return results
