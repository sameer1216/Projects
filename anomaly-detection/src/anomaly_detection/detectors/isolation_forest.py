"""Isolation Forest based anomaly detection."""

import logging
from typing import Optional

import numpy as np
from sklearn.ensemble import IsolationForest

from anomaly_detection.detectors.base import AnomalyResult, BaseDetector, Severity

logger = logging.getLogger(__name__)


class IsolationForestDetector(BaseDetector):
    """Detect anomalies using the Isolation Forest algorithm.

    Isolation Forest works by randomly selecting features and split values
    to isolate observations. Anomalies are isolated in fewer steps on average,
    resulting in shorter path lengths in the tree structure.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        random_state: int = 42,
    ) -> None:
        """Initialize the Isolation Forest detector.

        Args:
            contamination: Expected proportion of anomalies in the data.
            n_estimators: Number of trees in the forest.
            random_state: Random state for reproducibility.
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state

    @property
    def name(self) -> str:
        return "isolation_forest"

    def detect(
        self,
        values: np.ndarray,
        metric_name: str = "",
        timestamps: Optional[np.ndarray] = None,
    ) -> list[AnomalyResult]:
        """Detect anomalies using Isolation Forest.

        Args:
            values: Array of numeric values to analyze.
            metric_name: Name of the metric being analyzed.
            timestamps: Optional array of timestamps.

        Returns:
            List of AnomalyResult for each data point.
        """
        min_points = max(10, int(1 / self.contamination) + 1)
        if not self._validate_input(values, min_points=min_points):
            logger.warning(
                "Insufficient data for Isolation Forest detection (need >= %d points)",
                min_points,
            )
            return []

        clean_values = np.nan_to_num(values, nan=0.0)
        reshaped = clean_values.reshape(-1, 1)

        model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
        )
        predictions = model.fit_predict(reshaped)
        anomaly_scores = model.decision_function(reshaped)

        # Normalize anomaly scores to 0-1 (lower decision score = more anomalous)
        score_min = float(np.min(anomaly_scores))
        score_max = float(np.max(anomaly_scores))
        score_range = score_max - score_min

        results: list[AnomalyResult] = []

        for i, (value, pred, raw_score) in enumerate(
            zip(clean_values, predictions, anomaly_scores)
        ):
            is_anomaly = int(pred) == -1

            # Invert and normalize: more negative = more anomalous = higher score
            if score_range > 0:
                normalized_score = 1.0 - (float(raw_score) - score_min) / score_range
            else:
                normalized_score = 0.0

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
                        "decision_score": float(raw_score),
                        "prediction": int(pred),
                        "contamination": self.contamination,
                    },
                )
            )

        anomaly_count = sum(1 for r in results if r.is_anomaly)
        logger.info(
            "Isolation Forest detection found %d anomalies in %d data points for %s",
            anomaly_count,
            len(results),
            metric_name,
        )
        return results
