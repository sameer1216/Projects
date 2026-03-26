"""Tests for anomaly detection algorithms."""

import numpy as np

from anomaly_detection.detectors.base import Severity
from anomaly_detection.detectors.iqr import IQRDetector
from anomaly_detection.detectors.isolation_forest import IsolationForestDetector
from anomaly_detection.detectors.zscore import ZScoreDetector


class TestZScoreDetector:
    """Tests for the Z-Score anomaly detector."""

    def test_detects_obvious_outliers(self) -> None:
        values = np.array([10, 11, 10, 12, 11, 10, 11, 10, 100, 11])
        detector = ZScoreDetector(threshold=2.0)
        results = detector.detect(values, metric_name="test_metric")

        assert len(results) == 10
        anomalies = [r for r in results if r.is_anomaly]
        assert len(anomalies) >= 1
        # The value 100 should be detected
        anomaly_values = {r.value for r in anomalies}
        assert 100.0 in anomaly_values

    def test_no_anomalies_in_uniform_data(self) -> None:
        values = np.array([10.0] * 20)
        detector = ZScoreDetector(threshold=3.0)
        results = detector.detect(values, metric_name="uniform")

        anomalies = [r for r in results if r.is_anomaly]
        assert len(anomalies) == 0

    def test_empty_data(self) -> None:
        values = np.array([1.0])
        detector = ZScoreDetector()
        results = detector.detect(values)
        assert results == []

    def test_with_timestamps(self) -> None:
        values = np.array([10, 11, 10, 12, 100, 10])
        timestamps = np.array([1000, 1001, 1002, 1003, 1004, 1005])
        detector = ZScoreDetector(threshold=2.0)
        results = detector.detect(values, timestamps=timestamps)

        for r in results:
            assert r.timestamp is not None

    def test_method_name(self) -> None:
        detector = ZScoreDetector()
        assert detector.name == "zscore"

    def test_severity_assignment(self) -> None:
        values = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 200])
        detector = ZScoreDetector(threshold=2.0)
        results = detector.detect(values)

        anomalies = [r for r in results if r.is_anomaly]
        assert all(a.severity != Severity.LOW for a in anomalies)


class TestIQRDetector:
    """Tests for the IQR anomaly detector."""

    def test_detects_outliers(self) -> None:
        values = np.array([10, 11, 10, 12, 11, 10, 11, 10, 100, 11])
        detector = IQRDetector(multiplier=1.5)
        results = detector.detect(values, metric_name="test_metric")

        assert len(results) == 10
        anomalies = [r for r in results if r.is_anomaly]
        assert len(anomalies) >= 1
        anomaly_values = {r.value for r in anomalies}
        assert 100.0 in anomaly_values

    def test_detects_low_outliers(self) -> None:
        values = np.array([50, 51, 50, 52, 51, 50, 51, 50, -100, 51])
        detector = IQRDetector(multiplier=1.5)
        results = detector.detect(values)

        anomalies = [r for r in results if r.is_anomaly]
        assert len(anomalies) >= 1
        anomaly_values = {r.value for r in anomalies}
        assert -100.0 in anomaly_values

    def test_no_anomalies_in_tight_data(self) -> None:
        values = np.array([10.0, 10.1, 9.9, 10.2, 9.8, 10.0, 10.1, 9.9])
        detector = IQRDetector(multiplier=1.5)
        results = detector.detect(values)

        anomalies = [r for r in results if r.is_anomaly]
        assert len(anomalies) == 0

    def test_method_name(self) -> None:
        detector = IQRDetector()
        assert detector.name == "iqr"


class TestIsolationForestDetector:
    """Tests for the Isolation Forest anomaly detector."""

    def test_detects_outliers_in_large_dataset(self) -> None:
        rng = np.random.default_rng(42)
        normal_data = rng.normal(50, 5, 100)
        # Inject clear outliers
        outlier_data = np.array([200, 250, -50])
        values = np.concatenate([normal_data, outlier_data])

        detector = IsolationForestDetector(contamination=0.05)
        results = detector.detect(values, metric_name="test")

        anomalies = [r for r in results if r.is_anomaly]
        assert len(anomalies) >= 1

        # At least some of our injected outliers should be detected
        anomaly_indices = {r.index for r in anomalies}
        injected_indices = {100, 101, 102}
        assert len(anomaly_indices & injected_indices) >= 1

    def test_insufficient_data(self) -> None:
        values = np.array([1.0, 2.0, 3.0])
        detector = IsolationForestDetector(contamination=0.05)
        results = detector.detect(values)
        assert results == []

    def test_method_name(self) -> None:
        detector = IsolationForestDetector()
        assert detector.name == "isolation_forest"


class TestSeverity:
    """Tests for the Severity enum."""

    def test_from_score_critical(self) -> None:
        assert Severity.from_score(0.95) == Severity.CRITICAL

    def test_from_score_high(self) -> None:
        assert Severity.from_score(0.75) == Severity.HIGH

    def test_from_score_medium(self) -> None:
        assert Severity.from_score(0.5) == Severity.MEDIUM

    def test_from_score_low(self) -> None:
        assert Severity.from_score(0.2) == Severity.LOW
