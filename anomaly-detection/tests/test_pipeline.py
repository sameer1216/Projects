"""Tests for the anomaly detection pipeline."""

import numpy as np

from anomaly_detection.config import AppConfig
from anomaly_detection.pipeline import AnomalyDetectionPipeline, MetricSeries


class TestAnomalyDetectionPipeline:
    """Tests for the unified anomaly detection pipeline."""

    def _make_pipeline(self, **detection_kwargs: object) -> AnomalyDetectionPipeline:
        config = AppConfig()
        for key, value in detection_kwargs.items():
            setattr(config.detection, key, value)
        return AnomalyDetectionPipeline(config)

    def test_run_on_custom_data_detects_anomalies(self) -> None:
        pipeline = self._make_pipeline(min_data_points=5)
        rng = np.random.default_rng(42)
        values = rng.normal(100, 10, 50).tolist()
        # Inject anomalies
        values[10] = 500
        values[20] = -200
        values[30] = 800

        results = pipeline.run_on_custom_data(
            name="test_metric",
            values=values,
            source="test",
        )

        assert len(results) == 1
        result = results[0]
        assert result.metric_name == "test_metric"
        assert result.source == "test"
        assert result.total_points == 50
        assert result.anomaly_count >= 1

    def test_consensus_majority_mode(self) -> None:
        pipeline = self._make_pipeline(
            consensus_mode="majority",
            min_data_points=5,
        )

        values = list(range(50))
        values[25] = 9999  # Clear outlier

        results = pipeline.run_on_custom_data("test", values)
        assert len(results) == 1

        consensus = results[0].consensus_anomalies
        anomalies = [a for a in consensus if a.is_anomaly]
        # The extreme outlier should be flagged by majority
        assert len(anomalies) >= 1

    def test_consensus_any_mode(self) -> None:
        pipeline = self._make_pipeline(
            consensus_mode="any",
            min_data_points=5,
        )

        rng = np.random.default_rng(42)
        values = rng.normal(50, 5, 50).tolist()
        values[10] = 200

        results = pipeline.run_on_custom_data("test", values)
        assert len(results) == 1
        assert results[0].anomaly_count >= 1

    def test_skips_series_with_too_few_points(self) -> None:
        pipeline = self._make_pipeline(min_data_points=100)

        values = [1.0, 2.0, 3.0]
        results = pipeline.run_on_custom_data("test", values)
        assert len(results) == 0

    def test_run_detection_multiple_series(self) -> None:
        pipeline = self._make_pipeline(min_data_points=5)

        series_list = [
            MetricSeries(
                name="metric_a",
                source="splunk",
                values=list(range(50)),
            ),
            MetricSeries(
                name="metric_b",
                source="appdynamics",
                values=list(range(50)),
            ),
        ]

        # Inject anomalies
        series_list[0].values[25] = 9999
        series_list[1].values[25] = 9999

        results = pipeline.run_detection(series_list)
        assert len(results) == 2

    def test_methods_used_in_results(self) -> None:
        pipeline = self._make_pipeline(
            methods=["zscore", "iqr"],
            min_data_points=5,
        )

        values = list(range(50))
        results = pipeline.run_on_custom_data("test", values)
        assert len(results) == 1
        assert "zscore" in results[0].methods_used
        assert "iqr" in results[0].methods_used

    def test_with_timestamps(self) -> None:
        pipeline = self._make_pipeline(min_data_points=5)
        values = list(range(50))
        timestamps = [1000.0 + i * 60 for i in range(50)]
        values[25] = 9999

        results = pipeline.run_on_custom_data(
            "test", values, timestamps=timestamps
        )
        assert len(results) == 1


class TestMetricSeries:
    """Tests for MetricSeries data class."""

    def test_default_creation(self) -> None:
        series = MetricSeries(name="test", source="splunk")
        assert series.name == "test"
        assert series.source == "splunk"
        assert series.timestamps == []
        assert series.values == []
        assert series.metadata == {}
