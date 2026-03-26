"""Tests for the Splunk client."""

import pytest

from anomaly_detection.clients.splunk_client import SplunkClient, SplunkMetricPoint
from anomaly_detection.config import SplunkConfig


class TestSplunkMetricPoint:
    """Tests for SplunkMetricPoint data class."""

    def test_creation(self) -> None:
        point = SplunkMetricPoint(
            timestamp=1700000000.0,
            metric_name="error_count",
            value=42.0,
        )
        assert point.timestamp == 1700000000.0
        assert point.metric_name == "error_count"
        assert point.value == 42.0
        assert point.dimensions == {}

    def test_with_dimensions(self) -> None:
        point = SplunkMetricPoint(
            timestamp=1700000000.0,
            metric_name="response_time",
            value=150.5,
            dimensions={"host": "web01", "index": "main"},
        )
        assert point.dimensions["host"] == "web01"


class TestSplunkTimestampParsing:
    """Tests for timestamp parsing in Splunk client."""

    def test_parse_epoch(self) -> None:
        result = SplunkClient._parse_timestamp("1700000000.0")
        assert result == 1700000000.0

    def test_parse_iso_format(self) -> None:
        result = SplunkClient._parse_timestamp("2023-11-14T22:13:20")
        assert isinstance(result, float)
        assert result > 0

    def test_parse_iso_with_timezone(self) -> None:
        result = SplunkClient._parse_timestamp("2023-11-14T22:13:20+00:00")
        assert isinstance(result, float)

    def test_parse_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="Unable to parse timestamp"):
            SplunkClient._parse_timestamp("not-a-timestamp")

    def test_parse_metric_results(self) -> None:
        config = SplunkConfig(host="https://test", token="fake-token")
        # Use __new__ to avoid authentication
        client = object.__new__(SplunkClient)
        client.config = config

        results = [
            {"_time": "1700000000.0", "error_count": "42", "_raw": "raw data"},
            {"_time": "1700000300.0", "error_count": "15", "_raw": "raw data"},
        ]

        metrics = client._parse_metric_results(results)
        assert len(metrics) == 2
        assert metrics[0].metric_name == "error_count"
        assert metrics[0].value == 42.0
        assert metrics[1].value == 15.0
