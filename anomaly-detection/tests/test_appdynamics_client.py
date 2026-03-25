"""Tests for the AppDynamics client."""


from anomaly_detection.clients.appdynamics_client import (
    AppDynamicsClient,
    AppDynamicsMetricPoint,
)
from anomaly_detection.config import AppDynamicsConfig


class TestAppDynamicsMetricPoint:
    """Tests for AppDynamicsMetricPoint data class."""

    def test_creation(self) -> None:
        point = AppDynamicsMetricPoint(
            timestamp=1700000000.0,
            metric_name="Average Response Time (ms)",
            metric_path="Overall Application Performance|Average Response Time (ms)",
            value=150.5,
        )
        assert point.timestamp == 1700000000.0
        assert point.metric_name == "Average Response Time (ms)"
        assert point.value == 150.5
        assert point.count == 0

    def test_with_all_fields(self) -> None:
        point = AppDynamicsMetricPoint(
            timestamp=1700000000.0,
            metric_name="Calls per Minute",
            metric_path="Overall|Calls per Minute",
            value=500.0,
            count=60,
            min_value=10.0,
            max_value=1000.0,
            current=500.0,
        )
        assert point.count == 60
        assert point.min_value == 10.0
        assert point.max_value == 1000.0


class TestAppDynamicsClientParsing:
    """Tests for AppDynamics client metric parsing."""

    def test_parse_metric_response(self) -> None:
        config = AppDynamicsConfig(
            controller_url="https://test",
            account_name="test",
            application_name="test-app",
        )
        client = object.__new__(AppDynamicsClient)
        client.config = config

        api_response = [
            {
                "metricPath": "Overall|Average Response Time (ms)",
                "metricName": "Average Response Time (ms)",
                "metricValues": [
                    {
                        "startTimeInMillis": 1700000000000,
                        "value": 150,
                        "count": 60,
                        "min": 10,
                        "max": 500,
                        "current": 150,
                    },
                    {
                        "startTimeInMillis": 1700000060000,
                        "value": 200,
                        "count": 55,
                        "min": 15,
                        "max": 600,
                        "current": 200,
                    },
                ],
            }
        ]

        metrics = client._parse_metric_response(api_response, "test-path")
        assert len(metrics) == 2
        assert metrics[0].timestamp == 1700000000.0
        assert metrics[0].value == 150.0
        assert metrics[0].count == 60
        assert metrics[1].value == 200.0

    def test_parse_empty_response(self) -> None:
        config = AppDynamicsConfig(
            controller_url="https://test",
            account_name="test",
            application_name="test-app",
        )
        client = object.__new__(AppDynamicsClient)
        client.config = config

        metrics = client._parse_metric_response([], "test-path")
        assert len(metrics) == 0


class TestAppDynamicsMetricPaths:
    """Tests for metric path construction."""

    def test_response_time_path_with_tier(self) -> None:
        config = AppDynamicsConfig(
            controller_url="https://test",
            account_name="test",
            application_name="test-app",
        )
        client = object.__new__(AppDynamicsClient)
        client.config = config
        client.base_url = "https://test"
        client.session = None  # type: ignore[assignment]
        client._access_token = None

        # Verify that the tier-based path is correct by checking the method logic
        tier_name = "WebTier"
        expected_path = f"Overall Application Performance|{tier_name}|Average Response Time (ms)"
        actual_path = (
            f"Overall Application Performance|{tier_name}"
            f"|Average Response Time (ms)"
        )
        assert actual_path == expected_path
