"""AppDynamics REST API client for querying application performance metrics."""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import requests

from anomaly_detection.config import AppDynamicsConfig

logger = logging.getLogger(__name__)


@dataclass
class AppDynamicsMetricPoint:
    """A single metric data point from AppDynamics."""

    timestamp: float
    metric_name: str
    metric_path: str
    value: float
    count: int = 0
    min_value: float = 0.0
    max_value: float = 0.0
    current: float = 0.0
    dimensions: dict[str, str] = field(default_factory=dict)


class AppDynamicsClient:
    """Client for interacting with the AppDynamics REST API.

    Supports both username/password and OAuth client credentials authentication.
    Provides methods to query application performance metrics for anomaly detection.
    """

    def __init__(self, config: AppDynamicsConfig) -> None:
        self.config = config
        self.base_url = config.controller_url.rstrip("/")
        self.session = requests.Session()
        self._access_token: Optional[str] = None

        if config.client_id and config.client_secret:
            self._authenticate_oauth()
        elif config.username and config.password:
            self.session.auth = (
                f"{config.username}@{config.account_name}",
                config.password,
            )

    def _authenticate_oauth(self) -> None:
        """Authenticate using OAuth client credentials."""
        url = f"{self.base_url}/controller/api/oauth/access_token"
        payload = {
            "grant_type": "client_credentials",
            "client_id": f"{self.config.client_id}@{self.config.account_name}",
            "client_secret": self.config.client_secret,
        }

        try:
            response = self.session.post(url, data=payload)
            response.raise_for_status()
            token_data = response.json()
            self._access_token = token_data["access_token"]
            self.session.headers["Authorization"] = f"Bearer {self._access_token}"
            logger.info("Successfully authenticated with AppDynamics via OAuth")
        except requests.RequestException as e:
            logger.error("AppDynamics OAuth authentication failed: %s", e)
            raise ConnectionError(f"AppDynamics authentication failed: {e}") from e

    def get_metric_data(
        self,
        metric_path: str,
        time_range_minutes: Optional[int] = None,
        rollup: bool = True,
    ) -> list[AppDynamicsMetricPoint]:
        """Retrieve metric data from AppDynamics.

        Args:
            metric_path: The full metric path in AppDynamics
                         (e.g., "Overall Application Performance|Average Response Time (ms)").
            time_range_minutes: Time range in minutes to query. Defaults to config value.
            rollup: Whether to roll up metric data.

        Returns:
            List of AppDynamicsMetricPoint objects.
        """
        app_name = self.config.application_name
        duration = time_range_minutes or self.config.time_range_minutes

        url = (
            f"{self.base_url}/controller/rest/applications/{app_name}"
            f"/metric-data"
        )
        params: dict[str, Any] = {
            "metric-path": metric_path,
            "time-range-type": "BEFORE_NOW",
            "duration-in-mins": duration,
            "rollup": str(rollup).lower(),
            "output": "JSON",
        }

        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            return self._parse_metric_response(data, metric_path)
        except requests.RequestException as e:
            logger.error("Failed to get metric data from AppDynamics: %s", e)
            raise

    def _parse_metric_response(
        self,
        data: list[dict[str, Any]],
        metric_path: str,
    ) -> list[AppDynamicsMetricPoint]:
        """Parse the AppDynamics metric response into data points.

        Args:
            data: Raw JSON response from the API.
            metric_path: The metric path queried.

        Returns:
            List of parsed AppDynamicsMetricPoint objects.
        """
        metrics: list[AppDynamicsMetricPoint] = []

        for metric_entry in data:
            full_path = metric_entry.get("metricPath", metric_path)
            metric_name = metric_entry.get("metricName", full_path.split("|")[-1])
            metric_values = metric_entry.get("metricValues", [])

            for point in metric_values:
                timestamp = point.get("startTimeInMillis", 0) / 1000.0
                value = float(point.get("value", 0))
                count = int(point.get("count", 0))
                min_val = float(point.get("min", 0))
                max_val = float(point.get("max", 0))
                current = float(point.get("current", 0))

                metrics.append(
                    AppDynamicsMetricPoint(
                        timestamp=timestamp,
                        metric_name=metric_name,
                        metric_path=full_path,
                        value=value,
                        count=count,
                        min_value=min_val,
                        max_value=max_val,
                        current=current,
                    )
                )

        logger.info(
            "Parsed %d metric data points from AppDynamics for path: %s",
            len(metrics),
            metric_path,
        )
        return metrics

    def get_response_time_metrics(
        self,
        tier_name: Optional[str] = None,
        time_range_minutes: Optional[int] = None,
    ) -> list[AppDynamicsMetricPoint]:
        """Get average response time metrics.

        Args:
            tier_name: Optional tier name to filter by.
            time_range_minutes: Time range in minutes.

        Returns:
            List of metric data points for response times.
        """
        if tier_name:
            path = (
                f"Overall Application Performance|{tier_name}"
                f"|Average Response Time (ms)"
            )
        else:
            path = "Overall Application Performance|Average Response Time (ms)"

        return self.get_metric_data(path, time_range_minutes)

    def get_error_rate_metrics(
        self,
        tier_name: Optional[str] = None,
        time_range_minutes: Optional[int] = None,
    ) -> list[AppDynamicsMetricPoint]:
        """Get error rate metrics.

        Args:
            tier_name: Optional tier name to filter by.
            time_range_minutes: Time range in minutes.

        Returns:
            List of metric data points for error rates.
        """
        if tier_name:
            path = (
                f"Overall Application Performance|{tier_name}"
                f"|Errors per Minute"
            )
        else:
            path = "Overall Application Performance|Errors per Minute"

        return self.get_metric_data(path, time_range_minutes)

    def get_throughput_metrics(
        self,
        tier_name: Optional[str] = None,
        time_range_minutes: Optional[int] = None,
    ) -> list[AppDynamicsMetricPoint]:
        """Get throughput (calls per minute) metrics.

        Args:
            tier_name: Optional tier name to filter by.
            time_range_minutes: Time range in minutes.

        Returns:
            List of metric data points for throughput.
        """
        if tier_name:
            path = (
                f"Overall Application Performance|{tier_name}"
                f"|Calls per Minute"
            )
        else:
            path = "Overall Application Performance|Calls per Minute"

        return self.get_metric_data(path, time_range_minutes)

    def get_business_transaction_metrics(
        self,
        bt_name: str,
        metric_type: str = "Average Response Time (ms)",
        time_range_minutes: Optional[int] = None,
    ) -> list[AppDynamicsMetricPoint]:
        """Get metrics for a specific business transaction.

        Args:
            bt_name: Business transaction name.
            metric_type: Type of metric to retrieve.
            time_range_minutes: Time range in minutes.

        Returns:
            List of metric data points.
        """
        path = f"Business Transaction Performance|Business Transactions|{bt_name}|{metric_type}"
        return self.get_metric_data(path, time_range_minutes)

    def get_infrastructure_metrics(
        self,
        tier_name: str,
        node_name: Optional[str] = None,
        metric_type: str = "Hardware Resources|CPU|%Busy",
        time_range_minutes: Optional[int] = None,
    ) -> list[AppDynamicsMetricPoint]:
        """Get infrastructure-level metrics.

        Args:
            tier_name: Tier name.
            node_name: Optional node name for node-specific metrics.
            metric_type: Type of infrastructure metric.
            time_range_minutes: Time range in minutes.

        Returns:
            List of metric data points.
        """
        if node_name:
            path = (
                f"Application Infrastructure Performance|{tier_name}"
                f"|Individual Nodes|{node_name}|{metric_type}"
            )
        else:
            path = f"Application Infrastructure Performance|{tier_name}|{metric_type}"

        return self.get_metric_data(path, time_range_minutes)

    def list_applications(self) -> list[dict[str, Any]]:
        """List all applications in the AppDynamics controller.

        Returns:
            List of application dictionaries with id and name.
        """
        url = f"{self.base_url}/controller/rest/applications"
        params = {"output": "JSON"}

        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            apps: list[dict[str, Any]] = response.json()
            logger.info("Found %d applications in AppDynamics", len(apps))
            return apps
        except requests.RequestException as e:
            logger.error("Failed to list AppDynamics applications: %s", e)
            raise

    def list_tiers(self) -> list[dict[str, Any]]:
        """List all tiers for the configured application.

        Returns:
            List of tier dictionaries.
        """
        app_name = self.config.application_name
        url = f"{self.base_url}/controller/rest/applications/{app_name}/tiers"
        params = {"output": "JSON"}

        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            tiers: list[dict[str, Any]] = response.json()
            logger.info("Found %d tiers for application %s", len(tiers), app_name)
            return tiers
        except requests.RequestException as e:
            logger.error("Failed to list tiers: %s", e)
            raise

    def test_connection(self) -> bool:
        """Test the connection to AppDynamics.

        Returns:
            True if connection is successful, False otherwise.
        """
        try:
            apps = self.list_applications()
            logger.info("AppDynamics connection test successful, found %d apps", len(apps))
            return True
        except (requests.RequestException, ConnectionError) as e:
            logger.error("AppDynamics connection test failed: %s", e)
            return False
