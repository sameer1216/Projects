"""Splunk REST API client for querying logs and metrics."""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import requests

from anomaly_detection.config import SplunkConfig

logger = logging.getLogger(__name__)


@dataclass
class SplunkMetricPoint:
    """A single metric data point from Splunk."""

    timestamp: float
    metric_name: str
    value: float
    dimensions: dict[str, str] = field(default_factory=dict)


class SplunkClient:
    """Client for interacting with the Splunk REST API.

    Supports both token-based and username/password authentication.
    Provides methods to run searches and extract time-series metrics.
    """

    def __init__(self, config: SplunkConfig) -> None:
        self.config = config
        self.base_url = f"{config.host}:{config.port}"
        self.session = requests.Session()
        self.session.verify = config.verify_ssl
        self._session_key: Optional[str] = None

        if config.token:
            self.session.headers["Authorization"] = f"Bearer {config.token}"
        elif config.username and config.password:
            self._authenticate()

    def _authenticate(self) -> None:
        """Authenticate with Splunk using username/password and obtain a session key."""
        url = f"{self.base_url}/services/auth/login"
        payload = {
            "username": self.config.username,
            "password": self.config.password,
            "output_mode": "json",
        }
        try:
            response = self.session.post(url, data=payload)
            response.raise_for_status()
            self._session_key = response.json()["sessionKey"]
            self.session.headers["Authorization"] = f"Splunk {self._session_key}"
            logger.info("Successfully authenticated with Splunk")
        except requests.RequestException as e:
            logger.error("Failed to authenticate with Splunk: %s", e)
            raise ConnectionError(f"Splunk authentication failed: {e}") from e

    def search(
        self,
        query: str,
        earliest_time: Optional[str] = None,
        latest_time: Optional[str] = None,
        max_results: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Execute a Splunk search query and return results.

        Args:
            query: SPL search query string.
            earliest_time: Override earliest time for the search.
            latest_time: Override latest time for the search.
            max_results: Override max results to return.

        Returns:
            List of result dictionaries from the search.
        """
        search_params = {
            "search": f"search {query}" if not query.startswith("search ") else query,
            "earliest_time": earliest_time or self.config.search.earliest_time,
            "latest_time": latest_time or self.config.search.latest_time,
            "output_mode": "json",
            "exec_mode": "blocking",
            "max_count": max_results or self.config.search.max_results,
        }

        try:
            url = f"{self.base_url}/services/search/jobs"
            response = self.session.post(url, data=search_params)
            response.raise_for_status()
            job_sid = response.json()["sid"]
            logger.info("Created search job: %s", job_sid)

            return self._get_search_results(job_sid)
        except requests.RequestException as e:
            logger.error("Splunk search failed: %s", e)
            raise

    def _get_search_results(self, sid: str) -> list[dict[str, Any]]:
        """Retrieve results from a completed search job.

        Args:
            sid: The search job ID.

        Returns:
            List of result dictionaries.
        """
        url = f"{self.base_url}/services/search/jobs/{sid}/results"
        params = {"output_mode": "json", "count": 0}

        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            results: list[dict[str, Any]] = data.get("results", [])
            logger.info("Retrieved %d results from search job %s", len(results), sid)
            return results
        except requests.RequestException as e:
            logger.error("Failed to retrieve search results: %s", e)
            raise

    def get_metrics(
        self,
        metric_query: str,
        earliest_time: Optional[str] = None,
        latest_time: Optional[str] = None,
    ) -> list[SplunkMetricPoint]:
        """Query Splunk for time-series metrics data.

        This method runs a metrics-oriented SPL query and parses the
        results into SplunkMetricPoint objects suitable for anomaly detection.

        Args:
            metric_query: SPL query that returns time-series data with
                          _time and numeric value fields.
            earliest_time: Override earliest time for the search.
            latest_time: Override latest time for the search.

        Returns:
            List of SplunkMetricPoint objects.
        """
        results = self.search(
            query=metric_query,
            earliest_time=earliest_time,
            latest_time=latest_time,
        )
        return self._parse_metric_results(results)

    def _parse_metric_results(self, results: list[dict[str, Any]]) -> list[SplunkMetricPoint]:
        """Parse raw Splunk search results into metric data points.

        Args:
            results: Raw search result dictionaries.

        Returns:
            List of parsed SplunkMetricPoint objects.
        """
        metrics: list[SplunkMetricPoint] = []

        for result in results:
            timestamp_str = result.get("_time", "")
            if not timestamp_str:
                continue

            try:
                timestamp = self._parse_timestamp(timestamp_str)
            except ValueError:
                logger.warning("Could not parse timestamp: %s", timestamp_str)
                continue

            # Extract numeric fields as metric values (skip internal fields)
            for key, value in result.items():
                if key.startswith("_"):
                    continue
                try:
                    numeric_value = float(value)
                    dimensions = {
                        k: str(v)
                        for k, v in result.items()
                        if k.startswith("_") and k not in ("_time", "_raw")
                    }
                    metrics.append(
                        SplunkMetricPoint(
                            timestamp=timestamp,
                            metric_name=key,
                            value=numeric_value,
                            dimensions=dimensions,
                        )
                    )
                except (ValueError, TypeError):
                    continue

        logger.info("Parsed %d metric data points from Splunk results", len(metrics))
        return metrics

    @staticmethod
    def _parse_timestamp(timestamp_str: str) -> float:
        """Parse a Splunk timestamp string to epoch seconds.

        Args:
            timestamp_str: Timestamp string in ISO format or epoch.

        Returns:
            Epoch timestamp as float.
        """
        # Try epoch format first
        try:
            return float(timestamp_str)
        except ValueError:
            pass

        # Try ISO format
        from datetime import datetime, timezone

        formats = [
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
        ]
        for fmt in formats:
            try:
                dt = datetime.strptime(timestamp_str, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.timestamp()
            except ValueError:
                continue

        raise ValueError(f"Unable to parse timestamp: {timestamp_str}")

    def get_error_rate_metrics(
        self,
        index: str = "main",
        earliest_time: Optional[str] = None,
        latest_time: Optional[str] = None,
    ) -> list[SplunkMetricPoint]:
        """Get error rate metrics from Splunk logs.

        Args:
            index: Splunk index to search.
            earliest_time: Override earliest time.
            latest_time: Override latest time.

        Returns:
            List of metric data points for error rates.
        """
        query = (
            f'index={index} (status>=400 OR level=ERROR OR log_level=ERROR) '
            f'| bucket _time span=5m '
            f'| stats count as error_count by _time '
            f'| sort _time'
        )
        return self.get_metrics(query, earliest_time, latest_time)

    def get_response_time_metrics(
        self,
        index: str = "main",
        earliest_time: Optional[str] = None,
        latest_time: Optional[str] = None,
    ) -> list[SplunkMetricPoint]:
        """Get response time metrics from Splunk logs.

        Args:
            index: Splunk index to search.
            earliest_time: Override earliest time.
            latest_time: Override latest time.

        Returns:
            List of metric data points for response times.
        """
        query = (
            f'index={index} response_time=* '
            f'| bucket _time span=5m '
            f'| stats avg(response_time) as avg_response_time '
            f'p95(response_time) as p95_response_time '
            f'max(response_time) as max_response_time by _time '
            f'| sort _time'
        )
        return self.get_metrics(query, earliest_time, latest_time)

    def get_throughput_metrics(
        self,
        index: str = "main",
        earliest_time: Optional[str] = None,
        latest_time: Optional[str] = None,
    ) -> list[SplunkMetricPoint]:
        """Get throughput (request count) metrics from Splunk logs.

        Args:
            index: Splunk index to search.
            earliest_time: Override earliest time.
            latest_time: Override latest time.

        Returns:
            List of metric data points for throughput.
        """
        query = (
            f'index={index} '
            f'| bucket _time span=5m '
            f'| stats count as request_count by _time '
            f'| sort _time'
        )
        return self.get_metrics(query, earliest_time, latest_time)

    def test_connection(self) -> bool:
        """Test the connection to the Splunk instance.

        Returns:
            True if connection is successful, False otherwise.
        """
        try:
            url = f"{self.base_url}/services/server/info"
            params = {"output_mode": "json"}
            response = self.session.get(url, params=params)
            response.raise_for_status()
            logger.info("Splunk connection test successful")
            return True
        except requests.RequestException as e:
            logger.error("Splunk connection test failed: %s", e)
            return False
