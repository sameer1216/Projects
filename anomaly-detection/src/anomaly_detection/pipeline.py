"""Unified anomaly detection pipeline combining Splunk and AppDynamics data."""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from anomaly_detection.clients.appdynamics_client import (
    AppDynamicsClient,
    AppDynamicsMetricPoint,
)
from anomaly_detection.clients.splunk_client import SplunkClient, SplunkMetricPoint
from anomaly_detection.config import AppConfig
from anomaly_detection.detectors.base import AnomalyResult, BaseDetector, Severity
from anomaly_detection.detectors.iqr import IQRDetector
from anomaly_detection.detectors.isolation_forest import IsolationForestDetector
from anomaly_detection.detectors.zscore import ZScoreDetector

logger = logging.getLogger(__name__)


@dataclass
class MetricSeries:
    """A named time-series of metric values from any data source."""

    name: str
    source: str  # "splunk" or "appdynamics"
    timestamps: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Complete result from the anomaly detection pipeline."""

    metric_name: str
    source: str
    total_points: int
    anomaly_count: int
    anomaly_results: list[AnomalyResult]
    consensus_anomalies: list[AnomalyResult]
    methods_used: list[str]


class AnomalyDetectionPipeline:
    """Orchestrates data collection and anomaly detection across data sources.

    This pipeline:
    1. Collects metrics from Splunk and/or AppDynamics
    2. Runs multiple anomaly detection algorithms on each metric series
    3. Applies consensus logic to reduce false positives
    4. Produces unified results across all data sources
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.detectors = self._init_detectors()
        self._splunk_client: Optional[SplunkClient] = None
        self._appd_client: Optional[AppDynamicsClient] = None

    def _init_detectors(self) -> list[BaseDetector]:
        """Initialize anomaly detection methods based on configuration."""
        detectors: list[BaseDetector] = []
        methods = self.config.detection.methods

        if "zscore" in methods:
            detectors.append(ZScoreDetector(threshold=self.config.detection.zscore_threshold))

        if "iqr" in methods:
            detectors.append(IQRDetector(multiplier=self.config.detection.iqr_multiplier))

        if "isolation_forest" in methods:
            detectors.append(
                IsolationForestDetector(
                    contamination=self.config.detection.isolation_forest_contamination
                )
            )

        logger.info("Initialized %d detection methods: %s", len(detectors), methods)
        return detectors

    @property
    def splunk_client(self) -> SplunkClient:
        """Lazily initialize and return the Splunk client."""
        if self._splunk_client is None:
            self._splunk_client = SplunkClient(self.config.splunk)
        return self._splunk_client

    @property
    def appd_client(self) -> AppDynamicsClient:
        """Lazily initialize and return the AppDynamics client."""
        if self._appd_client is None:
            self._appd_client = AppDynamicsClient(self.config.appdynamics)
        return self._appd_client

    def collect_splunk_metrics(self) -> list[MetricSeries]:
        """Collect standard metrics from Splunk.

        Returns:
            List of MetricSeries from Splunk data.
        """
        all_series: list[MetricSeries] = []

        metric_collectors = [
            ("error_rate", self.splunk_client.get_error_rate_metrics),
            ("response_time", self.splunk_client.get_response_time_metrics),
            ("throughput", self.splunk_client.get_throughput_metrics),
        ]

        for metric_type, collector in metric_collectors:
            try:
                data_points = collector()
                series = self._splunk_points_to_series(data_points, metric_type)
                all_series.extend(series)
                logger.info(
                    "Collected %d series for Splunk %s metrics",
                    len(series),
                    metric_type,
                )
            except Exception as e:
                logger.error("Failed to collect Splunk %s metrics: %s", metric_type, e)

        return all_series

    def collect_appdynamics_metrics(
        self,
        tier_name: Optional[str] = None,
    ) -> list[MetricSeries]:
        """Collect standard metrics from AppDynamics.

        Args:
            tier_name: Optional tier name to filter metrics.

        Returns:
            List of MetricSeries from AppDynamics data.
        """
        all_series: list[MetricSeries] = []

        metric_collectors = [
            ("response_time", lambda: self.appd_client.get_response_time_metrics(tier_name)),
            ("error_rate", lambda: self.appd_client.get_error_rate_metrics(tier_name)),
            ("throughput", lambda: self.appd_client.get_throughput_metrics(tier_name)),
        ]

        for metric_type, collector in metric_collectors:
            try:
                data_points = collector()
                series = self._appd_points_to_series(data_points, metric_type)
                all_series.extend(series)
                logger.info(
                    "Collected %d series for AppDynamics %s metrics",
                    len(series),
                    metric_type,
                )
            except Exception as e:
                logger.error(
                    "Failed to collect AppDynamics %s metrics: %s", metric_type, e
                )

        return all_series

    def run_detection(self, series_list: list[MetricSeries]) -> list[PipelineResult]:
        """Run anomaly detection on a list of metric series.

        Args:
            series_list: List of MetricSeries to analyze.

        Returns:
            List of PipelineResult objects with detection results.
        """
        results: list[PipelineResult] = []

        for series in series_list:
            if len(series.values) < self.config.detection.min_data_points:
                logger.warning(
                    "Skipping %s/%s: only %d data points (min: %d)",
                    series.source,
                    series.name,
                    len(series.values),
                    self.config.detection.min_data_points,
                )
                continue

            values = np.array(series.values)
            timestamps = np.array(series.timestamps) if series.timestamps else None

            all_results: list[AnomalyResult] = []
            methods_used: list[str] = []

            for detector in self.detectors:
                try:
                    detector_results = detector.detect(
                        values=values,
                        metric_name=series.name,
                        timestamps=timestamps,
                    )
                    all_results.extend(detector_results)
                    methods_used.append(detector.name)
                except Exception as e:
                    logger.error(
                        "Detector %s failed on %s/%s: %s",
                        detector.name,
                        series.source,
                        series.name,
                        e,
                    )

            consensus = self._apply_consensus(all_results, methods_used)
            anomaly_count = sum(1 for r in consensus if r.is_anomaly)

            results.append(
                PipelineResult(
                    metric_name=series.name,
                    source=series.source,
                    total_points=len(series.values),
                    anomaly_count=anomaly_count,
                    anomaly_results=all_results,
                    consensus_anomalies=consensus,
                    methods_used=methods_used,
                )
            )

            logger.info(
                "Detection complete for %s/%s: %d anomalies out of %d points",
                series.source,
                series.name,
                anomaly_count,
                len(series.values),
            )

        return results

    def _apply_consensus(
        self,
        results: list[AnomalyResult],
        methods_used: list[str],
    ) -> list[AnomalyResult]:
        """Apply consensus logic across multiple detection methods.

        In "majority" mode, a point is only flagged if the majority of
        methods agree it's anomalous. In "any" mode, a point is flagged
        if any single method flags it.

        Args:
            results: All detection results from all methods.
            methods_used: List of method names that were used.

        Returns:
            Consensus anomaly results (one per data point).
        """
        if not results:
            return []

        # Group results by index
        by_index: dict[int, list[AnomalyResult]] = defaultdict(list)
        for result in results:
            by_index[result.index].append(result)

        consensus_mode = self.config.detection.consensus_mode
        majority_threshold = len(methods_used) / 2.0
        consensus_results: list[AnomalyResult] = []

        for index in sorted(by_index.keys()):
            point_results = by_index[index]
            anomaly_votes = sum(1 for r in point_results if r.is_anomaly)
            total_votes = len(point_results)

            if consensus_mode == "majority":
                is_anomaly = anomaly_votes > majority_threshold
            else:  # "any"
                is_anomaly = anomaly_votes > 0

            # Average the scores from methods that flagged it
            if is_anomaly:
                anomaly_scores = [r.score for r in point_results if r.is_anomaly]
                avg_score = sum(anomaly_scores) / len(anomaly_scores) if anomaly_scores else 0.0
            else:
                avg_score = 0.0

            severity = Severity.from_score(avg_score) if is_anomaly else Severity.LOW

            # Use the first result as a template for common fields
            template = point_results[0]
            consensus_results.append(
                AnomalyResult(
                    index=index,
                    value=template.value,
                    is_anomaly=is_anomaly,
                    score=avg_score,
                    severity=severity,
                    method="consensus",
                    metric_name=template.metric_name,
                    timestamp=template.timestamp,
                    details={
                        "anomaly_votes": float(anomaly_votes),
                        "total_methods": float(total_votes),
                        "consensus_mode": 1.0 if consensus_mode == "majority" else 0.0,
                    },
                )
            )

        return consensus_results

    def run_full_pipeline(
        self,
        include_splunk: bool = True,
        include_appdynamics: bool = True,
        tier_name: Optional[str] = None,
    ) -> list[PipelineResult]:
        """Run the complete anomaly detection pipeline.

        Collects metrics from configured sources and runs detection
        on all collected series.

        Args:
            include_splunk: Whether to collect Splunk metrics.
            include_appdynamics: Whether to collect AppDynamics metrics.
            tier_name: Optional AppDynamics tier name.

        Returns:
            List of PipelineResult objects.
        """
        all_series: list[MetricSeries] = []

        if include_splunk:
            try:
                splunk_series = self.collect_splunk_metrics()
                all_series.extend(splunk_series)
            except Exception as e:
                logger.error("Failed to collect Splunk metrics: %s", e)

        if include_appdynamics:
            try:
                appd_series = self.collect_appdynamics_metrics(tier_name)
                all_series.extend(appd_series)
            except Exception as e:
                logger.error("Failed to collect AppDynamics metrics: %s", e)

        if not all_series:
            logger.warning("No metric series collected from any source")
            return []

        logger.info("Collected %d metric series total, running detection", len(all_series))
        return self.run_detection(all_series)

    def run_on_custom_data(
        self,
        name: str,
        values: list[float],
        timestamps: Optional[list[float]] = None,
        source: str = "custom",
    ) -> list[PipelineResult]:
        """Run anomaly detection on custom data (useful for testing).

        Args:
            name: Name of the metric.
            values: List of metric values.
            timestamps: Optional list of timestamps.
            source: Data source name.

        Returns:
            List of PipelineResult objects.
        """
        series = MetricSeries(
            name=name,
            source=source,
            timestamps=timestamps or [],
            values=values,
        )
        return self.run_detection([series])

    @staticmethod
    def _splunk_points_to_series(
        points: list[SplunkMetricPoint],
        metric_type: str,
    ) -> list[MetricSeries]:
        """Convert Splunk metric points to MetricSeries grouped by metric name.

        Args:
            points: List of SplunkMetricPoint objects.
            metric_type: Category of the metric (e.g., "error_rate").

        Returns:
            List of MetricSeries objects.
        """
        by_name: dict[str, MetricSeries] = {}

        for point in points:
            series_name = f"splunk_{metric_type}_{point.metric_name}"
            if series_name not in by_name:
                by_name[series_name] = MetricSeries(
                    name=series_name,
                    source="splunk",
                    metadata={"metric_type": metric_type},
                )
            by_name[series_name].timestamps.append(point.timestamp)
            by_name[series_name].values.append(point.value)

        return list(by_name.values())

    @staticmethod
    def _appd_points_to_series(
        points: list[AppDynamicsMetricPoint],
        metric_type: str,
    ) -> list[MetricSeries]:
        """Convert AppDynamics metric points to MetricSeries grouped by path.

        Args:
            points: List of AppDynamicsMetricPoint objects.
            metric_type: Category of the metric.

        Returns:
            List of MetricSeries objects.
        """
        by_path: dict[str, MetricSeries] = {}

        for point in points:
            series_name = f"appd_{metric_type}_{point.metric_name}"
            if series_name not in by_path:
                by_path[series_name] = MetricSeries(
                    name=series_name,
                    source="appdynamics",
                    metadata={
                        "metric_type": metric_type,
                        "metric_path": point.metric_path,
                    },
                )
            by_path[series_name].timestamps.append(point.timestamp)
            by_path[series_name].values.append(point.value)

        return list(by_path.values())
