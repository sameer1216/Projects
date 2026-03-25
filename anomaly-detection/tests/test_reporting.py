"""Tests for reporting and alerting modules."""

import json

from anomaly_detection.config import ReportingConfig
from anomaly_detection.detectors.base import AnomalyResult, Severity
from anomaly_detection.pipeline import PipelineResult
from anomaly_detection.reporting import ReportGenerator


def _make_pipeline_result(anomaly_count: int = 2) -> PipelineResult:
    """Create a sample PipelineResult for testing."""
    consensus = [
        AnomalyResult(
            index=i,
            value=float(100 + i * 50),
            is_anomaly=i < anomaly_count,
            score=0.8 if i < anomaly_count else 0.1,
            severity=Severity.HIGH if i < anomaly_count else Severity.LOW,
            method="consensus",
            metric_name="test_metric",
            timestamp=1700000000.0 + i * 300,
        )
        for i in range(10)
    ]

    return PipelineResult(
        metric_name="test_metric",
        source="splunk",
        total_points=10,
        anomaly_count=anomaly_count,
        anomaly_results=[],
        consensus_anomalies=consensus,
        methods_used=["zscore", "iqr"],
    )


class TestReportGenerator:
    """Tests for the report generator."""

    def test_console_report(self) -> None:
        config = ReportingConfig(output_format="console")
        generator = ReportGenerator(config)
        result = _make_pipeline_result()

        report = generator.generate([result])
        assert "ANOMALY DETECTION REPORT" in report
        assert "test_metric" in report

    def test_json_report(self) -> None:
        config = ReportingConfig(output_format="json")
        generator = ReportGenerator(config)
        result = _make_pipeline_result()

        report = generator.generate([result])
        data = json.loads(report)
        assert "summary" in data
        assert data["summary"]["total_anomalies"] == 2
        assert len(data["metrics"]) == 1

    def test_html_report(self) -> None:
        config = ReportingConfig(output_format="html")
        generator = ReportGenerator(config)
        result = _make_pipeline_result()

        report = generator.generate([result])
        assert "<!DOCTYPE html>" in report
        assert "test_metric" in report

    def test_min_severity_filter(self) -> None:
        config = ReportingConfig(output_format="json", min_severity="high")
        generator = ReportGenerator(config)
        result = _make_pipeline_result()

        report = generator.generate([result])
        data = json.loads(report)
        # Only high+ severity anomalies should be included
        for metric in data["metrics"]:
            for anomaly in metric["anomalies"]:
                assert anomaly["severity"] in ("high", "critical")

    def test_no_anomalies(self) -> None:
        config = ReportingConfig(output_format="console")
        generator = ReportGenerator(config)
        result = _make_pipeline_result(anomaly_count=0)

        report = generator.generate([result])
        assert "Total Anomalies: 0" in report
