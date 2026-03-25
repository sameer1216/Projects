"""Reporting and alerting module for anomaly detection results."""

import json
import logging
from datetime import datetime, timezone
from typing import Any

import requests
from tabulate import tabulate

from anomaly_detection.config import AlertingConfig, ReportingConfig
from anomaly_detection.detectors.base import Severity
from anomaly_detection.pipeline import PipelineResult

logger = logging.getLogger(__name__)

SEVERITY_ORDER = {
    Severity.LOW: 0,
    Severity.MEDIUM: 1,
    Severity.HIGH: 2,
    Severity.CRITICAL: 3,
}

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html>
<head>
    <title>Anomaly Detection Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 8px; }}
        .summary {{ background: white; padding: 15px; margin: 15px 0; border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metric-card {{ background: white; padding: 15px; margin: 10px 0; border-radius: 8px;
                       box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-left: 4px solid #3498db; }}
        .anomaly-row {{ padding: 8px; margin: 5px 0; border-radius: 4px; }}
        .severity-critical {{ background: #e74c3c; color: white; }}
        .severity-high {{ background: #e67e22; color: white; }}
        .severity-medium {{ background: #f39c12; color: white; }}
        .severity-low {{ background: #95a5a6; color: white; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th {{ background: #34495e; color: white; padding: 10px; text-align: left; }}
        td {{ padding: 8px; border-bottom: 1px solid #eee; }}
        tr:hover {{ background: #f0f0f0; }}
        .stat {{ display: inline-block; margin: 0 20px; text-align: center; }}
        .stat-value {{ font-size: 2em; font-weight: bold; color: #2c3e50; }}
        .stat-label {{ color: #7f8c8d; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Anomaly Detection Report</h1>
        <p>Generated: {timestamp}</p>
    </div>
    <div class="summary">
        <h2>Summary</h2>
        <div class="stat">
            <div class="stat-value">{total_metrics}</div>
            <div class="stat-label">Metrics Analyzed</div>
        </div>
        <div class="stat">
            <div class="stat-value">{total_anomalies}</div>
            <div class="stat-label">Anomalies Found</div>
        </div>
        <div class="stat">
            <div class="stat-value">{total_points}</div>
            <div class="stat-label">Data Points</div>
        </div>
        <div class="stat">
            <div class="stat-value">{critical_count}</div>
            <div class="stat-label">Critical</div>
        </div>
    </div>
    {metric_sections}
</body>
</html>
"""

METRIC_SECTION_TEMPLATE = """\
    <div class="metric-card">
        <h3>{metric_name} ({source})</h3>
        <p>Points: {total_points} | Anomalies: {anomaly_count} |
           Methods: {methods}</p>
        {anomaly_table}
    </div>
"""


class ReportGenerator:
    """Generate anomaly detection reports in various formats."""

    def __init__(self, config: ReportingConfig) -> None:
        self.config = config
        self.min_severity = self._parse_severity(config.min_severity)

    def generate(self, results: list[PipelineResult]) -> str:
        """Generate a report from pipeline results.

        Args:
            results: List of PipelineResult objects.

        Returns:
            Formatted report string.
        """
        fmt = self.config.output_format.lower()
        if fmt == "json":
            report = self._generate_json(results)
        elif fmt == "html":
            report = self._generate_html(results)
        else:
            report = self._generate_console(results)

        if self.config.output_file:
            with open(self.config.output_file, "w") as f:
                f.write(report)
            logger.info("Report written to %s", self.config.output_file)

        return report

    def _generate_console(self, results: list[PipelineResult]) -> str:
        """Generate a console-friendly table report."""
        lines: list[str] = []
        lines.append("=" * 80)
        lines.append("ANOMALY DETECTION REPORT")
        lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
        lines.append("=" * 80)

        total_anomalies = sum(r.anomaly_count for r in results)
        total_points = sum(r.total_points for r in results)
        lines.append(f"\nMetrics Analyzed: {len(results)}")
        lines.append(f"Total Data Points: {total_points}")
        lines.append(f"Total Anomalies: {total_anomalies}")
        lines.append("")

        for result in results:
            if result.anomaly_count == 0:
                continue

            lines.append("-" * 80)
            lines.append(
                f"Metric: {result.metric_name} (Source: {result.source})"
            )
            lines.append(
                f"Points: {result.total_points} | "
                f"Anomalies: {result.anomaly_count} | "
                f"Methods: {', '.join(result.methods_used)}"
            )

            anomalies = [
                r for r in result.consensus_anomalies
                if r.is_anomaly and self._meets_severity(r.severity)
            ]

            if anomalies:
                table_data = []
                for a in anomalies:
                    timestamp_str = (
                        datetime.fromtimestamp(a.timestamp, tz=timezone.utc).strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                        if a.timestamp
                        else f"idx:{a.index}"
                    )
                    table_data.append([
                        timestamp_str,
                        f"{a.value:.4f}",
                        f"{a.score:.4f}",
                        a.severity.value.upper(),
                        a.method,
                    ])

                lines.append(
                    tabulate(
                        table_data,
                        headers=["Timestamp", "Value", "Score", "Severity", "Method"],
                        tablefmt="grid",
                    )
                )
            lines.append("")

        lines.append("=" * 80)
        return "\n".join(lines)

    def _generate_json(self, results: list[PipelineResult]) -> str:
        """Generate a JSON report."""
        report_data: dict[str, Any] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "metrics_analyzed": len(results),
                "total_data_points": sum(r.total_points for r in results),
                "total_anomalies": sum(r.anomaly_count for r in results),
            },
            "metrics": [],
        }

        for result in results:
            anomalies = [
                r for r in result.consensus_anomalies
                if r.is_anomaly and self._meets_severity(r.severity)
            ]

            metric_data: dict[str, Any] = {
                "name": result.metric_name,
                "source": result.source,
                "total_points": result.total_points,
                "anomaly_count": result.anomaly_count,
                "methods_used": result.methods_used,
                "anomalies": [
                    {
                        "index": a.index,
                        "value": a.value,
                        "score": a.score,
                        "severity": a.severity.value,
                        "timestamp": a.timestamp,
                        "details": a.details,
                    }
                    for a in anomalies
                ],
            }
            report_data["metrics"].append(metric_data)

        return json.dumps(report_data, indent=2)

    def _generate_html(self, results: list[PipelineResult]) -> str:
        """Generate an HTML report."""
        total_anomalies = sum(r.anomaly_count for r in results)
        total_points = sum(r.total_points for r in results)

        critical_count = 0
        for result in results:
            for a in result.consensus_anomalies:
                if a.is_anomaly and a.severity == Severity.CRITICAL:
                    critical_count += 1

        metric_sections = []
        for result in results:
            anomalies = [
                r for r in result.consensus_anomalies
                if r.is_anomaly and self._meets_severity(r.severity)
            ]

            if anomalies:
                rows = []
                for a in anomalies:
                    timestamp_str = (
                        datetime.fromtimestamp(a.timestamp, tz=timezone.utc).strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                        if a.timestamp
                        else f"idx:{a.index}"
                    )
                    severity_class = f"severity-{a.severity.value}"
                    rows.append(
                        f"<tr><td>{timestamp_str}</td><td>{a.value:.4f}</td>"
                        f"<td>{a.score:.4f}</td>"
                        f'<td class="anomaly-row {severity_class}">'
                        f"{a.severity.value.upper()}</td></tr>"
                    )

                table = (
                    "<table><tr><th>Timestamp</th><th>Value</th>"
                    "<th>Score</th><th>Severity</th></tr>"
                    + "\n".join(rows)
                    + "</table>"
                )
            else:
                table = "<p>No anomalies above minimum severity threshold.</p>"

            metric_sections.append(
                METRIC_SECTION_TEMPLATE.format(
                    metric_name=result.metric_name,
                    source=result.source,
                    total_points=result.total_points,
                    anomaly_count=result.anomaly_count,
                    methods=", ".join(result.methods_used),
                    anomaly_table=table,
                )
            )

        return HTML_TEMPLATE.format(
            timestamp=datetime.now(timezone.utc).isoformat(),
            total_metrics=len(results),
            total_anomalies=total_anomalies,
            total_points=total_points,
            critical_count=critical_count,
            metric_sections="\n".join(metric_sections),
        )

    def _meets_severity(self, severity: Severity) -> bool:
        """Check if a severity level meets the minimum threshold."""
        return SEVERITY_ORDER.get(severity, 0) >= SEVERITY_ORDER.get(self.min_severity, 0)

    @staticmethod
    def _parse_severity(severity_str: str) -> Severity:
        """Parse a severity string to Severity enum."""
        try:
            return Severity(severity_str.lower())
        except ValueError:
            logger.warning("Unknown severity '%s', defaulting to LOW", severity_str)
            return Severity.LOW


class AlertManager:
    """Send alerts for detected anomalies via webhooks."""

    def __init__(self, config: AlertingConfig) -> None:
        self.config = config
        self.min_severity = ReportGenerator._parse_severity(config.min_alert_severity)

    def send_alerts(self, results: list[PipelineResult]) -> int:
        """Send alerts for anomalies that meet the severity threshold.

        Args:
            results: List of PipelineResult objects.

        Returns:
            Number of alerts sent.
        """
        if not self.config.enabled:
            logger.info("Alerting is disabled")
            return 0

        if not self.config.webhook_url:
            logger.warning("No webhook URL configured for alerting")
            return 0

        alerts_sent = 0

        for result in results:
            critical_anomalies = [
                a
                for a in result.consensus_anomalies
                if a.is_anomaly
                and SEVERITY_ORDER.get(a.severity, 0)
                >= SEVERITY_ORDER.get(self.min_severity, 0)
            ]

            if not critical_anomalies:
                continue

            payload = self._build_alert_payload(result, critical_anomalies)

            try:
                response = requests.post(
                    self.config.webhook_url,
                    json=payload,
                    timeout=10,
                )
                response.raise_for_status()
                alerts_sent += 1
                logger.info(
                    "Alert sent for %s/%s (%d anomalies)",
                    result.source,
                    result.metric_name,
                    len(critical_anomalies),
                )
            except requests.RequestException as e:
                logger.error("Failed to send alert: %s", e)

        logger.info("Sent %d alerts total", alerts_sent)
        return alerts_sent

    @staticmethod
    def _build_alert_payload(
        result: PipelineResult,
        anomalies: list[Any],
    ) -> dict[str, Any]:
        """Build the webhook alert payload."""
        max_severity = max(
            (a.severity for a in anomalies),
            key=lambda s: SEVERITY_ORDER.get(s, 0),
        )

        return {
            "text": (
                f"Anomaly Alert: {result.metric_name} ({result.source})\n"
                f"Severity: {max_severity.value.upper()}\n"
                f"Anomalies: {len(anomalies)} detected out of "
                f"{result.total_points} data points\n"
                f"Methods: {', '.join(result.methods_used)}"
            ),
            "metric_name": result.metric_name,
            "source": result.source,
            "severity": max_severity.value,
            "anomaly_count": len(anomalies),
            "total_points": result.total_points,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
