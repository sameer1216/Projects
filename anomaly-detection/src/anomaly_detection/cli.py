"""Command-line interface for the anomaly detection system."""

import logging
import sys
from typing import Optional

import click
import numpy as np

from anomaly_detection.config import load_config
from anomaly_detection.pipeline import AnomalyDetectionPipeline
from anomaly_detection.reporting import AlertManager, ReportGenerator


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the application."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@click.group()
@click.option("--config", "-c", "config_path", default=None, help="Path to config YAML file.")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging.")
@click.pass_context
def main(ctx: click.Context, config_path: Optional[str], verbose: bool) -> None:
    """Anomaly Detection System - Splunk & AppDynamics Integration.

    Detect anomalies in metrics collected from Splunk and AppDynamics
    using multiple statistical and ML-based detection methods.
    """
    setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config_path)


@main.command()
@click.option("--splunk/--no-splunk", default=True, help="Include Splunk metrics.")
@click.option("--appdynamics/--no-appdynamics", default=True, help="Include AppDynamics metrics.")
@click.option("--tier", default=None, help="AppDynamics tier name to filter.")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["console", "json", "html"]),
    default=None,
    help="Output format (overrides config).",
)
@click.option("--output", "-o", "output_file", default=None, help="Output file path.")
@click.pass_context
def run(
    ctx: click.Context,
    splunk: bool,
    appdynamics: bool,
    tier: Optional[str],
    output_format: Optional[str],
    output_file: Optional[str],
) -> None:
    """Run anomaly detection on live data from configured sources."""
    config = ctx.obj["config"]

    if output_format:
        config.reporting.output_format = output_format
    if output_file:
        config.reporting.output_file = output_file

    pipeline = AnomalyDetectionPipeline(config)
    results = pipeline.run_full_pipeline(
        include_splunk=splunk,
        include_appdynamics=appdynamics,
        tier_name=tier,
    )

    reporter = ReportGenerator(config.reporting)
    report = reporter.generate(results)

    if not config.reporting.output_file:
        click.echo(report)

    if config.alerting.enabled:
        alert_manager = AlertManager(config.alerting)
        alert_manager.send_alerts(results)

    total_anomalies = sum(r.anomaly_count for r in results)
    if total_anomalies > 0:
        sys.exit(1)


@main.command()
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["console", "json", "html"]),
    default=None,
    help="Output format.",
)
@click.option("--output", "-o", "output_file", default=None, help="Output file path.")
@click.option("--points", "-n", default=200, help="Number of data points to generate.")
@click.option("--anomaly-ratio", default=0.05, help="Ratio of injected anomalies.")
@click.pass_context
def demo(
    ctx: click.Context,
    output_format: Optional[str],
    output_file: Optional[str],
    points: int,
    anomaly_ratio: float,
) -> None:
    """Run anomaly detection on synthetic demo data.

    Generates sample metric data with injected anomalies to demonstrate
    the detection capabilities without requiring live Splunk/AppDynamics connections.
    """
    config = ctx.obj["config"]

    if output_format:
        config.reporting.output_format = output_format
    if output_file:
        config.reporting.output_file = output_file

    click.echo("Generating synthetic demo data...")
    pipeline = AnomalyDetectionPipeline(config)

    # Generate synthetic metrics mimicking real-world patterns
    demo_metrics = _generate_demo_metrics(points, anomaly_ratio)
    all_results = []

    for name, values, timestamps, source in demo_metrics:
        results = pipeline.run_on_custom_data(
            name=name,
            values=values,
            timestamps=timestamps,
            source=source,
        )
        all_results.extend(results)

    reporter = ReportGenerator(config.reporting)
    report = reporter.generate(all_results)

    if not config.reporting.output_file:
        click.echo(report)

    total_anomalies = sum(r.anomaly_count for r in all_results)
    click.echo(f"\nDemo complete: {total_anomalies} anomalies detected across all metrics.")


@main.command()
@click.pass_context
def test_connections(ctx: click.Context) -> None:
    """Test connectivity to Splunk and AppDynamics."""
    config = ctx.obj["config"]

    click.echo("Testing Splunk connection...")
    try:
        from anomaly_detection.clients.splunk_client import SplunkClient

        splunk = SplunkClient(config.splunk)
        if splunk.test_connection():
            click.echo("  Splunk: Connected successfully")
        else:
            click.echo("  Splunk: Connection failed")
    except Exception as e:
        click.echo(f"  Splunk: Error - {e}")

    click.echo("Testing AppDynamics connection...")
    try:
        from anomaly_detection.clients.appdynamics_client import AppDynamicsClient

        appd = AppDynamicsClient(config.appdynamics)
        if appd.test_connection():
            click.echo("  AppDynamics: Connected successfully")
        else:
            click.echo("  AppDynamics: Connection failed")
    except Exception as e:
        click.echo(f"  AppDynamics: Error - {e}")


def _generate_demo_metrics(
    num_points: int,
    anomaly_ratio: float,
) -> list[tuple[str, list[float], list[float], str]]:
    """Generate synthetic metric data for demo purposes.

    Returns:
        List of (name, values, timestamps, source) tuples.
    """
    rng = np.random.default_rng(42)
    base_time = 1700000000.0
    timestamps = [base_time + i * 300 for i in range(num_points)]  # 5-min intervals

    num_anomalies = max(1, int(num_points * anomaly_ratio))
    anomaly_indices = rng.choice(num_points, size=num_anomalies, replace=False)

    metrics: list[tuple[str, list[float], list[float], str]] = []

    # Splunk-style: response time (ms) with normal ~200ms, anomalies spike to 1000+
    response_times = rng.normal(200, 30, num_points).tolist()
    for idx in anomaly_indices:
        response_times[idx] = float(rng.uniform(800, 2000))
    metrics.append(("splunk_response_time_avg", response_times, timestamps, "splunk"))

    # Splunk-style: error rate with normal ~5, anomalies spike
    error_rates = rng.poisson(5, num_points).astype(float).tolist()
    for idx in anomaly_indices:
        error_rates[idx] = float(rng.uniform(30, 100))
    metrics.append(("splunk_error_rate_count", error_rates, timestamps, "splunk"))

    # AppDynamics-style: calls per minute, normal ~500
    throughput = rng.normal(500, 50, num_points).tolist()
    for idx in anomaly_indices:
        # Anomaly can be very low (outage) or very high (traffic spike)
        throughput[idx] = float(rng.choice([rng.uniform(10, 50), rng.uniform(1500, 3000)]))
    metrics.append(("appd_throughput_calls_per_min", throughput, timestamps, "appdynamics"))

    # AppDynamics-style: CPU usage %, normal ~45%
    cpu_usage = rng.normal(45, 10, num_points).tolist()
    for idx in anomaly_indices:
        cpu_usage[idx] = float(rng.uniform(90, 100))
    metrics.append(("appd_infra_cpu_busy", cpu_usage, timestamps, "appdynamics"))

    return metrics


if __name__ == "__main__":
    main()
