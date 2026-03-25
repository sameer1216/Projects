# Anomaly Detection System - Splunk & AppDynamics Integration

A Python-based anomaly detection system that integrates with **Splunk** and **AppDynamics** to collect application performance metrics and detect anomalies using multiple statistical and machine learning methods.

## Features

- **Multi-source data collection**: Pull metrics from both Splunk (via REST API) and AppDynamics (via REST API)
- **Multiple detection algorithms**:
  - **Z-Score**: Statistical deviation from the mean
  - **IQR (Interquartile Range)**: Robust outlier detection based on quartiles
  - **Isolation Forest**: ML-based anomaly detection using tree ensembles
- **Consensus-based detection**: Combine results from multiple methods to reduce false positives
- **Flexible reporting**: Console tables, JSON, and HTML report formats
- **Webhook alerting**: Send alerts to Slack or other webhook endpoints
- **Demo mode**: Run with synthetic data to test without live connections

## Architecture

```
anomaly-detection/
├── src/anomaly_detection/
│   ├── cli.py                  # Click-based CLI interface
│   ├── config.py               # YAML configuration management
│   ├── pipeline.py             # Unified detection pipeline
│   ├── reporting.py            # Report generation & alerting
│   ├── clients/
│   │   ├── splunk_client.py    # Splunk REST API client
│   │   └── appdynamics_client.py  # AppDynamics REST API client
│   └── detectors/
│       ├── base.py             # Base detector interface & severity
│       ├── zscore.py           # Z-Score detector
│       ├── iqr.py              # IQR detector
│       └── isolation_forest.py # Isolation Forest detector
├── tests/                      # Unit tests
├── config/config.yaml.example  # Example configuration
└── pyproject.toml              # Project configuration
```

## Installation

```bash
cd anomaly-detection
pip install -e ".[dev]"
```

## Quick Start

### 1. Demo Mode (No credentials required)

```bash
# Run with synthetic data
anomaly-detect demo

# Generate HTML report
anomaly-detect demo --format html --output report.html

# Custom data size
anomaly-detect demo --points 500 --anomaly-ratio 0.03
```

### 2. Configure Data Sources

```bash
cp config/config.yaml.example config/config.yaml
# Edit config/config.yaml with your Splunk and AppDynamics credentials
```

### 3. Test Connections

```bash
anomaly-detect -c config/config.yaml test-connections
```

### 4. Run Detection

```bash
# Run on both sources
anomaly-detect -c config/config.yaml run

# Splunk only
anomaly-detect -c config/config.yaml run --no-appdynamics

# AppDynamics only with specific tier
anomaly-detect -c config/config.yaml run --no-splunk --tier WebTier

# JSON output
anomaly-detect -c config/config.yaml run --format json --output results.json
```

## Configuration

See `config/config.yaml.example` for all available options:

- **Splunk**: Host, port, credentials (username/password or token), search parameters
- **AppDynamics**: Controller URL, account, credentials (basic or OAuth), application name
- **Detection**: Methods to use, thresholds, consensus mode
- **Reporting**: Output format, severity filters
- **Alerting**: Webhook URL, severity thresholds

## Detection Methods

| Method | Best For | How It Works |
|--------|----------|-------------|
| Z-Score | Normally distributed data | Flags points > N standard deviations from mean |
| IQR | Skewed distributions | Uses quartile-based bounds (Q1 - 1.5*IQR, Q3 + 1.5*IQR) |
| Isolation Forest | Complex patterns | ML ensemble that isolates anomalies in fewer tree splits |

### Consensus Modes

- **majority**: A point is anomalous only if the majority of methods agree (reduces false positives)
- **any**: A point is anomalous if any single method flags it (higher sensitivity)

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check src/ tests/

# Type check
mypy src/
```
