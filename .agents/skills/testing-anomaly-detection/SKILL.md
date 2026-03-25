# Testing the Anomaly Detection Application

## Overview
The anomaly detection app lives in `anomaly-detection/` and integrates with Splunk and AppDynamics via REST APIs. All functionality can be tested end-to-end using demo mode (synthetic data) without live credentials.

## Setup
```bash
cd anomaly-detection
pip install -e ".[dev]"
```

## Key CLI Commands

### Demo mode (no credentials needed)
```bash
# Console output (default)
anomaly-detect demo

# JSON output
anomaly-detect demo --format json

# HTML report to file
anomaly-detect demo --format html --output report.html

# Custom data size and anomaly ratio
anomaly-detect demo --points 50 --anomaly-ratio 0.1
```

### Live mode (requires Splunk/AppDynamics credentials)
```bash
# Copy and fill in config
cp config/config.yaml.example config/config.yaml
# Edit config.yaml with credentials
anomaly-detect run
```

### Connection testing
```bash
anomaly-detect test-connections
```
Note: This will fail without live credentials configured — that is expected behavior.

## Unit Tests
```bash
python -m pytest tests/ -v
```
Expect 47 tests covering detectors, pipeline, config, reporting, and client parsing.

## Lint and Type Checks
```bash
ruff check src/ tests/
mypy src/anomaly_detection/
```

## Expected Demo Output Behavior
- Default demo generates 4 metrics (2 Splunk-style, 2 AppDynamics-style) with 200 points each
- Anomalies are injected at 5% ratio by default (10 indices shared across all metrics)
- Console output includes tabulate grid tables with Timestamp, Value, Score, Severity columns
- JSON output is embedded between "Generating synthetic demo data..." and "Demo complete:" lines — pipe through `sed -n '/^{/,/^}/p'` to extract clean JSON
- HTML output writes directly to file when `--output` is specified; console only shows status messages
- Demo command always exits with code 0 (unlike `run` which exits 1 when anomalies are found)
- Results are deterministic (seeded with rng seed 42)

## Devin Secrets Needed
- `SPLUNK_TOKEN` or `SPLUNK_USERNAME`/`SPLUNK_PASSWORD` — only for live mode testing
- `APPDYNAMICS_CLIENT_ID`/`APPDYNAMICS_CLIENT_SECRET` or `APPDYNAMICS_USERNAME`/`APPDYNAMICS_PASSWORD` — only for live mode testing
- No secrets needed for demo mode or unit tests

## Testing Tips
- The JSON output from `--format json` mixes with CLI status messages. Use `sed -n '/^{/,/^}/p'` or redirect stderr to separate them.
- When verifying anomaly detection works correctly, check that `total_anomalies > 0` in the summary — the exact count may vary slightly if detection parameters change.
- The HTML report renders well in browsers with color-coded severity badges (CRITICAL=red, HIGH=orange, MEDIUM=yellow, LOW=green).
