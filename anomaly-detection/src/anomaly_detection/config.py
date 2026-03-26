"""Configuration management for the anomaly detection system."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class SplunkSearchConfig:
    """Splunk search configuration."""

    earliest_time: str = "-1h"
    latest_time: str = "now"
    max_results: int = 10000


@dataclass
class SplunkConfig:
    """Splunk connection configuration."""

    host: str = "https://localhost"
    port: int = 8089
    username: str = ""
    password: str = ""
    token: Optional[str] = None
    verify_ssl: bool = True
    search: SplunkSearchConfig = field(default_factory=SplunkSearchConfig)


@dataclass
class AppDynamicsConfig:
    """AppDynamics connection configuration."""

    controller_url: str = "https://localhost"
    account_name: str = ""
    username: str = ""
    password: str = ""
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    application_name: str = ""
    time_range_minutes: int = 60


@dataclass
class DetectionConfig:
    """Anomaly detection configuration."""

    methods: list[str] = field(default_factory=lambda: ["zscore", "iqr", "isolation_forest"])
    zscore_threshold: float = 3.0
    iqr_multiplier: float = 1.5
    isolation_forest_contamination: float = 0.05
    min_data_points: int = 10
    consensus_mode: str = "majority"


@dataclass
class ReportingConfig:
    """Reporting configuration."""

    output_format: str = "console"
    output_file: Optional[str] = None
    min_severity: str = "low"


@dataclass
class AlertingConfig:
    """Alerting configuration."""

    enabled: bool = False
    webhook_url: Optional[str] = None
    min_alert_severity: str = "high"


@dataclass
class AppConfig:
    """Root application configuration."""

    splunk: SplunkConfig = field(default_factory=SplunkConfig)
    appdynamics: AppDynamicsConfig = field(default_factory=AppDynamicsConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    alerting: AlertingConfig = field(default_factory=AlertingConfig)


def _dict_to_dataclass(cls: type, data: dict[str, Any]) -> Any:
    """Recursively convert a dictionary to a dataclass instance."""
    if not data:
        return cls()

    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    kwargs: dict[str, Any] = {}

    for key, value in data.items():
        if key not in field_types:
            continue
        field_type = field_types[key]
        if isinstance(value, dict) and hasattr(field_type, "__dataclass_fields__"):
            kwargs[key] = _dict_to_dataclass(field_type, value)
        else:
            kwargs[key] = value

    return cls(**kwargs)


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.
                     If None, returns default configuration.

    Returns:
        AppConfig instance with loaded settings.
    """
    if config_path is None:
        return AppConfig()

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(path) as f:
        raw_config = yaml.safe_load(f) or {}

    config = AppConfig()

    if "splunk" in raw_config:
        splunk_data = raw_config["splunk"]
        search_data = splunk_data.pop("search", {})
        config.splunk = _dict_to_dataclass(SplunkConfig, splunk_data)
        if search_data:
            config.splunk.search = _dict_to_dataclass(SplunkSearchConfig, search_data)

    if "appdynamics" in raw_config:
        config.appdynamics = _dict_to_dataclass(AppDynamicsConfig, raw_config["appdynamics"])

    if "detection" in raw_config:
        config.detection = _dict_to_dataclass(DetectionConfig, raw_config["detection"])

    if "reporting" in raw_config:
        config.reporting = _dict_to_dataclass(ReportingConfig, raw_config["reporting"])

    if "alerting" in raw_config:
        config.alerting = _dict_to_dataclass(AlertingConfig, raw_config["alerting"])

    return config
