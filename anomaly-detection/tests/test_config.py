"""Tests for configuration management."""

import os
import tempfile

import pytest
import yaml

from anomaly_detection.config import (
    AppConfig,
    DetectionConfig,
    load_config,
)


class TestLoadConfig:
    """Tests for configuration loading."""

    def test_default_config(self) -> None:
        config = load_config()
        assert isinstance(config, AppConfig)
        assert config.splunk.port == 8089
        assert config.detection.zscore_threshold == 3.0

    def test_load_from_yaml(self) -> None:
        config_data = {
            "splunk": {
                "host": "https://splunk.example.com",
                "port": 9089,
                "username": "admin",
                "password": "secret",
            },
            "detection": {
                "methods": ["zscore"],
                "zscore_threshold": 2.5,
            },
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(config_data, f)
            f.flush()
            config = load_config(f.name)

        os.unlink(f.name)

        assert config.splunk.host == "https://splunk.example.com"
        assert config.splunk.port == 9089
        assert config.detection.zscore_threshold == 2.5
        assert config.detection.methods == ["zscore"]

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")

    def test_partial_config(self) -> None:
        config_data = {"detection": {"zscore_threshold": 4.0}}

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(config_data, f)
            f.flush()
            config = load_config(f.name)

        os.unlink(f.name)

        assert config.detection.zscore_threshold == 4.0
        # Other defaults should be preserved
        assert config.splunk.port == 8089
        assert config.reporting.output_format == "console"


class TestDetectionConfig:
    """Tests for detection configuration."""

    def test_defaults(self) -> None:
        config = DetectionConfig()
        assert "zscore" in config.methods
        assert "iqr" in config.methods
        assert "isolation_forest" in config.methods
        assert config.consensus_mode == "majority"
