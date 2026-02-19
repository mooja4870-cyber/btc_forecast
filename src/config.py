import os
import sys
import yaml
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Project Root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.yaml")

class Config:
    def __init__(self, config_path=CONFIG_PATH):
        self.config_path = config_path
        self._config = self._load_config()

        # Shortcuts for common sections
        self.project = self._config.get("project", {})
        self.data_config = self._config.get("data", {})
        self.features_config = self._config.get("features", {})
        self.feature_flags = self.features_config.get("enable", {})
        self.features_asof_config = self.features_config.get("as_of", {})
        self.futures_feature_config = self.features_config.get("futures", {})
        self.rates_expectation_config = self.features_config.get("rates_expectation", {})
        self.geopolitical_feature_config = self.features_config.get("geopolitical", {})
        self.model_config = self._config.get("modeling", {})
        self.promotion_config = self._config.get("promotion", {})
        self.monitoring = self._config.get("monitoring", {})
        self.logging_config = self._config.get("logging", {})

        # Expose important paths (construct absolute paths)
        self.data_dir = os.path.join(PROJECT_ROOT, "data")
        self.raw_dir = os.path.join(PROJECT_ROOT, self.data_config.get("raw_dir", "data/raw"))
        self.processed_dir = os.path.join(PROJECT_ROOT, self.data_config.get("processed_dir", "data/processed"))
        self.models_dir = os.path.join(PROJECT_ROOT, self.model_config.get("artifact_dir", "models"))
        
        # Monitoring paths
        self.reports_dir = os.path.join(PROJECT_ROOT, self.monitoring.get("report_dir", "data/reports"))
        self.logs_dir = os.path.join(PROJECT_ROOT, self.monitoring.get("logs_dir", "data/logs"))
        
        # Ensure directories exist
        for d in [self.raw_dir, self.processed_dir, self.models_dir, self.reports_dir, self.logs_dir]:
            os.makedirs(d, exist_ok=True)

    def _load_config(self):
        """Load YAML config file."""
        if not os.path.exists(self.config_path):
            logger.error(f"Config file not found at {self.config_path}")
            sys.exit(1)
        
        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config file: {e}")
            sys.exit(1)

    def get(self, key, default=None):
        """Get a value from the config using dot notation (e.g., 'data.tickers.btc')."""
        keys = key.split(".")
        val = self._config
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k)
            else:
                return default
        return val if val is not None else default

    def feature_enabled(self, key: str, default: bool = False) -> bool:
        """Get feature-group enabled flag from features.enable."""
        return bool(self.feature_flags.get(key, default))

# Global instance
cfg = Config()

if __name__ == "__main__":
    # Test loading
    print(f"Project: {cfg.project.get('name')}")
    print(f"Raw Data Dir: {cfg.raw_dir}")
    print(f"BTC Ticker: {cfg.get('data.tickers.btc')}")
