import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | None = None) -> dict:
    """Load config. Order: explicit path, then $SENTIMENT_CONFIG, then config/config.yaml."""
    chosen = path or os.environ.get("SENTIMENT_CONFIG")
    cfg_path = Path(chosen) if chosen else ROOT / "config" / "config.yaml"
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def base_uri(cfg: dict) -> str:
    """Storage base as a URI Spark understands (hdfs://... or file:///...)."""
    base = cfg["paths"]["base"]
    if "://" in base:
        return base.rstrip("/")
    return (ROOT / base).resolve().as_uri()


def data_path(cfg: dict, *parts: str) -> str:
    return "/".join([base_uri(cfg), *parts])


def results_dir(cfg: dict) -> Path:
    d = ROOT / cfg["paths"]["results"]
    d.mkdir(parents=True, exist_ok=True)
    return d
