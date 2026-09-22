"""Project configuration: one YAML file, resolved relative to the project root."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]

ID_COLS = ["sendTime", "sender", "senderPseudo", "messageID"]
TARGET = "class"
RAW_FEATURES = [f"{g}{a}{s}" for g in ("pos", "spd", "acl", "hed") for s in ("", "_n") for a in ("x", "y")]
UNITS = {"pos": "m", "spd": "m/s", "acl": "m/s^2", "hed": "unit vector component"}


@lru_cache(maxsize=1)
def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["classes"] = {int(k): v for k, v in cfg["classes"].items()}
    return cfg


def path(key: str) -> Path:
    p = ROOT / load_config()["paths"][key]
    (p if p.suffix == "" else p.parent).mkdir(parents=True, exist_ok=True)
    return p


def class_names() -> dict[int, str]:
    return {k: v["name"] for k, v in load_config()["classes"].items()}


def load_dotenv() -> None:
    """Minimal .env reader (KEY=VALUE lines) so no extra dependency is needed."""
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            if v.strip():
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
