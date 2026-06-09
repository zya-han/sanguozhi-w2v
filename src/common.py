"""공용 유틸 — config 로딩, 경로, 시드 고정, 로깅.

모든 Stage 스크립트가 import 한다. 설정은 단일 config.yaml(저장소 루트)에서만 읽는다.
"""
from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | os.PathLike | None = None) -> dict:
    cfg_path = Path(path) if path else REPO_ROOT / "config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(cfg: dict, key: str) -> Path:
    """config['paths'][key] 를 저장소 루트 기준 절대경로로."""
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else REPO_ROOT / p


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(cfg: dict) -> None:
    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)
