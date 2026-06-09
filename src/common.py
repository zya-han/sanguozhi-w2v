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


def make_script_normalizer(opencc_config: str | None, protect: str = ""):
    """글자 단위 OpenCC 변환기 — 단, protect 글자와 《》는 변환하지 않는다.

    혼합 자형(번체+간체 오염) 코퍼스를 번체로 통일하되, 고전 의미가 s2t 타깃과
    구분되는 글자(于≠於·云≠雲·后≠後·里≠裏·征≠徵 …)는 보호한다.
    opencc_config가 None이면 항등 함수 반환.
    """
    if not opencc_config:
        return lambda s: s
    import opencc
    cc = opencc.OpenCC(opencc_config)
    protect_set = set(protect) | set("《》")
    cache: dict[str, str] = {}

    def conv_char(c: str) -> str:
        if c in protect_set:
            return c
        r = cache.get(c)
        if r is None:
            r = cc.convert(c)
            cache[c] = r
        return r

    def normalize(text: str) -> str:
        return "".join(conv_char(c) for c in text)

    return normalize


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)
