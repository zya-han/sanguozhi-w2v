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
    """코퍼스 설정 로드. 우선순위: 인자 > $CORPUS_CONFIG > config/sanguozhi.yaml.

    다중 코퍼스: `CORPUS_CONFIG=config/houhanshu.yaml python src/01_…`.
    """
    chosen = path or os.environ.get("CORPUS_CONFIG")
    cfg_path = Path(chosen) if chosen else REPO_ROOT / "config" / "sanguozhi.yaml"
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
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


def load_variant_map(path) -> dict:
    """異體字 매핑 TSV(변이형\\t표준형) 로드. 주석(#)·빈 줄 무시."""
    if not path:
        return {}
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.exists():
        return {}
    m = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 2 and parts[0] and parts[1]:
            m[parts[0]] = parts[1]
    return m


def make_script_normalizer(opencc_config: str | None, protect: str = "",
                           variant_map: dict | None = None):
    """글자 단위 자형 정규화기.

    1) OpenCC opencc_config(예: s2t)로 簡→繁 통일. 단, protect 글자·《》는 변환 안 함
       (고전 의미가 s2t 타깃과 구분: 于≠於·云≠雲·后≠後·里≠裏·征≠徵 …).
    2) 그 뒤 variant_map으로 異體字(일본 신자체 등)를 표준형으로 통합(呉→吳·靣→面 …).
    opencc_config·variant_map 모두 비면 항등 함수.
    """
    variant_map = variant_map or {}
    protect_set = set(protect) | set("《》")
    cc = None
    if opencc_config:
        import opencc as _opencc
        cc = _opencc.OpenCC(opencc_config)
    if cc is None and not variant_map:
        return lambda s: s
    cache: dict[str, str] = {}

    def conv_char(c: str) -> str:
        r = cache.get(c)
        if r is None:
            r = c if (c in protect_set or cc is None) else cc.convert(c)
            r = variant_map.get(r, r)   # s2t 후 異體字 통합
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
