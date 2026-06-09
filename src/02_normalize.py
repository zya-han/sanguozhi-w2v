"""Stage 2 — 정규화.

확정 결정(사용자): 四庫본 코퍼스는 이미 正字 번체이므로 **OpenCC 변환을 적용하지 않는다**.
근거: s2t는 고전한문에서 의미가 다른 글자를 병합한다 — 于→於(644), 云→雲(417),
咸熙→鹹熙, 里→裏, 征→徵 … 이는 명세 §2.1(허사·연어 분포 보존)을 위반한다.
자형 정합은 Stage 3에서 **가제티어를 코퍼스에 맞추는** 방식으로 달성한다.

따라서 본 단계는:
  1. (선택) 큐레이션된 異體字 매핑(원→정)만 적용. config.normalize.variant_map 지정 시.
  2. 문장/구 경계: 四庫본 WYG 판본은 표점이 없으므로 segment를 그대로 윈도 단위로 둔다
     (句讀 자동화는 비목표). 빈 텍스트 제거.
출력: data/interim/normalized.parquet (segments.parquet과 동일 스키마, 정규화된 text).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (get_logger, load_config, load_variant_map,  # noqa: E402
                    make_script_normalizer, resolve)

log = get_logger("02_normalize")


def main():
    cfg = load_config()
    norm = cfg.get("normalize", {})

    interim = resolve(cfg, "interim")
    df = pd.read_parquet(interim / "segments.parquet")

    protect = norm.get("opencc_protect", "")
    vmap = load_variant_map(norm.get("variant_map"))
    normalize = make_script_normalizer(norm.get("corpus_opencc"), protect, vmap)
    before = df["text"].copy()
    df["text"] = df["text"].map(normalize)
    n_changed = int((before != df["text"]).sum())
    log.info("자형 정규화(opencc=%s 보호 %d자, 異體字 %d쌍): %d/%d 세그먼트 변경 "
             "— 簡↔繁·異體 통일, 고전 의미 보존.",
             norm.get("corpus_opencc"), len(set(protect)), len(vmap), n_changed, len(df))

    # 빈 텍스트 제거
    before = len(df)
    df["text"] = df["text"].fillna("").astype(str)
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    if len(df) < before:
        log.info("빈 세그먼트 %d개 제거", before - len(df))

    out = interim / "normalized.parquet"
    df.to_parquet(out, index=False)
    log.info("normalized.parquet 작성: %d 세그먼트 (총 %d자) -> %s",
             len(df), int(df["text"].str.len().sum()), out)


if __name__ == "__main__":
    main()
