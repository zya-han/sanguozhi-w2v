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
from common import get_logger, load_config, make_script_normalizer, resolve  # noqa: E402

log = get_logger("02_normalize")


def load_variant_map(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    p = Path(path)
    if not p.is_absolute():
        from common import REPO_ROOT
        p = REPO_ROOT / p
    mapping = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        src, dst = line.split("\t")[:2]
        mapping[src] = dst
    log.info("異體字 매핑 %d개 로드: %s", len(mapping), p)
    return mapping


def apply_variant_map(text: str, mapping: dict[str, str], counter: dict) -> str:
    if not mapping:
        return text
    out = []
    for ch in text:
        repl = mapping.get(ch)
        if repl is not None and repl != ch:
            counter[ch] = counter.get(ch, 0) + 1
            out.append(repl)
        else:
            out.append(ch)
    return "".join(out)


def main():
    cfg = load_config()
    norm = cfg.get("normalize", {})

    interim = resolve(cfg, "interim")
    df = pd.read_parquet(interim / "segments.parquet")

    if norm.get("corpus_opencc"):
        protect = norm.get("opencc_protect", "")
        normalize = make_script_normalizer(norm["corpus_opencc"], protect)
        before = df["text"].copy()
        df["text"] = df["text"].map(normalize)
        n_changed = int((before != df["text"]).sum())
        log.info("자형 정규화(%s, 보호 %d자): %d/%d 세그먼트 변경 — 혼합 자형 통일, 고전 의미 보존.",
                 norm["corpus_opencc"], len(set(protect)), n_changed, len(df))
    else:
        log.info("코퍼스 OpenCC 미적용 (正字 번체 보존).")

    vmap = load_variant_map(norm.get("variant_map"))
    if vmap:
        counter: dict[str, int] = {}
        df["text"] = df["text"].map(lambda t: apply_variant_map(t, vmap, counter))
        total = sum(counter.values())
        log.info("異體字 정규화 적용: 총 %d자 변경", total)
        for ch, n in sorted(counter.items(), key=lambda x: -x[1])[:15]:
            log.info("  %s→%s: %d", ch, vmap[ch], n)

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
