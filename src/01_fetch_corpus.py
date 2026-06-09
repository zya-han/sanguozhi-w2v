"""Stage 1 — 원문 수집 및 本文/裴注 분리 (中文維基文庫 zh.wikisource).

정본: zh.wikisource 三國志 — 전체 65卷(魏30·蜀15·吳20), 正字 번체, 표점 포함.
  (당초 Kanripo KR2a0012는 魏志 30卷만 담겨 蜀/吳書가 누락 → 周瑜 등 다수 개체 부재.
   wikisource는 전체 65卷을 번체로 제공하므로 교체.)

위키텍스트 마크업:
  - `{{*|...}}`  = 裴松之 注 (인라인 雙行夾注에 해당)  → 注 스트림(is_peizhu=True)
  - 그 밖의 텍스트 = 本文(陳壽)                       → 本文 스트림
  - `==전기명==`  = 傳 단위 섹션 헤더 (드롭)
  - `[[링크|표시]]`→표시, `《書名》` 유지, `'''굵게'''` 제거, 기타 {{템플릿}} 제거
표점(。！？)으로 문장 분할 → 각 문장이 윈도 단위 segment (명세 §2.6).
토큰화 대상은 한자(漢字)만; 구두점은 문장 경계로만 쓰고 토큰에서 제외(Stage 4).

출력: data/raw/wikisource/卷NN.wiki (캐시), data/raw/provenance.json,
      data/interim/segments.parquet (segment_id, text, source, juan, shu, kind, is_peizhu).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import regex as re
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ensure_dir, get_logger, load_config, resolve  # noqa: E402

log = get_logger("01_fetch")

# 한자 + 《》 만 보존(《書名》을 토큰화 단계에서 통째로 한 토큰으로 방출하기 위함).
# 나머지 구두점은 문장 경계로만 쓰고 제거한다.
KEEP = re.compile(r"[\p{Han}《》]+")
HAN_ANY = re.compile(r"\p{Han}")
SECTION_RE = re.compile(r"^=+\s*[^=]+?\s*=+\s*$", re.MULTILINE)


def fetch_juan_wikitext(cfg: dict, juan: int) -> str:
    """卷 위키텍스트를 API로 받되, data/raw/wikisource/ 에 캐시."""
    cache_dir = ensure_dir(resolve(cfg, "raw") / "wikisource")
    pad = int(cfg["corpus"]["juan_zero_pad"])
    cache = cache_dir / f"卷{juan:0{pad}d}.wiki"
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    page = f"{cfg['corpus']['page_prefix']}{juan:0{pad}d}"
    headers = {"User-Agent": "sanguozhi-w2v-research/1.0 (academic; contact via github)"}
    params = {"action": "parse", "page": page, "prop": "wikitext",
              "format": "json", "maxlag": 5}
    for attempt in range(6):
        r = requests.get(cfg["corpus"]["wikisource_api"], params=params,
                         headers=headers, timeout=30)
        if r.status_code == 429 or (r.status_code == 200 and "maxlag" in r.text[:200]):
            wait = int(r.headers.get("Retry-After", 2 ** attempt + 2))
            log.warning("卷%d rate-limit(%s), %ds 대기 후 재시도", juan, r.status_code, wait)
            time.sleep(wait)
            continue
        r.raise_for_status()
        wt = r.json().get("parse", {}).get("wikitext", {}).get("*", "")
        cache.write_text(wt, encoding="utf-8")
        time.sleep(1.0)  # 예의상 rate limit
        return wt
    raise RuntimeError(f"卷{juan} 반복 rate-limit으로 실패")


def split_main_note(wikitext: str) -> list[tuple[str, str]]:
    """{{...}} 균형 파싱으로 本文/裴注 분리.

    `{{*|...}}` → ('peizhu', 내용). 그 외 템플릿 → 드롭. 나머지 → ('main', ...).
    인라인 注가 本文을 끊으므로, 호출부에서 main 조각을 이어붙여 문장 분할한다.
    """
    out: list[tuple[str, str]] = []
    buf: list[str] = []
    i, n = 0, len(wikitext)
    while i < n:
        if wikitext[i:i + 2] == "{{":
            # 매칭되는 }} 까지 (중첩 허용)
            depth, j = 0, i
            while j < n:
                if wikitext[j:j + 2] == "{{":
                    depth += 1; j += 2
                elif wikitext[j:j + 2] == "}}":
                    depth -= 1; j += 2
                    if depth == 0:
                        break
                else:
                    j += 1
            inner = wikitext[i + 2:j - 2]
            if inner.startswith("*|"):           # 裴注
                out.append(("main", "".join(buf))); buf = []
                out.append(("peizhu", inner[2:]))
            # 그 외 템플릿(헤더/라이선스 등) → 드롭
            i = j
        else:
            buf.append(wikitext[i]); i += 1
    out.append(("main", "".join(buf)))
    return out


def clean_wiki(text: str) -> str:
    """위키 마크업 제거 → 표점 포함 평문."""
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"</?(?:onlyinclude|noinclude|includeonly|ref|small|sub|sup)[^>]*>", "", text)
    text = re.sub(r"__\w+__", "", text)                       # __FORCETOC__ 등
    text = re.sub(r"\[\[(?:[^\[\]|]*\|)?([^\[\]|]+)\]\]", r"\1", text)  # [[a|b]]→b, [[a]]→a
    text = re.sub(r"\[\[[^\]]*\]\]", "", text)                 # 잔여 링크([[Category:..]])
    text = re.sub(r"'{2,}", "", text)                         # ''' '' 강조
    text = re.sub(r"\{\{[^{}]*\}\}", "", text)                 # 잔여 단순 템플릿
    text = SECTION_RE.sub("\n", text)                         # ==전기명== 헤더 드롭
    text = text.replace("　", "").replace("​", "")
    return text


def to_sentences(cleaned: str, delims: str) -> list[str]:
    """(이미 clean_wiki된) 텍스트를 표점으로 문장 분할. 한자 + 《》만 보존. 빈 문장 제외."""
    pattern = "[" + re.escape(delims) + "]"
    sents = []
    for chunk in re.split(pattern, cleaned):
        kept = "".join(KEEP.findall(chunk))
        if HAN_ANY.search(kept):
            sents.append(kept)
    return sents


def shu_of(juan: int) -> str:
    return "魏書" if juan <= 30 else "蜀書" if juan <= 45 else "吳書"


def build_segments(cfg: dict) -> pd.DataFrame:
    delims = cfg["corpus"]["sentence_delims"]
    n_juan = int(cfg["corpus"]["juan_count"])
    rows = []
    for juan in range(1, n_juan + 1):
        wt = fetch_juan_wikitext(cfg, juan)
        if not wt.strip():
            log.warning("卷%d 위키텍스트 비어있음", juan)
            continue
        parts = split_main_note(wt)
        main_clean = clean_wiki("".join(t for k, t in parts if k == "main"))
        note_cleans = [clean_wiki(t) for k, t in parts if k == "peizhu"]

        jid = f"卷{juan:02d}"
        shu = shu_of(juan)
        for si, sent in enumerate(to_sentences(main_clean, delims)):
            rows.append(dict(segment_id=f"sgz_{jid}_m{si:04d}", text=sent,
                             source="zh.wikisource:三國志", juan=jid, shu=shu,
                             kind="main", is_peizhu=False))
        ni = 0
        for nc in note_cleans:
            for sent in to_sentences(nc, delims):
                rows.append(dict(segment_id=f"sgz_{jid}_n{ni:05d}", text=sent,
                                 source="zh.wikisource:三國志", juan=jid, shu=shu,
                                 kind="peizhu", is_peizhu=True))
                ni += 1
    return pd.DataFrame(rows)


def main():
    cfg = load_config()
    df = build_segments(cfg)
    # 청대 考證·front matter는 wikisource 본문에 포함되지 않으므로 별도 제외 불필요.

    interim = ensure_dir(resolve(cfg, "interim"))
    df.to_parquet(interim / "segments.parquet", index=False)

    raw_dir = ensure_dir(resolve(cfg, "raw"))
    prov = {
        "corpus": "三國志 (正史) — Chen Shou 撰, Pei Songzhi 注 (全 65卷)",
        "source": "中文維基文庫 (zh.wikisource.org) 三國志",
        "api": cfg["corpus"]["wikisource_api"],
        "juan": f"卷01–卷{cfg['corpus']['juan_count']:02d} (魏書1-30, 蜀書31-45, 吳書46-65)",
        "license": "CC BY-SA 4.0 (Wikisource)",
        "note": "{{*|...}}=裴松之 注, 그 외=陳壽 本文. 표점(。！？)으로 문장 분할, 한자만 토큰화.",
    }
    (raw_dir / "provenance.json").write_text(
        json.dumps(prov, ensure_ascii=False, indent=2), encoding="utf-8")

    by = df.groupby(["shu", "kind"]).agg(
        n=("text", "size"), chars=("text", lambda s: s.str.len().sum()))
    log.info("segments.parquet: %d 문장 세그먼트", len(df))
    for (shu, kind), r in by.iterrows():
        log.info("  %s/%-6s seg=%5d chars=%d", shu, kind, int(r.n), int(r.chars))


if __name__ == "__main__":
    main()
