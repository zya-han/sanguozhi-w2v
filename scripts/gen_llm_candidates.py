"""LLM 전수분류용 가제티어 후보 생성 (docs/GAZETTEER_REFINEMENT.md).

漢書에서 검증한 '재설계 + LLM 전수 분류' 방법을 코퍼스 불문 재사용하기 위한 도구.
가제티어의 위양성(FP) 검토 대상과 위음성(FN) 후보를 KWIC와 함께 JSONL로 덤프한다.
이후 멀티에이전트 Workflow가 ENTITY/NONENTITY를 분류 → apply_llm_classification.py로 반영.

두 모드:
  main  : ① 현 가제티어 적중(auto, 비seed) freq>=3 = FP 검토 + ② entity-shape n-gram = FN 후보
  alt   : cbdb_altname=zi_only이 떨군 CBDB 別名(字/號) 중 미등재 빈출 = 누락 인물 別名 후보

스트림 범위: include_peizhu_in_training=false면 本文만, true면 注 포함(학습 대상과 일치).

실행: CORPUS_CONFIG=config/<id>.yaml python scripts/gen_llm_candidates.py --mode main
출력: reports/<id>/llm_candidates.jsonl (alt 모드는 llm_candidates2.jsonl)
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import regex as re

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from common import REPO_ROOT, ensure_dir, get_logger, load_config, resolve  # noqa: E402

log = get_logger("gen_llm_candidates")

PER_SUF = ("侯", "王", "君", "公")
OFI_SUF = ("將軍", "校尉", "都尉", "中郎將", "單于", "昆彌", "昆莫", "閼氏")
LOC_SUF = ("水", "陽", "陰", "城", "郡", "國", "縣", "陵", "關", "宮", "臺", "津", "丘", "鄉", "山")
# 경계어로 시작하면 봉호/지명 모양이어도 어구일 확률이 높다 (為王·立侯·之水…)
FUNC = set("為以從與遣拜封立大今願自時其諸是後前左右上下中將兵卒漢匈奴皆復更之故年者問召徙襲及不無有可使令遷置領行平定破")
NUM = set("一二三四五六七八九十百千萬")


def load_set(path):
    if not path:
        return set()
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.exists():
        return set()
    out = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            out.add(s.split("\t")[0].strip())
    return out


def get_stream_text(cfg) -> tuple[str, list[str]]:
    """학습 대상 스트림(注 포함 여부는 include_peizhu)에 해당하는 본문 결합 + 세그먼트 리스트."""
    norm = pd.read_parquet(resolve(cfg, "interim") / "normalized.parquet")
    if not cfg["corpus"].get("include_peizhu_in_training", True):
        norm = norm[~norm["is_peizhu"]]
    segs = norm["text"].astype(str).tolist()
    return "\n".join(segs), segs


def kwic(text, s, n=2, w=14):
    out, start = [], 0
    while len(out) < n:
        i = text.find(s, start)
        if i == -1:
            break
        out.append((text[max(0, i - w):i] + "〖" + s + "〗"
                    + text[i + len(s):i + len(s) + w]).replace("\n", "·"))
        start = i + len(s)
    return out


def gen_main(cfg, text, segs):
    g = cfg["gazetteer"]
    gaz = pd.read_csv(resolve(cfg, "gazetteer") / "gazetteer.tsv", sep="\t")
    gset = set(gaz["surface"].astype(str))
    typ = dict(zip(gaz["surface"].astype(str), gaz["type"]))
    seed = load_set(g.get("seed_supplement")) | load_set(g.get("user_dict"))
    db = sorted((resolve(cfg, "vendor") / "cbdb_sqlite").glob("*.sqlite3"))[0]
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    addr = {r[0] for r in con.execute("SELECT c_name_chn FROM ADDR_CODES WHERE c_name_chn IS NOT NULL")}
    con.close()
    # A) FP 검토: 현 가제티어 적중(비seed) freq>=3
    A = [(s, text.count(s), f"gaz:{typ[s]}") for s in gset if s not in seed and text.count(s) >= 3]
    # B) FN 후보: entity-shape n-gram (ADDR 멤버십 또는 어미 모양)
    cnt = Counter()
    for seg in segs:
        for run in re.findall(r"\p{Han}+", seg):
            for L in (2, 3, 4):
                for i in range(len(run) - L + 1):
                    cnt[run[i:i + L]] += 1
    B, seen = [], set()
    for s, n in cnt.items():
        if s in gset or s in seed or s in seen or not (2 <= len(s) <= 4):
            continue
        pref_ok = s[0] not in FUNC
        if s in addr and n >= 4:
            B.append((s, n, "fn:ADDR")); seen.add(s); continue
        if any(s.endswith(x) for x in OFI_SUF) and n >= 4 and pref_ok:
            B.append((s, n, "fn:OFIshape")); seen.add(s); continue
        if any(s.endswith(x) for x in PER_SUF) and n >= 5 and pref_ok and len(s) <= 3:
            B.append((s, n, "fn:PERshape")); seen.add(s); continue
        if any(s.endswith(x) for x in LOC_SUF) and n >= 8 and pref_ok:
            B.append((s, n, "fn:LOCshape")); seen.add(s); continue
    log.info("main: FP검토(gaz-auto) %d + FN후보(shape/ADDR) %d", len(A), len(B))
    return sorted(A, key=lambda x: -x[1]) + sorted(B, key=lambda x: -x[1])


def gen_alt(cfg, text, segs):
    g = cfg["gazetteer"]
    gaz = pd.read_csv(resolve(cfg, "gazetteer") / "gazetteer.tsv", sep="\t")
    gset = set(gaz["surface"].astype(str))
    stop = load_set(g.get("stoplist"))
    seed = load_set(g.get("seed_supplement")) | load_set(g.get("user_dict"))
    db = sorted((resolve(cfg, "vendor") / "cbdb_sqlite").glob("*.sqlite3"))[0]
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    alt = {r[0] for r in con.execute(
        "SELECT c_alt_name_chn FROM ALTNAME_DATA "
        "WHERE c_alt_name_chn IS NOT NULL AND length(c_alt_name_chn) BETWEEN 2 AND 3")}
    con.close()
    out = []
    for s in alt:
        if s in gset or s in stop or s in seed:
            continue
        if all(c in NUM for c in s):       # 순수 숫자 제외
            continue
        n = text.count(s)
        if n >= 6:
            out.append((s, n, "fn:ALT"))
    log.info("alt: 누락 別名 후보 %d (freq>=6)", len(out))
    return sorted(out, key=lambda x: -x[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["main", "alt"], default="main")
    args = ap.parse_args()
    cfg = load_config()
    text, segs = get_stream_text(cfg)
    rows = gen_main(cfg, text, segs) if args.mode == "main" else gen_alt(cfg, text, segs)
    rep = ensure_dir(resolve(cfg, "reports"))
    fn = "llm_candidates.jsonl" if args.mode == "main" else "llm_candidates2.jsonl"
    out = rep / fn
    with open(out, "w", encoding="utf-8") as f:
        for i, (s, n, pri) in enumerate(rows):
            f.write(json.dumps({"idx": i, "surface": s, "freq": n, "prior": pri,
                                "kwic": kwic(text, s)}, ensure_ascii=False) + "\n")
    import math
    log.info("덤프 %d종 → %s", len(rows), out)
    log.info("Workflow: TOTAL=%d, BATCH=60 → 에이전트 %d개", len(rows), math.ceil(len(rows) / 60))


if __name__ == "__main__":
    main()
