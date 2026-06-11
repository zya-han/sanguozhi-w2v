"""토큰화 정밀 감사 — 위양성(가제티어 오병합)·위음성(누락 개체) 검토 자료 생성.

위양성: 코퍼스 적중 가제티어 표면형 전수를 빈도·유형·출처(시드/CBDB계)·KWIC과 함께 덤프.
위음성: 결정론 패턴으로 후보 채굴(전기 도입 'X者，'·'帝X'·'X氏'·'姓X'·'名曰X'·'號曰X'),
        가제티어 미수록분만 빈도·KWIC과 함께 덤프. 채택은 수동 큐레이션(시드 반영).

실행: CORPUS_CONFIG=config/<id>.yaml python scripts/audit_tokens.py
출력: reports/<id>/audit_fp_review.tsv, reports/<id>/audit_fn_candidates.tsv
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import regex as re

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from common import REPO_ROOT, ensure_dir, get_logger, load_config, resolve  # noqa: E402

log = get_logger("audit_tokens")

KWIC_N = 3
KWIC_W = 12


def load_surfaces(path):
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.exists():
        return set()
    out = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            out.add(line.split("\t")[0].strip())
    return out


def kwic(text: str, term: str, n: int = KWIC_N, w: int = KWIC_W) -> list[str]:
    out, start = [], 0
    while len(out) < n:
        i = text.find(term, start)
        if i == -1:
            break
        ctx = text[max(0, i - w):i] + "▸" + term + "◂" + text[i + len(term):i + len(term) + w]
        out.append(ctx.replace("\n", "·"))
        start = i + len(term)
    return out


def main():
    cfg = load_config()
    g = cfg["gazetteer"]
    rep = ensure_dir(resolve(cfg, "reports"))

    norm = pd.read_parquet(resolve(cfg, "interim") / "normalized.parquet")
    text = "\n".join(norm["text"])

    gaz = pd.read_csv(resolve(cfg, "gazetteer") / "gazetteer.tsv", sep="\t")
    seed = load_surfaces(g.get("seed_supplement")) | load_surfaces(g.get("user_dict"))

    # ── 위양성 검토: 적중 표면형 전수 + KWIC ──
    rows = []
    for _, r in gaz.iterrows():
        s = str(r["surface"])
        n = text.count(s)
        if n == 0:
            continue
        rows.append((n, s, r["type"], "seed" if s in seed else "auto", " ｜ ".join(kwic(text, s))))
    rows.sort(key=lambda x: -x[0])
    fp = rep / "audit_fp_review.tsv"
    with open(fp, "w", encoding="utf-8") as f:
        f.write("freq\tsurface\ttype\torigin\tkwic\n")
        for n, s, t, o, k in rows:
            f.write(f"{n}\t{s}\t{t}\t{o}\t{k}\n")
    log.info("위양성 검토 목록: %d 표면형 -> %s (auto %d / seed %d)",
             len(rows), fp, sum(1 for r in rows if r[3] == "auto"),
             sum(1 for r in rows if r[3] == "seed"))

    # ── 위음성 후보: 결정론 패턴 채굴 ──
    gset = set(gaz["surface"].astype(str))
    pats = {
        "者도입": re.compile(r"(?:^|\n)(\p{Han}{2,4})者，"),
        "帝X":   re.compile(r"帝(\p{Han}{2})"),
        "X氏":   re.compile(r"(\p{Han}{2,3}氏)"),
        "姓X":   re.compile(r"姓(\p{Han}{2})"),
        "名曰X": re.compile(r"名曰(\p{Han}{2})"),
        "號曰X": re.compile(r"號曰?(\p{Han}{2,4})"),
    }
    cand = Counter()
    src = {}
    for name, pat in pats.items():
        for m in pat.finditer(text):
            c = m.group(1)
            if c in gset:
                continue
            cand[c] += 1
            src.setdefault(c, set()).add(name)
    fn = rep / "audit_fn_candidates.tsv"
    with open(fn, "w", encoding="utf-8") as f:
        f.write("freq_pat\tfreq_text\tcand\tpatterns\tkwic\n")
        for c, n in cand.most_common():
            f.write(f"{n}\t{text.count(c)}\t{c}\t{','.join(sorted(src[c]))}\t{' ｜ '.join(kwic(text, c))}\n")
    log.info("위음성 후보: %d개 -> %s", len(cand), fn)


if __name__ == "__main__":
    main()
