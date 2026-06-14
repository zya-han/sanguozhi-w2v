"""epochs 스윕 비교 (15/25/40) — config 와 동일 하이퍼파라미터, epochs 만 변경.

각 probe 의 최근접 이웃을 epoch 별로 나란히 출력하고,
top-10 이웃 집합의 Jaccard 안정성(15↔25, 25↔40)으로 수렴 여부를 판단한다.
include_peizhu 는 각 config 의 실제 값을 따른다(史記=제외, 三國志=포함).
"""
import json
import sys
from pathlib import Path

from gensim.models import Word2Vec

REPO = Path(__file__).resolve().parent.parent

# (corpus, include_peizhu, probes)  — config 실측값
JOBS = {
    "shiji": dict(
        include_peizhu=False,
        probes=["項羽", "高祖", "黃帝", "孔子", "丞相", "太守", "關中", "匈奴"],
    ),
    "sanguozhi": dict(
        include_peizhu=True,
        probes=["諸葛亮", "曹操", "劉備", "孫權", "丞相", "太守", "益州", "荆州"],
    ),
}
EPOCHS = [15, 25, 40]
# config 공통 (vector_size=100, sg=1, window=5, min_count=3, neg=10, seed=42, workers=1)
BASE = dict(sg=1, vector_size=100, window=5, min_count=3, negative=10, seed=42, workers=1)
TOPN = 8


def read_corpus(corpus, include_peizhu):
    sents = []
    p = REPO / "data" / corpus / "tokenized" / "corpus.jsonl"
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        if r["is_peizhu"] and not include_peizhu:
            continue
        sents.append(r["tokens"])
    toks = sum(len(s) for s in sents)
    return sents, toks


def jaccard(a, b):
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0


for corpus, spec in JOBS.items():
    sents, toks = read_corpus(corpus, spec["include_peizhu"])
    print(f"\n{'='*70}\n{corpus}  (학습토큰 {toks:,}, 注포함={spec['include_peizhu']})\n{'='*70}")
    models = {}
    for e in EPOCHS:
        m = Word2Vec(sentences=sents, epochs=e, **BASE)
        models[e] = m
        print(f"  trained epochs={e}  vocab={len(m.wv)}", flush=True)

    for probe in spec["probes"]:
        if probe not in models[EPOCHS[0]].wv:
            print(f"\n  [{probe}] OOV — 스킵")
            continue
        print(f"\n  ── {probe} ──")
        nbrs = {}
        for e in EPOCHS:
            sims = models[e].wv.most_similar(probe, topn=TOPN)
            nbrs[e] = [t for t, _ in sims]
            line = "  ".join(f"{t}·{s:.2f}" for t, s in sims)
            print(f"    e={e:<3} {line}")
        # 안정성: top-10 Jaccard
        top10 = {e: [t for t, _ in models[e].wv.most_similar(probe, topn=10)] for e in EPOCHS}
        j_15_25 = jaccard(top10[15], top10[25])
        j_25_40 = jaccard(top10[25], top10[40])
        print(f"    Jaccard@10  15↔25={j_15_25:.2f}  25↔40={j_25_40:.2f}")
