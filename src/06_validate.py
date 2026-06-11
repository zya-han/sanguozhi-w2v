"""Stage 6 — 검증 / 산출 점검 (명세 §6).

1. 토큰화 정합성: 諸葛亮·孔明·曹操·丞相 등이 단일 토큰인지, 단자 名(亮·操)이 병합되지 않았는지 자동 검사.
2. 커버리지 리포트: 가제티어 적중 수, 개체 토큰 빈도 분포, OOV/희소 개체.
출력: reports/validation.md
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ensure_dir, get_logger, load_config, resolve  # noqa: E402

log = get_logger("06_validate")


def main():
    cfg = load_config()
    from gensim.models import Word2Vec

    # 로드
    gaz = pd.read_csv(resolve(cfg, "gazetteer") / "gazetteer.tsv", sep="\t")
    gaz_set = set(gaz["surface"].astype(str))
    recs = [json.loads(l) for l in open(resolve(cfg, "tokenized") / "corpus.jsonl", encoding="utf-8")]
    all_tokens = [t for r in recs for t in r["tokens"]]
    tok_freq = Counter(all_tokens)
    model = Word2Vec.load(str(resolve(cfg, "models")
                              / cfg["word2vec"].get("model_name", "w2v_sanguozhi.model")))

    # ── 1. 토큰화 정합성 검사 (센티넬은 사서별 config `validate`) ──
    vcfg = cfg.get("validate", {})
    must_single = vcfg.get("must_single",
                           ["諸葛亮", "孔明", "曹操", "劉備", "孫權", "周瑜", "呂蒙",
                            "丞相", "太守", "益州"])
    must_char = vcfg.get("must_char", ["亮", "操", "備", "羽"])  # 단자 名 — 글자 토큰으로 남아야
    alias_a, alias_b = vcfg.get("alias_pair", ["諸葛亮", "孔明"])
    title_probe = vcfg.get("title_probe", "《魏略》")
    checks = []
    for w in must_single:
        present = tok_freq.get(w, 0) > 0
        in_gaz = w in gaz_set
        ok = present and in_gaz
        checks.append((f"개체 '{w}' 단일 토큰", ok, f"빈도 {tok_freq.get(w,0)}, 가제티어 {'O' if in_gaz else 'X'}"))
    for w in must_char:
        # 단자가 토큰으로 존재하고, 가제티어에 단자로 포함되지 않아야
        ok = (tok_freq.get(w, 0) > 0) and (w not in gaz_set)
        checks.append((f"단자 名 '{w}' 미병합", ok, f"글자토큰 빈도 {tok_freq.get(w,0)}, 가제티어 포함 {'O(문제!)' if w in gaz_set else 'X'}"))
    # 별명 비정규화: 名과 字가 서로 다른 토큰으로 공존
    ok_alias = tok_freq.get(alias_a, 0) > 0 and tok_freq.get(alias_b, 0) > 0
    checks.append((f"별명 비정규화({alias_a}·{alias_b} 독립 공존)", ok_alias,
                   f"{alias_a} {tok_freq.get(alias_a,0)}, {alias_b} {tok_freq.get(alias_b,0)}"))
    # 《書名》 = 괄호 포함 단일 토큰
    n_titles = sum(1 for t in tok_freq if t.startswith("《") and t.endswith("》"))
    ok_title = tok_freq.get(title_probe, 0) > 0 and n_titles > 0
    checks.append(("《書名》 괄호포함 단일 토큰", ok_title,
                   f"{title_probe} {tok_freq.get(title_probe,0)}, 표제 토큰 {n_titles}종"))
    # 마크업 누출 가드: ASCII 영숫자 토큰 0 (colspan·<p>·Category 등 잔재 검출)
    import re as _re
    ascii_toks = sorted((t for t in tok_freq if _re.search(r"[A-Za-z0-9]", t)),
                        key=lambda t: -tok_freq[t])
    checks.append(("마크업 누출 없음(ASCII 토큰 0종)", not ascii_toks,
                   f"발견 {len(ascii_toks)}종" + (f": {' '.join(ascii_toks[:10])}" if ascii_toks else "")))
    all_pass = all(c[1] for c in checks)

    # ── 2. 커버리지 ──
    entity_types = {len(t): 0 for t in gaz_set}
    gaz_hit = [s for s in gaz_set if tok_freq.get(s, 0) > 0]
    n_entity_tok_occ = sum(tok_freq.get(s, 0) for s in gaz_set)
    total_tok = len(all_tokens)
    by_type = {}
    for _, row in gaz.iterrows():
        if tok_freq.get(row["surface"], 0) > 0:
            by_type.setdefault(row["type"], 0)
            by_type[row["type"]] += 1
    # 빈도 상위 개체
    top_entities = [(s, tok_freq[s]) for s in gaz_set if tok_freq.get(s, 0) > 0]
    top_entities.sort(key=lambda x: -x[1])
    # 희소 개체 (가제티어엔 있으나 코퍼스 빈도 1~2 = min_count 미만으로 OOV 위험)
    rare = [(s, tok_freq.get(s, 0)) for s in gaz_set if 0 < tok_freq.get(s, 0) < int(cfg["word2vec"]["min_count"])]
    in_vocab = sum(1 for s in gaz_hit if s in model.wv)

    # ── most_similar 예시 ──
    sim_examples = {}
    for probe in vcfg.get("sim_probes",
                          ["諸葛亮", "曹操", "劉備", "孫權", "丞相", "太守", "益州", "荆州"]):
        if probe in model.wv:
            sim_examples[probe] = model.wv.most_similar(probe, topn=8)

    # ── 文/注 분포 ──
    main_tok = sum(len(r["tokens"]) for r in recs if not r["is_peizhu"])
    pei_tok = sum(len(r["tokens"]) for r in recs if r["is_peizhu"])

    # ── 리포트 작성 ──
    book = cfg["corpus"].get("book_title", "三國志")
    L = []
    L.append(f"# 검증 리포트 — 《{book}》 개체 인식 토큰화 + Word2Vec\n")
    L.append(f"- 코퍼스 토큰: **{total_tok:,}** (本文 {main_tok:,} / 注 {pei_tok:,})")
    L.append(f"- 어휘(vocab, min_count={cfg['word2vec']['min_count']}): **{len(model.wv):,}**")
    L.append(f"- 가제티어 표면형: {len(gaz_set):,} → 코퍼스 적중: **{len(gaz_hit):,}** "
             f"(개체 토큰 출현 {n_entity_tok_occ:,}회, 전체의 {100*n_entity_tok_occ/total_tok:.1f}%)\n")

    L.append("## 1. 토큰화 정합성 자동 검사")
    L.append(f"\n**결과: {'✅ 전부 통과' if all_pass else '❌ 일부 실패'}**\n")
    L.append("| 검사 | 통과 | 상세 |")
    L.append("|---|:--:|---|")
    for name, ok, detail in checks:
        L.append(f"| {name} | {'✅' if ok else '❌'} | {detail} |")

    L.append("\n## 2. 커버리지")
    L.append("\n### 유형별 적중 표면형")
    L.append("| 유형 | 적중 수 |")
    L.append("|---|--:|")
    for t, n in sorted(by_type.items(), key=lambda x: -x[1]):
        L.append(f"| {t} | {n} |")
    L.append(f"\n- 적중 개체 중 vocab 진입(min_count 충족): **{in_vocab}** / {len(gaz_hit)}")
    L.append(f"- 희소 개체(빈도 1~{int(cfg['word2vec']['min_count'])-1}, OOV 위험): **{len(rare)}**개")

    L.append("\n### 빈도 상위 개체 토큰 (상위 25)")
    L.append("| 개체 | 빈도 |")
    L.append("|---|--:|")
    for s, n in top_entities[:25]:
        L.append(f"| {s} | {n} |")

    L.append("\n## 3. 분포 의미 점검 — `most_similar` 예시")
    for probe, sims in sim_examples.items():
        items = ", ".join(f"{t}({s:.2f})" for t, s in sims)
        L.append(f"- **{probe}**: {items}")

    L.append("\n## 4. 명시된 한계 (명세 §2.4, §6)")
    L.append(f"- **단자(單字) 名 미병합**: {'·'.join(must_char)} 등은 동형이의라 WSD 없이 안전 분리 불가 → 글자 토큰 유지.")
    L.append("- **개체 희소성**: CBDB의 해당 시대 인물 커버리지가 제한적이고 본문 언급이 적은 개체는 빈도가 낮아 "
             "임베딩이 불안정할 수 있음. 위 희소 개체 수 참조.")
    L.append(f"- **별명 비정규화(의도적)**: {alias_a}(名)·{alias_b}(字) 등 名·字·諡는 각각 독립 토큰 — 서로 다른 화용 분포 관측이 연구 목적.")
    L.append("- **가제티어 정밀도**: CBDB OFFICE/ADDR에서 유입된 비개체어는 stoplist로 제거했으나 일부 저빈도 잔재 가능.")
    L.append("- **자형**: 코퍼스(四庫 WYG)는 正字 번체 그대로 보존(OpenCC 미적용) — 于/於·云/雲 등 고전 의미 구분 유지.")

    out = ensure_dir(resolve(cfg, "reports")) / "validation.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    log.info("validation.md 작성: %s", out)
    log.info("토큰화 정합성: %s (%d/%d 통과)",
             "전부 통과" if all_pass else "일부 실패",
             sum(1 for c in checks if c[1]), len(checks))
    if not all_pass:
        for name, ok, detail in checks:
            if not ok:
                log.warning("  실패: %s — %s", name, detail)


if __name__ == "__main__":
    main()
