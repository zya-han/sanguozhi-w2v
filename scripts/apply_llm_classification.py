"""LLM 분류 결과를 stoplist/seed에 반영 (docs/GAZETTEER_REFINEMENT.md).

gen_llm_candidates.py가 만든 후보를 Workflow가 분류한 결과(JSON: {classified:[{surface,decision,type,note}]})
를 받아:
  - prior gaz:* 가 NONENTITY → stoplist 추가 (가제티어에서 제거)
  - prior fn:*  가 ENTITY    → seed_supplement 추가 (유형 포함)
필터(오분류 방어): ERA(간지 오류)·干支 2자·봉호꼬리(OFI인데 侯/王/君/公 끝)·type=NA 제외.
충돌: seed로 채택된 표면형이 기존 stoplist에 있으면 stoplist에서 제거(LLM 판정 우선).

실행: CORPUS_CONFIG=config/<id>.yaml python scripts/apply_llm_classification.py \
        --result <workflow_output.json> --candidates reports/<id>/llm_candidates.jsonl
워크플로 결과는 task 출력 파일(JSON)을 그대로 넘기면 된다(키 result.classified 자동 탐색).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from common import REPO_ROOT, get_logger, load_config, resolve  # noqa: E402

log = get_logger("apply_llm")

GAN = set("甲乙丙丁戊己庚辛壬癸")
ZHI = set("子丑寅卯辰巳午未申酉戌亥")


def find_classified(obj):
    """워크플로 출력 JSON에서 classified 배열을 탐색(중첩 result 허용)."""
    if isinstance(obj, dict):
        if isinstance(obj.get("classified"), list):
            return obj["classified"]
        for k in ("result", "results"):
            if k in obj:
                r = find_classified(obj[k])
                if r is not None:
                    return r
    return None


def accept_entity(s, t):
    """ENTITY를 seed로 받을지 + 필터. (받으면 True)"""
    if t in ("ERA", "NA", None):
        return False
    if len(s) == 2 and s[0] in GAN and s[1] in ZHI:   # 干支 날짜
        return False
    if t == "OFI" and s[-1] in "侯王君公":             # 봉호-꼬리(평侯·안侯…)
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True, help="Workflow 출력 JSON 경로")
    ap.add_argument("--candidates", required=True, help="gen_llm_candidates.py JSONL 경로")
    args = ap.parse_args()
    cfg = load_config()
    g = cfg["gazetteer"]

    cls = find_classified(json.load(open(args.result, encoding="utf-8")))
    if cls is None:
        log.error("결과에서 classified 배열을 못 찾음"); sys.exit(1)
    prior = {}
    for line in open(args.candidates, encoding="utf-8"):
        r = json.loads(line); prior[r["surface"]] = r["prior"]

    stop_add, seed_add, skipped = [], [], Counter()
    for c in cls:
        s, d, t = c["surface"], c["decision"], c.get("type", "NA")
        p = prior.get(s, "")
        if p.startswith("gaz:"):
            if d == "NONENTITY":
                stop_add.append(s)
        elif p.startswith("fn:"):
            if d == "ENTITY":
                if accept_entity(s, t):
                    seed_add.append((s, t))
                else:
                    skipped[t] += 1
    log.info("적용: stoplist FP +%d, seed FN +%d (유형 %s), 필터제외 %d",
             len(stop_add), len(seed_add), dict(Counter(t for _, t in seed_add)), sum(skipped.values()))

    seed_surf = {s for s, _ in seed_add}
    sl_path = Path(g["stoplist"]);  sl_path = sl_path if sl_path.is_absolute() else REPO_ROOT / sl_path
    cur = sl_path.read_text(encoding="utf-8").splitlines()
    cur_set = {l.strip() for l in cur if l.strip() and not l.startswith("#")}
    conflict = seed_surf & cur_set                    # LLM이 ENTITY로 본 기존 stoplist 항목
    kept = [l for l in cur if l.strip() not in conflict]
    new_fp = [s for s in dict.fromkeys(stop_add) if s not in cur_set]
    with open(sl_path, "w", encoding="utf-8") as f:
        f.write("\n".join(kept).rstrip("\n") + "\n")
        f.write("\n# ── LLM 전수분류 FP (비개체) ──\n")
        for s in new_fp:
            f.write(s + "\n")
    log.info("stoplist: 충돌 %d 제거 + 신규 FP %d", len(conflict), len(new_fp))

    seed_path = Path(g["seed_supplement"])
    seed_path = seed_path if seed_path.is_absolute() else REPO_ROOT / seed_path
    with open(seed_path, "a", encoding="utf-8") as f:
        f.write("\n# ── LLM 전수분류 FN 회복 ──\n")
        for s, t in sorted(seed_add, key=lambda x: (x[1], x[0])):
            f.write(f"{s}\t{t}\tLLM\n")
    log.info("seed_supplement: +%d 추가 → %s", len(seed_add), seed_path)


if __name__ == "__main__":
    main()
