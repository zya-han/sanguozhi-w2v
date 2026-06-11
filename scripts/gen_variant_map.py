"""variant_map.tsv 생성 — 코퍼스별 異體字 통합 맵 (docs/ADDING_A_CORPUS.md §5).

방법(三國志 'Unify 異體字' 커밋 0cc6c51 방식 재현):
  1. segments.parquet 텍스트에 s2t(보호 글자 제외)만 적용 — variant_map 적용 전 상태.
  2. OpenCC JPVariants(번체→신자체) 역방향 쌍 + 보충 쌍(靣→面 등)을 후보로,
     코퍼스에 양쪽 모두 등장하는 쌍만 채택해 빈도 다수형으로 통합한다.
  3. 의미상이 글자(才≠纔·御≠禦·予≠豫·弁≠辯·余≠餘)와 opencc_protect 글자는 제외.

실행: CORPUS_CONFIG=config/<id>.yaml python scripts/gen_variant_map.py
출력: config의 normalize.variant_map 경로 (이후 Stage 2/3가 로드).
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from common import REPO_ROOT, get_logger, load_config, make_script_normalizer, resolve  # noqa: E402

log = get_logger("gen_variant_map")

# 의미가 다른 글자로의 오통합 방지 (신자체가 별개 고전 글자와 동형)
BLACKLIST = set("才御予弁余")
# JPVariants에 없는 보충 후보 (변이형, 표준형)
EXTRA_PAIRS = [("靣", "面"), ("曁", "暨")]


def main():
    cfg = load_config()
    nm = cfg.get("normalize", {})

    df = pd.read_parquet(resolve(cfg, "interim") / "segments.parquet")
    s2t = make_script_normalizer(nm.get("corpus_opencc"), nm.get("opencc_protect", ""))
    text = "".join(s2t(t) for t in df["text"].astype(str))
    freq = Counter(text)
    log.info("코퍼스 %d자 (s2t 적용, variant_map 미적용)", len(text))

    jp_path = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" \
        / "site-packages" / "opencc" / "dictionary" / "JPVariants.txt"
    pairs = list(EXTRA_PAIRS)
    for line in jp_path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and len(parts[0]) == 1 and len(parts[1]) == 1:
            pairs.append((parts[1], parts[0]))  # 역방향: 신자체 → 번체

    protect = set(nm.get("opencc_protect", "")) | BLACKLIST
    rows = []
    for a, b in pairs:
        if a in protect or b in protect or a == b:
            continue
        na, nb = freq.get(a, 0), freq.get(b, 0)
        if na == 0 or nb == 0:
            continue
        src, tgt = (a, b) if na <= nb else (b, a)  # 소수형 → 다수형
        rows.append((src, tgt, freq[src], freq[tgt]))

    rows.sort(key=lambda r: -r[2])
    out_path = nm["variant_map"]
    out = Path(out_path) if Path(out_path).is_absolute() else REPO_ROOT / out_path
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# 異體字 정규화 (변이형→표준형). s2t 후 적용. JPVariants(일본 신자체) 역방향 중",
             "# 코퍼스 다수형으로만 통합, 의미상이(才≠纔·御≠禦 등) 제외. "
             "scripts/gen_variant_map.py 생성."]
    for src, tgt, na, nb in rows:
        lines.append(f"{src}\t{tgt}\t# {na}→{nb}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("variant_map.tsv 작성: %d쌍 -> %s", len(rows), out)
    for src, tgt, na, nb in rows[:15]:
        log.info("  %s→%s (%d→%d)", src, tgt, na, nb)


if __name__ == "__main__":
    main()
