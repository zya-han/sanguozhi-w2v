"""Stage 3 — 가제티어 구축 (PER / OFI / LOC).

소스: CBDB SQLite (vendor/cbdb_sqlite/*.sqlite3). CBDB는 正體(번체) 보관이라 코퍼스와 동일 자형.
  - PER: BIOG_MAIN.c_name_chn (본명) + ALTNAME_DATA.c_alt_name_chn (字·號·諡 등 별명)
         후漢~三國 인물(c_dy ∈ dynasty_codes) 한정. 별명·본명은 각각 독립 행(병합 금지, 명세 §2.3).
  - OFI: OFFICE_CODES.c_office_chn (+ _alt). 관직은 시대횡단이므로 전 왕조 채택 후 코퍼스 필터.
  - LOC: ADDR_CODES.c_name_chn (=CHGIS 연계). CHGIS 연도는 후대 행정인스턴스라 신뢰 불가 →
         코퍼스 출현 필터로 三國 스코핑.
  + 보충: seed_supplement.tsv (CBDB 누락 canonical 字/號, 검증 §6 충족).

후처리(명세 §3.4):
  - CBDB 이름의 괄호 주기 제거 (例: 夏侯氏(曹文叔妻) → 夏侯氏), 공백/괄호 제거.
  - 2자 미만 제거 (단자 名 병합 금지).
  - (opencc_config 지정 시) 코퍼스 자형 정규화. 기본 null(CBDB 이미 번체).
  - 중복 제거.
  - require_corpus_occurrence: 코퍼스에 실제 등장하는 표면형만 유지(三國 스코핑·노이즈 제거).
  - 길이 내림차순 정렬(최장일치용).
출력: data/gazetteer/gazetteer.tsv (surface, type), gazetteer_stats.json.
"""
from __future__ import annotations

import json
import regex as re
import sqlite3
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (REPO_ROOT, ensure_dir, get_logger, load_config,  # noqa: E402
                    load_variant_map, make_script_normalizer, resolve)

log = get_logger("03_gazetteer")

# CBDB 이름 내 괄호 주기/공백 제거용
PAREN_RE = re.compile(r"[（(〔【［\[].*?[）)〕】］\]]")
CLEAN_RE = re.compile(r"[\s　・·,，、。;；:：!！?？\"'’”“]+")


def find_cbdb(cfg: dict) -> Path:
    vendor = resolve(cfg, "vendor")
    cands = sorted((vendor / "cbdb_sqlite").glob("*.sqlite3")) + \
        sorted(vendor.glob("**/*.sqlite3"))
    if not cands:
        raise FileNotFoundError(
            "CBDB sqlite를 찾을 수 없음. vendor/cbdb_sqlite/ 에 *.sqlite3 필요 "
            "(HuggingFace latest.zip 에서 받아 압축 해제).")
    log.info("CBDB sqlite: %s", cands[0])
    return cands[0]


def clean_surface(s: str | None) -> str | None:
    if not s:
        return None
    s = PAREN_RE.sub("", s)
    s = CLEAN_RE.sub("", s)
    s = s.strip()
    return s or None


def extract_per(con: sqlite3.Connection, dy_codes: list[int],
                y0: int, y1: int) -> list[tuple[str, str]]:
    dy = ",".join(str(int(c)) for c in dy_codes)
    cur = con.cursor()
    # 본명: 시대코드 ∈ dy_codes 이거나 생몰/활동 연도가 [y0,y1]와 중첩
    cur.execute(f"""
        SELECT c_personid, c_name_chn FROM BIOG_MAIN
        WHERE c_name_chn IS NOT NULL AND c_name_chn <> ''
          AND (
            c_dy IN ({dy})
            OR (COALESCE(c_birthyear,c_index_year,c_fl_earliest_year) <= ?
                AND COALESCE(c_deathyear,c_fl_latest_year,c_index_year) >= ?)
          )
    """, (y1, y0))
    rows = cur.fetchall()
    pids = {r[0] for r in rows}
    out = [(r[1], "PER") for r in rows]
    # 별명: 위 인물의 字·號·諡 등 (각각 독립 표면형, 병합 금지)
    if pids:
        qm = ",".join("?" * len(pids))
        cur.execute(f"""
            SELECT c_alt_name_chn FROM ALTNAME_DATA
            WHERE c_alt_name_chn IS NOT NULL AND c_alt_name_chn <> ''
              AND c_personid IN ({qm})
        """, tuple(pids))
        out += [(r[0], "PER") for r in cur.fetchall()]
    log.info("PER: 인물 %d명, 표면형(본명+별명) %d개(정제 전)", len(pids), len(out))
    return out


# 전기 도입부 인명: 한자만(부호 불포함) 2~4자 + 字 + 한자. 문장 내 어디서나.
ZI_RE = re.compile(r"(\p{Han}{2,4}?)字\p{Han}")


def common_surnames(con: sqlite3.Connection, threshold: int) -> tuple[set[str], set[str]]:
    """CBDB 빈출 姓(threshold 인원 이상). 노이즈 글자(曰·子 등) 배제용."""
    cur = con.execute(
        "SELECT c_surname_chn, COUNT(*) n FROM BIOG_MAIN "
        "WHERE c_surname_chn IS NOT NULL AND c_surname_chn<>'' "
        "GROUP BY c_surname_chn HAVING n>=?", (threshold,))
    sur = {r[0] for r in cur.fetchall()}
    return {s for s in sur if len(s) == 1}, {s for s in sur if len(s) == 2}


def extract_per_from_zi(con: sqlite3.Connection, cfg: dict, threshold: int) -> list[tuple[str, str]]:
    """코퍼스 전기 도입부 '{姓名}字{字}'에서 인명 추출(문장 시작 앵커 + 빈출 姓).

    CBDB가 빠뜨린 三國 인물(周瑜·呂蒙 등)을 코퍼스 자체에서 결정론적으로 확보.
    """
    sur1, sur2 = common_surnames(con, threshold)
    norm = pd.read_parquet(resolve(cfg, "interim") / "normalized.parquet")
    bad2 = set("弟子兄父母妻女姊妹")  # 3자 후보의 2번째 글자가 이러면 친족어 → 배제
    names = set()
    for t in norm["text"].astype(str):
        for m in ZI_RE.finditer(t):
            nm = m.group(1)
            if len(nm) >= 3 and nm[:2] in sur2:
                names.add(nm[:3])
            elif nm[0] in sur1 and len(nm) <= 3:
                if len(nm) == 3 and nm[1] in bad2:
                    continue
                names.add(nm)
    log.info("코퍼스 '字' 도입부 인명 추출: %d개 (빈출 姓 %d/%d)",
             len(names), len(sur1), len(sur2))
    return [(n, "PER") for n in names]


def extract_ofi(con: sqlite3.Connection) -> list[tuple[str, str]]:
    cur = con.cursor()
    out = []
    for col in ("c_office_chn", "c_office_chn_alt"):
        cur.execute(f"SELECT {col} FROM OFFICE_CODES WHERE {col} IS NOT NULL AND {col} <> ''")
        out += [(r[0], "OFI") for r in cur.fetchall()]
    log.info("OFI: 관직 표면형 %d개(정제 전)", len(out))
    return out


def extract_loc(con: sqlite3.Connection) -> list[tuple[str, str]]:
    cur = con.cursor()
    cur.execute("SELECT c_name_chn FROM ADDR_CODES WHERE c_name_chn IS NOT NULL AND c_name_chn <> ''")
    out = [(r[0], "LOC") for r in cur.fetchall()]
    log.info("LOC: 지명 표면형 %d개(정제 전)", len(out))
    return out


def load_stoplist(path: str | None) -> set[str]:
    if not path:
        return set()
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.exists():
        log.warning("stoplist 없음: %s", p)
        return set()
    words = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            words.add(line)
    log.info("stoplist %d개 로드", len(words))
    return words


def load_seed(path: str | None) -> list[tuple[str, str]]:
    if not path:
        return []
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.exists():
        log.warning("seed_supplement 없음: %s", p)
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 2 and parts[0].strip():
            out.append((parts[0].strip(), parts[1].strip()))
    log.info("보충 시드 %d개 로드", len(out))
    return out


def main():
    cfg = load_config()
    g = cfg["gazetteer"]
    db = find_cbdb(cfg)
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)

    user_entries = load_seed(g.get("user_dict"))
    user_surfaces = {s for s, _ in user_entries}
    raw = (extract_per(con, g["dynasty_codes"], g["year_start"], g["year_end"])
           + extract_ofi(con)
           + extract_loc(con)
           + load_seed(g.get("seed_supplement"))
           + user_entries)
    if g.get("extract_names_from_zi", True):
        raw += extract_per_from_zi(con, cfg, int(g.get("surname_min_persons", 20)))
    con.close()

    # 정제: 괄호주기·공백 제거 → DataFrame
    df = pd.DataFrame(raw, columns=["surface", "type"])
    df["surface"] = df["surface"].map(clean_surface)
    df = df.dropna(subset=["surface"])

    # 2자 미만 제거
    minlen = int(g.get("min_surface_len", 2))
    df = df[df["surface"].str.len() >= minlen]

    # §3.4 sanity 필터: 비개체 stoplist 제거
    stop = load_stoplist(g.get("stoplist"))
    if stop:
        n0 = len(df)
        # 사용자 사전 항목은 stoplist 면제
        df = df[~df["surface"].isin(stop) | df["surface"].isin(user_surfaces)]
        log.info("stoplist 적용: %d → %d 표면형 (user_dict %d 면제)",
                 n0, len(df), len(user_surfaces))

    # 자형 정규화 — 코퍼스와 동일 설정(normalize)으로 가제티어도 통일(簡↔繁·異體字).
    nm = cfg.get("normalize", {})
    normalize = make_script_normalizer(
        nm.get("corpus_opencc"), nm.get("opencc_protect", ""),
        load_variant_map(nm.get("variant_map")))
    df["surface"] = df["surface"].map(normalize)

    # 유형 우선순위(PER>OFI>LOC)로 중복 제거: 동일 표면형은 한 유형만
    prio = {"PER": 0, "APP": 1, "ERA": 2, "OFI": 3, "LOC": 4}
    df["_p"] = df["type"].map(prio).fillna(9)
    df = df.sort_values("_p").drop_duplicates(subset=["surface"], keep="first")
    df = df.drop(columns="_p")
    n_before_occ = len(df)

    # 코퍼스 출현 필터 (三國 스코핑)
    n_total_cand = n_before_occ
    if g.get("require_corpus_occurrence", True):
        norm = pd.read_parquet(resolve(cfg, "interim") / "normalized.parquet")
        corpus_text = "".join(norm["text"].tolist())
        mask = df["surface"].map(lambda s: s in corpus_text)
        df = df[mask]
        log.info("코퍼스 출현 필터: %d → %d 표면형", n_before_occ, len(df))

    # 길이 내림차순 정렬(최장일치)
    df = df.assign(_len=df["surface"].str.len()).sort_values(
        ["_len", "surface"], ascending=[False, True]).drop(columns="_len")
    df = df.reset_index(drop=True)

    out_dir = ensure_dir(resolve(cfg, "gazetteer"))
    out = out_dir / "gazetteer.tsv"
    df[["surface", "type"]].to_csv(out, sep="\t", index=False)

    stats = {
        "cbdb_sqlite": db.name,
        "candidates_total": int(n_total_cand),
        "final_entries": int(len(df)),
        "by_type": df["type"].value_counts().to_dict(),
        "len_distribution": df["surface"].str.len().value_counts().sort_index().to_dict(),
        "require_corpus_occurrence": bool(g.get("require_corpus_occurrence", True)),
        "license": "CBDB CC BY-NC-SA 4.0 (place/office/person data); CHGIS for places.",
    }
    (out_dir / "gazetteer_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    log.info("gazetteer.tsv 작성: %d 표면형 -> %s", len(df), out)
    log.info("  유형별: %s", stats["by_type"])
    log.info("  길이분포: %s", {k: v for k, v in list(stats["len_distribution"].items())[:8]})


if __name__ == "__main__":
    main()
