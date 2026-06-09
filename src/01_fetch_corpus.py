"""Stage 1 — 원문 수집 및 本文/裴注 분리.

정본: Kanseki Repository KR2a0012 (三國志, 文淵閣四庫全書本 WYG; 陳壽 撰 + 裴松之 注).
Kanripo 파일은 Mandoku/KR 마크업(org-mode 헤더 + 평문 본문)이다. 관찰된 마크업:
  - `#+...` org 헤더 (파일당 PROPERTY: JUAN 등)        → 제거
  - `<pb:...>` 판심 페이지 경계 마커 (2163개)          → 제거
  - `¶`        목판 행/열 경계 (레이아웃, 문장 아님)    → 제거
  - `( ... )`  雙行夾注 = 裴松之 注 (반각괄호, 전역 균형) → 注 스트림
  - 괄호 밖     = 本文 (陳壽)                            → 本文 스트림
각 卷 끝에는 `[魏蜀吳]志卷N考證` (청대 四庫館臣 校勘 주석) 이 붙으며, 이는 陳壽 本文도
裴松之 注도 아니므로 kind='kaozheng'로 분리하여 학습에서 제외한다(raw 에는 보존).
파일 _000 은 御製詩·目録·提要 등 front matter → kind='frontmatter'로 분리.

설계(명세서 §2-6): 별명/단자 정규화·분절 일절 없음. 여기서는 텍스트 추출과 文/注 분리만.

출력:
  data/raw/provenance.json                      — 출처·판본 메타데이터
  data/interim/segments.parquet                 — segment_id, text, source, juan, kind, is_peizhu
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ensure_dir, get_logger, load_config, resolve  # noqa: E402

log = get_logger("01_fetch")

PB_RE = re.compile(r"<pb:[^>]*>")
HEADER_RE = re.compile(r"^#.*$", re.MULTILINE)
JUAN_PROP_RE = re.compile(r"^#\+PROPERTY:\s*JUAN\s+(.+?)\s*$", re.MULTILINE)
# 卷 말미 考證 섹션 헤딩: 例) 魏志卷一考證 / 蜀志卷五考證 / 吳志卷二考證
KAOZHENG_RE = re.compile(r"[魏蜀吳]志卷[一二三四五六七八九十百零〇]+考證")
# 卷 말미 colophon (考證 직전 또는 本文 말미): 例) 魏志卷一
COLOPHON_TAIL_RE = re.compile(r"[魏蜀吳]志卷[一二三四五六七八九十百零〇]+$")
# 卷 머리 서지·찬자 보일러플레이트:
#   欽定四庫全書 + [魏蜀吳]志卷N + (byline) 晉著作郎…陳壽撰…裴松之注
# 30권 중 29권에 존재(卷十六은 판각상 머리글 없이 본문 시작 → 앵커 불일치로 자연 보존).
JUAN_HEADER_RE = re.compile(
    r"^欽定四庫全書[魏蜀吳]志卷[一二三四五六七八九十百零〇]+(?:.{0,40}?松之注)?"
)


def clone_corpus(cfg: dict) -> Path:
    vendor = ensure_dir(resolve(cfg, "vendor"))
    dest = vendor / cfg["corpus"]["kanripo_id"]
    if dest.exists():
        log.info("Kanripo repo 이미 존재: %s", dest)
        return dest
    url = cfg["corpus"]["kanripo_repo"]
    log.info("clone %s -> %s", url, dest)
    subprocess.run(["git", "clone", "--depth", "1", url, str(dest)], check=True)
    return dest


def clean_markup(raw: str) -> str:
    """org 헤더·pb 마커·¶ 제거 후 단일 문자열 반환."""
    t = HEADER_RE.sub("", raw)
    t = PB_RE.sub("", t)
    t = t.replace("¶", "")
    t = t.replace("\n", "")
    t = t.replace("　", "")  # 전각 공백(편집 들여쓰기) 제거
    return t


def split_kaozheng(text: str) -> tuple[str, str]:
    """本文+注 부분과 考證 부분을 분리. 考證 없으면 (text, '')."""
    m = KAOZHENG_RE.search(text)
    if not m:
        return text, ""
    main = text[: m.start()]
    kaozheng = text[m.start():]
    # 本文 말미 colophon (例: 魏志卷一) 제거
    main = COLOPHON_TAIL_RE.sub("", main)
    return main, kaozheng


def partition_main_note(text: str):
    """반각 괄호 깊이로 本文(depth 0) / 裴注(depth>0) 런 분할.

    반환: [(kind, run_text), ...]  kind ∈ {'main','peizhu'}.
    목판 열-분할로 인접한 `)(` (사이 本文 0글자)는 동일 注로 병합된다:
    빈 런을 버린 뒤 연속 동일 kind 런을 합친다.
    """
    runs: list[tuple[str, list[str]]] = []
    depth = 0
    cur_kind = "main"
    buf: list[str] = []

    def flush():
        if buf:
            runs.append((cur_kind, "".join(buf)))

    for ch in text:
        if ch == "(":
            if depth == 0:
                flush(); buf.clear(); cur_kind = "peizhu"
            depth += 1
            continue
        if ch == ")":
            depth = max(0, depth - 1)
            if depth == 0:
                flush(); buf.clear(); cur_kind = "main"
            continue
        buf.append(ch)
    flush()

    # 빈 런 제거 + 연속 동일 kind 병합
    merged: list[tuple[str, str]] = []
    for kind, txt in runs:
        if not txt:
            continue
        if merged and merged[-1][0] == kind:
            merged[-1] = (kind, merged[-1][1] + txt)
        else:
            merged.append((kind, txt))
    return merged


def build_segments(cfg: dict, repo: Path) -> pd.DataFrame:
    kid = cfg["corpus"]["kanripo_id"]
    concat_main = not cfg["corpus"].get("peizhu_splits_bentext", False)
    rows = []
    files = sorted(repo.glob(f"{kid}_*.txt"))
    for fp in files:
        raw = fp.read_text(encoding="utf-8")
        jm = JUAN_PROP_RE.search(raw)
        juan = jm.group(1).strip() if jm else fp.stem.split("_")[-1]
        idx = fp.stem.split("_")[-1]

        # _000 = front matter 전체
        if idx == "000":
            t = clean_markup(raw)
            if t:
                rows.append(dict(segment_id=f"{kid}_{idx}_front",
                                 text=t, source=kid, juan=juan,
                                 kind="frontmatter", is_peizhu=False))
            continue

        cleaned = clean_markup(raw)
        main_note, kaozheng = split_kaozheng(cleaned)

        parts = partition_main_note(main_note)
        main_runs = [txt for k, txt in parts if k == "main"]
        note_runs = [txt for k, txt in parts if k == "peizhu"]

        if concat_main:
            joined = JUAN_HEADER_RE.sub("", "".join(main_runs), count=1)
            if joined:
                rows.append(dict(segment_id=f"{kid}_{idx}_main",
                                 text=joined, source=kid, juan=juan,
                                 kind="main", is_peizhu=False))
        else:
            if main_runs:
                main_runs[0] = JUAN_HEADER_RE.sub("", main_runs[0], count=1)
            for i, txt in enumerate(main_runs):
                if not txt:
                    continue
                rows.append(dict(segment_id=f"{kid}_{idx}_main{i:04d}",
                                 text=txt, source=kid, juan=juan,
                                 kind="main", is_peizhu=False))
        for i, txt in enumerate(note_runs):
            rows.append(dict(segment_id=f"{kid}_{idx}_note{i:04d}",
                             text=txt, source=kid, juan=juan,
                             kind="peizhu", is_peizhu=True))
        if kaozheng:
            rows.append(dict(segment_id=f"{kid}_{idx}_kaozheng",
                             text=kaozheng, source=kid, juan=juan,
                             kind="kaozheng", is_peizhu=False))

    df = pd.DataFrame(rows)
    return df


def main():
    cfg = load_config()
    repo = clone_corpus(cfg)
    df = build_segments(cfg, repo)

    interim = ensure_dir(resolve(cfg, "interim"))
    out = interim / "segments.parquet"
    df.to_parquet(out, index=False)

    # provenance
    raw_dir = ensure_dir(resolve(cfg, "raw"))
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    prov = {
        "corpus": "三國志 (正史) — Chen Shou 撰, Pei Songzhi 注",
        "source": "Kanseki Repository (kanripo)",
        "repo": cfg["corpus"]["kanripo_repo"],
        "repo_id": cfg["corpus"]["kanripo_id"],
        "edition": "文淵閣四庫全書本 (WYG)",
        "git_head": head,
        "license": "Kanripo terms; see https://www.kanripo.org/",
        "note": "괄호 안=裴松之 注, 괄호 밖=陳壽 本文. 考證=청대 四庫館臣 교감(학습 제외). _000=front matter.",
    }
    (raw_dir / "provenance.json").write_text(
        json.dumps(prov, ensure_ascii=False, indent=2), encoding="utf-8")

    # 요약 로그
    by_kind = df.groupby("kind").agg(n_seg=("text", "size"),
                                     n_char=("text", lambda s: s.str.len().sum()))
    log.info("segments.parquet 작성: %d 세그먼트 -> %s", len(df), out)
    for kind, r in by_kind.iterrows():
        log.info("  %-11s seg=%5d  chars=%d", kind, int(r.n_seg), int(r.n_char))


if __name__ == "__main__":
    main()
