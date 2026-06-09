"""Stage 4 — 토큰화 (핵심): 결정론적 전방 최장일치 병합.

명세 §4:
  - 가제티어를 길이순 사전(+max 길이)로 로드.
  - 각 segment에서 전방 최장일치로 가제티어 항목을 1토큰 병합.
  - 매칭 안 된 위치는 글자 1개 = 토큰 1개.
  - 文/注 플래그(is_peizhu)·segment 경계 보존.
  - 금지: 통계 분절, subword 분해, 별명 치환.

대상: kind ∈ {main, peizhu} (陳壽 本文 + 裴松之 注). 청대 考證·front matter는 三國志 본체가
아니므로 학습 코퍼스에서 제외(raw/interim 에는 보존됨).

출력: data/tokenized/corpus.jsonl — 각 행 {segment_id, kind, is_peizhu, juan, tokens:[...]}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ensure_dir, get_logger, load_config, resolve  # noqa: E402

log = get_logger("04_tokenize")


class MaxMatcher:
    """전방 최장일치 병합기. 가제티어 표면형 집합 + 최대 길이로 동작."""

    def __init__(self, surfaces: set[str]):
        self.vocab = surfaces
        self.maxlen = max((len(s) for s in surfaces), default=1)
        # 가지치기용 prefix 집합(선택): 길이별 존재 여부로 조기 종료 가능하나
        # maxlen이 작아(≤6) 단순 길이 역순 탐색으로 충분.

    def tokenize(self, text: str) -> list[str]:
        out = []
        i, n = 0, len(text)
        while i < n:
            matched = None
            hi = min(self.maxlen, n - i)
            # 최장(길이 hi)부터 2까지 시도 — 첫 매칭이 최장일치
            for L in range(hi, 1, -1):
                cand = text[i:i + L]
                if cand in self.vocab:
                    matched = cand
                    break
            if matched is not None:
                out.append(matched)
                i += len(matched)
            else:
                out.append(text[i])  # 글자 1개 = 토큰 1개
                i += 1
        return out


def main():
    cfg = load_config()

    gaz = pd.read_csv(resolve(cfg, "gazetteer") / "gazetteer.tsv", sep="\t")
    surfaces = set(gaz["surface"].astype(str))
    matcher = MaxMatcher(surfaces)
    log.info("가제티어 %d 표면형 로드 (최대길이 %d)", len(surfaces), matcher.maxlen)

    norm = pd.read_parquet(resolve(cfg, "interim") / "normalized.parquet")
    # 학습 대상 = 三國志 본체(main + peizhu). 편집 텍스트(kaozheng/frontmatter) 제외.
    target = norm[norm["kind"].isin(["main", "peizhu"])].reset_index(drop=True)
    log.info("토큰화 대상 세그먼트: %d (제외: kaozheng/frontmatter %d)",
             len(target), len(norm) - len(target))

    out_dir = ensure_dir(resolve(cfg, "tokenized"))
    out = out_dir / "corpus.jsonl"

    n_tokens = 0
    n_entity_tokens = 0
    with open(out, "w", encoding="utf-8") as f:
        for r in target.itertuples(index=False):
            toks = matcher.tokenize(r.text)
            n_tokens += len(toks)
            n_entity_tokens += sum(1 for t in toks if len(t) > 1)
            rec = {
                "segment_id": r.segment_id,
                "kind": r.kind,
                "is_peizhu": bool(r.is_peizhu),
                "juan": r.juan,
                "tokens": toks,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    log.info("corpus.jsonl 작성: %d 세그먼트, %d 토큰(개체토큰 %d, %.1f%%) -> %s",
             len(target), n_tokens, n_entity_tokens,
             100 * n_entity_tokens / max(n_tokens, 1), out)


if __name__ == "__main__":
    main()
