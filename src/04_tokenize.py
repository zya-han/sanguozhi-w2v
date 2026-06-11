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
import regex as re

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ensure_dir, get_logger, load_config, resolve  # noqa: E402

log = get_logger("04_tokenize")


HAN = re.compile(r"\p{Han}")


def _is_han(ch: str) -> bool:
    return bool(HAN.match(ch))


class MaxMatcher:
    """전방 최장일치 병합기. 가제티어 표면형 집합 + 최대 길이로 동작.

    문장부호 등 비(非)한자는 **병합 경계**다: 최장일치는 한자 연속 구간(run) 안에서만
    이뤄지므로 "相，國"의 相·國이 相國으로 병합되지 않는다.
    punctuation_as_token=True면 부호를 독립 토큰으로 방출, False면 드롭.
    """

    def __init__(self, surfaces: set[str], punctuation_as_token: bool = True):
        self.vocab = surfaces
        self.maxlen = max((len(s) for s in surfaces), default=1)
        self.punct_token = punctuation_as_token

    def tokenize(self, text: str) -> list[str]:
        out = []
        i, n = 0, len(text)
        while i < n:
            ch = text[i]
            # 《書名》 = 괄호 포함 통째로 한 토큰 (가제티어보다 우선)
            if ch == "《":
                j = text.find("》", i + 1)
                if j != -1:
                    out.append(text[i:j + 1])
                    i = j + 1
                    continue
                i += 1  # 닫는 》 없는 고립 《
                continue
            if not _is_han(ch):
                # 부호 등 비한자 = 병합 경계. 토큰 포함 여부는 옵션.
                if self.punct_token and ch != "》":
                    out.append(ch)
                i += 1
                continue
            # 한자 연속 구간(run) 안에서만 최장일치 — run 경계(부호)를 넘지 않음
            run_end = i
            while run_end < n and _is_han(text[run_end]):
                run_end += 1
            hi = min(self.maxlen, run_end - i)
            matched = None
            for L in range(hi, 1, -1):
                cand = text[i:i + L]
                if cand in self.vocab:
                    matched = cand
                    break
            if matched is not None:
                out.append(matched)
                i += len(matched)
            else:
                out.append(ch)  # 글자 1개 = 토큰 1개
                i += 1
        return out


def main():
    cfg = load_config()

    gaz = pd.read_csv(resolve(cfg, "gazetteer") / "gazetteer.tsv", sep="\t")
    surfaces = set(gaz["surface"].astype(str))
    punct_token = cfg.get("tokenize", {}).get("punctuation_as_token", True)
    matcher = MaxMatcher(surfaces, punctuation_as_token=punct_token)
    log.info("가제티어 %d 표면형 로드 (최대길이 %d, 부호토큰=%s)",
             len(surfaces), matcher.maxlen, punct_token)

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
