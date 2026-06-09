"""Stage 7 — 웹 탐색기용 데이터 추출.

`models/w2v_sanguozhi.model` → 브라우저에서 코사인 유사도를 직접 계산할 수 있는
정적 에셋으로 변환한다. 백엔드 없이 정적 호스팅(zyahan.blog 하위 경로 등)으로 서비스.

출력 (web/data/):
  - vectors.bin   : Float32Array, [count × dim] row-major, **각 벡터 L2 정규화**.
                    정규화해 두면 브라우저에서 코사인 = 단순 내적(dot)으로 끝남.
  - vocab.json    : {dim, count, tokens[], freq[], is_entity[]}  (인덱스 = vectors.bin 행)
  - readings.json : {token: "한글음"}  — 한자→한글 독음(hanja), 《》 괄호 제거 후.

설계 메모: int8 양자화(~600KB)도 가능하나 현재 float32(~2.3MB)로 충분하므로 채택 안 함.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ensure_dir, get_logger, load_config, resolve  # noqa: E402

log = get_logger("07_export_web")

# 기본 독음이 어색한 고유명사 수동 오버라이드(소수). 필요 시 확장.
READING_OVERRIDES: dict[str, str] = {}


def fix_special_readings(bare: str, reading: str) -> str:
    """문맥 의존 특수음 교정. 한자 1자 = 독음 1음절 가정(순수 한자 토큰).
    특수음:
      - 僕射의 射는 '사'가 아니라 '야' → 복야·상서좌복야
      - 祭酒의 祭는 '제'가 아니라 '좨' → 좨주·군사좨주
      - 邯鄲의 邯은 '감'이 아니라 '한' → 한단·한단순
      - 行狀의 狀은 '상'이 아니라 '장' → 행장(先賢行狀)"""
    if len(reading) != len(bare):
        return reading                       # 1:1 매핑이 아니면 손대지 않음
    out = list(reading)
    for i, ch in enumerate(bare):
        if ch == "射" and i > 0 and bare[i - 1] == "僕":
            out[i] = "야"
        elif ch == "祭" and i + 1 < len(bare) and bare[i + 1] == "酒":
            out[i] = "좨"
        elif ch == "邯" and i + 1 < len(bare) and bare[i + 1] == "鄲":
            out[i] = "한"
        elif ch == "狀" and i > 0 and bare[i - 1] == "行":
            out[i] = "장"
    return "".join(out)


def _bonum(ch: str, translate) -> str:
    """단자(單字)의 본음(本音) — 두음법칙 미적용 독음.
    어두가 아니면 두음이 적용되지 않으므로, 앞에 중립 글자(之)를 붙여
    번역한 뒤 마지막 음절을 취한다. 예: 亮 → (단독 '양') → 본음 '량'."""
    out = translate("之" + ch, "substitution")
    last = out[-1] if out else ch
    if last == ch:                      # 미매핑(한자 그대로) → 단독 결과로 폴백
        solo = translate(ch, "substitution")
        return solo if solo != ch else ""
    return last


def build_reading(token: str, translate) -> str:
    """토큰 → 한글 독음. 《書名》은 괄호 제거 후 독음. 실패하면 빈 문자열.
    단음절 토큰(亮·劉·呂 등 단자 名/姓)은 두음법칙을 적용하지 않고 본음을 쓴다."""
    if token in READING_OVERRIDES:
        return READING_OVERRIDES[token]
    bare = token.strip("《》")
    try:
        if len(bare) == 1:
            return _bonum(bare, translate)
        # 다자 토큰: 어두 두음법칙은 정상(劉備→유비, 諸葛亮→제갈량)
        return fix_special_readings(bare, translate(bare, "substitution"))
    except Exception:
        return ""


def main():
    cfg = load_config()
    from gensim.models import Word2Vec
    import hanja

    model_path = resolve(cfg, "models") / "w2v_sanguozhi.model"
    model = Word2Vec.load(str(model_path))
    wv = model.wv
    dim = wv.vector_size
    tokens = list(wv.index_to_key)
    count = len(tokens)
    log.info("모델 로드: vocab %d, dim %d", count, dim)

    # 벡터 L2 정규화 → float32 row-major
    mat = np.asarray(wv.vectors, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat = (mat / norms).astype(np.float32)

    out_dir = ensure_dir(resolve(cfg, "models").parent / "web" / "data")

    (out_dir / "vectors.bin").write_bytes(mat.tobytes(order="C"))
    log.info("vectors.bin: %.2f MB", (out_dir / "vectors.bin").stat().st_size / 1e6)

    # freq / is_entity (vocab.tsv 우선, 없으면 모델 count)
    freq_map: dict[str, int] = {}
    vocab_tsv = resolve(cfg, "models") / "vocab.tsv"
    if vocab_tsv.exists():
        df = pd.read_csv(vocab_tsv, sep="\t")
        freq_map = dict(zip(df["token"], df["freq"]))
    freq = [int(freq_map.get(t, wv.get_vecattr(t, "count"))) for t in tokens]
    is_entity = [int(len(t.strip("《》")) > 1) for t in tokens]

    vocab = {"dim": dim, "count": count, "tokens": tokens,
             "freq": freq, "is_entity": is_entity}
    (out_dir / "vocab.json").write_text(
        json.dumps(vocab, ensure_ascii=False), encoding="utf-8")

    # 독음
    readings = {}
    for t in tokens:
        r = build_reading(t, hanja.translate)
        if r and r != t:           # 매핑이 일어난 경우만 저장
            readings[t] = r
    (out_dir / "readings.json").write_text(
        json.dumps(readings, ensure_ascii=False), encoding="utf-8")
    log.info("readings.json: %d/%d 토큰 독음 매핑", len(readings), count)

    # ── 검증: JS 방식(정규화 내적)이 gensim most_similar와 일치하는지 대조 ──
    for probe in ["荀彧", "曹操"]:
        if probe not in wv:
            continue
        i = wv.key_to_index[probe]
        sims = mat @ mat[i]                       # 정규화돼 있으므로 = 코사인
        order = np.argsort(-sims)
        js_top = [(tokens[j], float(sims[j])) for j in order if j != i][:5]
        gs_top = wv.most_similar(probe, topn=5)
        ok = all(a[0] == b[0] and abs(a[1] - b[1]) < 1e-4
                 for a, b in zip(js_top, gs_top))
        log.info("대조 %s: %s | JS=%s", probe, "일치" if ok else "불일치!!",
                 ", ".join(f"{t}({s:.3f})" for t, s in js_top))

    log.info("웹 데이터 추출 완료 → %s", out_dir)


if __name__ == "__main__":
    main()
