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
# 樂은 문맥마다 락/악/요로 갈려(樂浪=낙랑·樂安=낙안 ↔ 樂進=악진) 일반 규칙이 위험 → 토큰 단위로.
# 전 코퍼스 공유 — 토큰이 어휘에 있을 때만 적용되므로 사서 간 무해 (docs/ADDING_AN_EXPLORER.md §5).
READING_OVERRIDES: dict[str, str] = {
    # --- 三國志 ---
    "樂進": "악진",       # 인명: 樂=악
    "樂綝": "악침",       # 인명(樂進의 아들): 樂=악
    "樂人": "악인",       # 악공·음악인: 樂=악
    "車騎": "거기",       # 관직(車騎將軍): 車=거
    "奉車都尉": "봉거도위",  # 관직: 車=거
    # --- 史記 ---
    "單于": "선우",       # 흉노 군주 칭호: 單=선
    "單父": "선보",       # 지명: 單=선, 父=보
    "章邯": "장한",       # 인명(秦 장수): 邯=한
    "樂毅": "악의",       # 인명(燕 장수): 성씨 樂=악
    "龍且": "용저",       # 인명(楚 장수): 且=저
    "夏無且": "하무저",   # 인명(始皇 侍醫): 且=저
    "樂羊": "악양",       # 인명(魏 장수): 성씨 樂=악
    "桓齮": "환의",       # 인명(秦 장수): 齮=의
    "閼氏": "연지",       # 흉노 왕비 칭호
    "中行說": "중항열",   # 인명: 中行=중항(씨), 說=열
    "韓說": "한열",       # 인명(漢 장수): 說=열
    "傅說": "부열",       # 인명(殷 재상): 說=열
    "《說難》": "세난",   # 한비자 편명: 說=세
    "主父": "주보",       # 趙 武靈王 칭호·복성: 父=보
    "主父偃": "주보언",   # 인명: 父=보
    "梁父": "양보",       # 지명(梁父山): 父=보
    "城父": "성보",       # 지명(楚): 父=보
    "番禺": "번우",       # 지명(南越 도읍): 禺=우
    "皋陶": "고요",       # 인명(舜 신하): 陶=요
    "蒙恬": "몽염",       # 인명(秦 장수): 관행 표기
    "晁錯": "조조",       # 인명(漢 정치가): 관행 표기
    "鼂錯": "조조",       # 인명(= 晁錯 이표기)
    "辟陽侯": "벽양후",   # 봉호(審食其): 辟=벽
    "酈山": "여산",       # 지명(= 驪山): 酈=려
    "摢裏疾": "저리질",   # 인명(= 樗里疾 이표기)
    # --- 漢書 ---
    "冒頓": "묵돌",       # 흉노 單于(冒頓): 관행 표기(墨毒)
    "金日磾": "김일제",   # 인명(흉노 출신 漢 신하): 성씨 金=김, 磾=제
    "兒寬": "예관",       # 인명(= 倪寬): 성씨 兒=예
    "呼韓邪": "호한야",   # 흉노 單于 칭호: 邪=야
    "車師": "거사",       # 西域國(車師前/後國): 車=거
    "輕車將軍": "경거장군",  # 將軍號: 車=거(輕車=병거)
}


def fix_special_readings(bare: str, reading: str) -> str:
    """문맥 의존 특수음 교정. 한자 1자 = 독음 1음절 가정(순수 한자 토큰).
    특수음:
      - 僕射의 射는 '사'가 아니라 '야' → 복야·상서좌복야
      - 祭酒의 祭는 '제'가 아니라 '좨' → 좨주·군사좨주
      - 邯鄲의 邯은 '감'이 아니라 '한' → 한단·한단순
      - 行狀의 狀은 '상'이 아니라 '장' → 행장(先賢行狀)
      - 寧은 hanja가 비어두에서 '령'으로 잘못 주므로 '녕'으로 교정(어두는 두음 '영' 유지)
        → 甘寧 감녕·管寧 관녕·安寧 안녕, 寧國 영국(어두)은 그대로.
      - 月氏의 氏는 '씨'가 아니라 '지' → 월지·대월지·소월지
      - 閼氏(흉노 왕비 칭호)는 '알씨'가 아니라 '연지' → 연지·전거연지(顓渠閼氏)·대연지(大閼氏)·선우연지(單于閼氏)
      - 邪는 琅邪·渾邪·昆邪에서 '사'가 아니라 '야' → 낭야·혼야왕·곤야왕
      - 中行·太行의 行은 '행'이 아니라 '항' → 중항(씨)·태항(산). 大行(대행)은 그대로.
      - 車騎·車府·車師(西域國)·公車(관서)의 車는 '차'가 아니라 '거' → 거기장군·중거부령·거사왕(車師王)·공거사마(公車司馬)
      - 食其(인명)의 食은 '식'이 아니라 '이' → 역이기(酈食其)·심이기(審食其)
      - 單于의 單은 '단'이 아니라 '선' → 선우·질지선우(郅支單于)·흉노선우(匈奴單于)
      - 身毒(인도 古名)의 身은 '신'이 아니라 '연' → 연독·연독국(身毒國)
      - 冒頓(흉노 單于)은 '모돈'이 아니라 '묵돌' → 묵돌·묵돌선우(冒頓單于)
      - 休屠의 屠는 '도'가 아니라 '저' → 휴저왕(休屠王). (屠耆 등은 '도' 유지)"""
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
        elif ch == "寧" and not (i == 0 and len(bare) > 1):  # 어두(다자) 외에는 '녕'
            out[i] = "녕"
        elif ch == "劭":          # hanja가 '초'로 잘못 줌 → 항상 '소'(應劭 응소·劉劭 유소)
            out[i] = "소"
        elif ch == "氏" and i > 0 and bare[i - 1] in "月閼":
            out[i] = "지"          # 月氏=월지·閼氏=연지
        elif ch == "閼" and i + 1 < len(bare) and bare[i + 1] == "氏":
            out[i] = "연"          # 閼氏=연지 (흉노 왕비 칭호)
        elif ch == "邪" and i > 0 and bare[i - 1] in "琅渾昆":
            out[i] = "야"          # 邪=야 (琅邪 낭야·渾邪王 혼야왕·昆邪王 곤야왕)
        elif ch == "行" and i > 0 and bare[i - 1] in "中太":
            out[i] = "항"
        elif ch == "車" and (
            (i + 1 < len(bare) and bare[i + 1] in "騎府師")   # 車騎·車府·車師(거사)
            or (i > 0 and bare[i - 1] == "公")                # 公車(공거)司馬
        ):
            out[i] = "거"
        elif ch == "食" and i + 1 < len(bare) and bare[i + 1] == "其":
            out[i] = "이"
        elif ch == "單" and i + 1 < len(bare) and bare[i + 1] == "于":
            out[i] = "선"          # 흉노 君長號 單于=선우 → 郅支單于 질지선우·匈奴單于 흉노선우
        elif ch == "身" and i + 1 < len(bare) and bare[i + 1] == "毒":
            out[i] = "연"          # 身毒=연독 (인도 古名) → 身毒國 연독국
        elif ch == "冒" and i + 1 < len(bare) and bare[i + 1] == "頓":
            out[i] = "묵"          # 冒頓=묵돌 (흉노 單于) → 冒頓單于 묵돌선우
        elif ch == "頓" and i > 0 and bare[i - 1] == "冒":
            out[i] = "돌"
        elif ch == "屠" and i > 0 and bare[i - 1] == "休":
            out[i] = "저"          # 休屠=휴저 (匈奴 王號; 金日磾가 休屠王 太子). 屠耆 등은 도 유지
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
            # 단자도 특수음 교정 적용(寧→녕 등). 본음 계산 후 보정.
            return fix_special_readings(bare, _bonum(bare, translate))
        # 다자 토큰: 어두 두음법칙은 정상(劉備→유비, 諸葛亮→제갈량)
        return fix_special_readings(bare, translate(bare, "substitution"))
    except Exception:
        return ""


def main():
    cfg = load_config()
    from gensim.models import Word2Vec
    import hanja

    model_name = cfg["word2vec"].get("model_name", "w2v_sanguozhi.model")
    model_path = resolve(cfg, "models") / model_name
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

    out_dir = ensure_dir(resolve(cfg, "web") / "data")

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
