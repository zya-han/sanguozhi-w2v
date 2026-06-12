"""Stage 1 — 원문 수집 및 本文/注 분리 (中文維基文庫 zh.wikisource).

코퍼스 불문 공용: `corpus.book_title`의 `<書名>/卷…` 실페이지를 allpages API로 발견한다
(卷 네이밍이 사서마다 제각각 — 三國志 卷01, 史記 卷001, 後漢書 卷1 — 이라 range 구성 불가).

위키텍스트 마크업:
  - `{{*|...}}`  = 注 (裴松之注·李賢注 등 인라인 雙行夾注)  → 注 스트림(is_peizhu=True)
    (史記·漢書처럼 注 미수록 사서는 자동으로 main만 나온다)
  - 그 밖의 텍스트 = 本文                                  → 本文 스트림
  - `==전기명==`  = 傳 단위 섹션 헤더 (드롭)
  - `[[링크|표시]]`→표시, `《書名》` 유지, `'''굵게'''` 제거, 기타 {{템플릿}} 제거
표점(。！？)으로 문장 분할 → 각 문장이 윈도 단위 segment (명세 §2.6).
토큰화 대상은 한자(漢字)만; 구두점은 문장 경계로만 쓰고 토큰에서 제외(Stage 4).

출력: data/<id>/raw/wikisource/<제목>.wiki (캐시), data/<id>/raw/provenance.json,
      data/<id>/interim/segments.parquet (segment_id, text, source, juan, shu, kind, is_peizhu).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import regex as re
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ensure_dir, get_logger, load_config, resolve  # noqa: E402

log = get_logger("01_fetch")

# 한자 + 《》 만 보존(《書名》을 토큰화 단계에서 통째로 한 토큰으로 방출하기 위함).
# 나머지 구두점은 문장 경계로만 쓰고 제거한다.
KEEP = re.compile(r"[\p{Han}《》]+")
HAN_ANY = re.compile(r"\p{Han}")
SECTION_RE = re.compile(r"^=+\s*[^=]+?\s*=+\s*$", re.MULTILINE)


UA = {"User-Agent": "han-w2v-research/1.0 (academic; contact via github)"}


def list_juan_pages(cfg: dict) -> list[str]:
    """`<書名>/卷…` 페이지를 allpages API로 발견, 卷 번호 자연 정렬.

    리다이렉트도 포함해 수집한다 — 일부 卷은 편명 페이지(史記/卷092→史記/淮陰侯列傳)로의
    리다이렉트로만 존재한다. 대신 리다이렉트 타깃을 해석해 같은 본문을 가리키는 중복
    (漢書 卷NNN上/下→卷NNN 등)은 낮은 卷 번호 하나만 남긴다. fetch는 redirects=1로 따라간다.
    """
    book = cfg["corpus"]["book_title"]
    api = cfg["corpus"]["wikisource_api"]
    pages, cont = [], None
    while True:
        params = {"action": "query", "list": "allpages",
                  "apprefix": f"{book}/卷", "apnamespace": 0,
                  "apfilterredir": "all", "aplimit": "max", "format": "json"}
        if cont:
            params["apcontinue"] = cont
        r = requests.get(api, params=params, headers=UA, timeout=30)
        r.raise_for_status()
        j = r.json()
        pages += [p["title"] for p in j["query"]["allpages"]]
        cont = j.get("continue", {}).get("apcontinue")
        if not cont:
            break
    # 자연 정렬: 卷 뒤 숫자 + 上中下
    ord_ = {"上": 0, "中": 1, "下": 2}

    def key(t):
        m = re.search(r"卷(\d+)([上中下])?", t)
        return (int(m.group(1)) if m else 9999, ord_.get(m.group(2) if m else None, -1))

    pages = sorted(pages, key=key)

    # 리다이렉트 타깃 해석(배치) → canonical 본문 기준 중복 제거
    target = {}
    for i in range(0, len(pages), 50):
        chunk = pages[i:i + 50]
        r = requests.get(api, params={"action": "query", "titles": "|".join(chunk),
                                      "redirects": 1, "format": "json"},
                         headers=UA, timeout=30)
        r.raise_for_status()
        for rd in r.json().get("query", {}).get("redirects", []):
            target[rd["from"]] = rd["to"]
        time.sleep(0.5)
    seen, out = set(), []
    for t in pages:
        canon = target.get(t, t)
        if canon in seen:
            log.info("중복 본문 생략: %s (= %s)", t, canon)
            continue
        seen.add(canon)
        out.append(t)
    return out


def fetch_juan_wikitext(cfg: dict, title: str) -> str:
    """페이지 위키텍스트를 API로 받되, data/<id>/raw/wikisource/ 에 캐시."""
    cache_dir = ensure_dir(resolve(cfg, "raw") / "wikisource")
    cache = cache_dir / (title.replace("/", "_") + ".wiki")
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    params = {"action": "parse", "page": title, "prop": "wikitext",
              "format": "json", "redirects": 1, "maxlag": 5}
    for attempt in range(6):
        r = requests.get(cfg["corpus"]["wikisource_api"], params=params,
                         headers=UA, timeout=30)
        if r.status_code == 429 or (r.status_code == 200 and "maxlag" in r.text[:200]):
            wait = int(r.headers.get("Retry-After", 2 ** attempt + 2))
            log.warning("%s rate-limit(%s), %ds 대기 후 재시도", title, r.status_code, wait)
            time.sleep(wait)
            continue
        r.raise_for_status()
        wt = r.json().get("parse", {}).get("wikitext", {}).get("*", "")
        cache.write_text(wt, encoding="utf-8")
        time.sleep(1.0)  # 예의상 rate limit
        return wt
    raise RuntimeError(f"{title} 반복 rate-limit으로 실패")


REF_RE = re.compile(r"<ref[^>/]*>.*?</ref>|<ref[^>]*/>", re.S)
LANGCONV_RE = re.compile(r"-\{(.*?)\}-", re.S)
# 末尾 校勘記 섹션(==校勘記==·==【校勘記】== 等) → EOF 통째 드롭. 後人 교감 apparatus
# (「X頁Y行…按：殿本考證…據汲本」)라 原文이 아니며, 板本名·編者名으로 vocab을 오염시킨다.
# 後漢書 953세그먼트 출처. 史記·漢書 raw에도 존재(재실행 시 혜택). 末尾 섹션 전제(실측).
KANKAN_RE = re.compile(r"\n=+[ 　]*【?[ 　]*校勘記[ 　]*】?[ 　]*=+.*\Z", re.S)
# 본문 내용을 감싸기만 하는 템플릿(專名號·표시·색상·인용 등) — 내용을 살린다.
TMPL_INNER_RE = re.compile(r"\{\{([^{}|]*)((?:\|[^{}]*)?)\}\}")
UNWRAP_NAMES = {"propernoun", "專", "標", "deeppink", "green", "yl", "quote"}
FIRST_ARG_NAMES = {"另", "參"}   # {{另|本文字|교감주}} → 本文字만 (주는 후대 교감)
# 本文(上書·制詔 등)을 색상으로만 감싼 래퍼 — 내용은 本文이므로 살리되, 안에 박힌
# {{*|師古曰…}}注는 보존해 split_main_note가 peizhu로 분리하게 한다(漢書·三國志 실측).
# TMPL_INNER_RE는 중첩 {{}}가 있으면 매칭 못 해 → 균형 괄호 스캔(unwrap_wrappers)으로 처리.
WRAPPER_NAMES = {"blue", "red", "green", "deeppink", "purple", "brown", "orange", "gray", "grey"}


def pick_langconv(m: re.Match) -> str:
    """`-{zh:脩;zh-hans:修;zh-hant:脩;}-` 언어변환 → zh-hant > zh > 첫 값. `-{於}-` → 於."""
    inner = m.group(1)
    if ":" not in inner:
        return inner
    choices = {}
    for part in inner.split(";"):
        k, sep, v = part.strip().partition(":")
        if sep:
            choices[k.strip()] = v
    return choices.get("zh-hant") or choices.get("zh") or next(iter(choices.values()), "")


def unwrap_wrappers(text: str, names: set) -> str:
    """색상 래퍼 {{name|本文}}을 균형 괄호로 벗겨 本文만 남긴다(중첩 {{*|}}注는 보존).

    {{blue|本文{{*|師古曰…}}本文2}}처럼 注가 박힌 本文 래퍼를 살리기 위해 수동 스캔으로
    바깥 래퍼만 제거한다(TMPL_INNER_RE는 내부 중첩 {{}}가 있으면 매칭 못 함). 래퍼가 아닌
    템플릿({{*|}}·헤더 등)은 그대로 둬 split_main_note가 注 분리·드롭을 처리하게 한다.
    """
    out, i, n = [], 0, len(text)
    while i < n:
        if text[i:i + 2] == "{{":
            depth, j = 0, i
            while j < n:
                if text[j:j + 2] == "{{":
                    depth += 1; j += 2
                elif text[j:j + 2] == "}}":
                    depth -= 1; j += 2
                    if depth == 0:
                        break
                else:
                    j += 1
            inner = text[i + 2:j - 2]
            name, _, rest = inner.partition("|")
            nl = name.strip().lower()
            if nl == "color":
                # {{color|<색상명>|本文}} (後漢書 論曰·贊曰·校勘 강조 등 3-인자 래퍼):
                # 색상명 인자만 버리고 本文은 살린다(2-인자 {{blue|本文}}와 형태가 다름).
                _, _, rest = rest.partition("|")
                out.append(unwrap_wrappers(rest, names))
            elif nl in names:
                out.append(unwrap_wrappers(rest, names))   # 本文만(중첩 래퍼도 재귀)
            else:
                out.append(text[i:j])                       # {{*|}}·헤더 등은 그대로
            i = j
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def preprocess_wikitext(wikitext: str) -> str:
    """split_main_note 전 전처리 (史記·漢書 등에서 실측된 마크업; 三國志에는 영향 없음).

    1. `<ref>校勘記</ref>` 제거 — 後人 교감, 原文 아님.
    2. 언어변환 `-{…}-` → 번체 분기 선택 (간체 분기의 `{{!|代替字|IDS}}`도 함께 버려짐).
    3. 색상 래퍼 {{blue|…}}·{{red|…}}(上書·制詔 등 本文; 내부 {{*|}}注 보존) 균형 언랩 —
       漢書는 治安策·至言·詔書 등 ~3.2만字 本文이 여기 들어 있어 드롭하면 손실(三國志도 동일).
    4. 내용 보유 템플릿 언랩(안쪽부터): {{ProperNoun|X}}·{{專|X}}·{{標|X}}·{{deepPink|X}}·
       {{green|X}}·{{YL|X}}·{{Quote|X}} → X (다중 인자는 이어붙임: {{ProperNoun|江|淮}}→江淮),
       {{WavyBookMark|X}} → 《X》, {{!|X|IDS}}·{{另|X|주}}·{{參|X|주}} → X.
       그 외 템플릿은 보존 → split_main_note({{*|}}·{{註|}}·{{annotate|}})와 드롭이 처리.
    """
    wikitext = KANKAN_RE.sub("\n", wikitext)   # 末尾 ==校勘記== 섹션(後人 교감) 통째 드롭
    wikitext = REF_RE.sub("", wikitext)
    wikitext = LANGCONV_RE.sub(pick_langconv, wikitext)
    wikitext = unwrap_wrappers(wikitext, WRAPPER_NAMES)

    def repl(m: re.Match) -> str:
        name = m.group(1).strip().lower()
        pos = [a for a in m.group(2).lstrip("|").split("|") if "=" not in a] \
            if m.group(2) else []
        if name in UNWRAP_NAMES:
            return "".join(pos)
        if name in FIRST_ARG_NAMES or name == "!":
            return pos[0] if pos else "\n"
        if name == "wavybookmark":
            return "《" + "".join(pos) + "》"
        return m.group(0)

    prev = None
    while prev != wikitext:
        prev = wikitext
        wikitext = TMPL_INNER_RE.sub(repl, wikitext)
    return wikitext


def split_main_note(wikitext: str) -> list[tuple[str, str]]:
    """{{...}} 균형 파싱으로 本文/注 분리.

    `{{*|...}}`·`{{註|...}}`·`{{annotate|...}}` → ('peizhu', 내용). 그 외 템플릿 → 드롭.
    나머지 → ('main', ...). 인라인 注가 本文을 끊으므로 호출부에서 main 조각을 이어붙인다.
    (漢書 顏注는 {{*|師古曰…}}, 番號注 〔一〕는 {{annotate|<poem>…劉德曰…</poem>}}로 수록.)
    """
    out: list[tuple[str, str]] = []
    buf: list[str] = []
    i, n = 0, len(wikitext)
    while i < n:
        if wikitext[i:i + 2] == "{{":
            # 매칭되는 }} 까지 (중첩 허용)
            depth, j = 0, i
            while j < n:
                if wikitext[j:j + 2] == "{{":
                    depth += 1; j += 2
                elif wikitext[j:j + 2] == "}}":
                    depth -= 1; j += 2
                    if depth == 0:
                        break
                else:
                    j += 1
            inner = wikitext[i + 2:j - 2]
            if inner.startswith("*|"):           # 인라인 注(裴注·三家注·顏注 等)
                out.append(("main", "".join(buf))); buf = []
                out.append(("peizhu", inner[2:]))
            elif inner.startswith("註|"):        # 年表 등 소주(小註)
                out.append(("main", "".join(buf))); buf = []
                out.append(("peizhu", inner[2:]))
            elif inner.startswith("annotate|"):  # 番號注 〔一〕(漢書 顏注 등)
                out.append(("main", "".join(buf))); buf = []
                out.append(("peizhu", inner[len("annotate|"):]))
            # 그 외 템플릿(헤더/라이선스 등) → 드롭
            i = j
        else:
            buf.append(wikitext[i]); i += 1
    out.append(("main", "".join(buf)))
    return out


def clean_wiki(text: str) -> str:
    """위키 마크업 제거 → 표점 포함 평문. 줄바꿈은 문장 flush 경계로 보존."""
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"</?(?:onlyinclude|noinclude|includeonly|ref|references|small|sub|sup"
                  r"|div|span|center|u|br)[^>]*>", "", text)
    text = re.sub(r"__\w+__", "", text)                       # __FORCETOC__ 등
    # 분류/파일 링크는 언랩 전에 통째 제거 — [[Category:香港…]]이 본문에 누출되는 버그 방지
    text = re.sub(r"\[\[\s*(?:Category|category|File|Image|分類|文件)\s*:[^\]]*\]\]", "", text)
    text = re.sub(r"\[\[(?:[^\[\]|]*\|)?([^\[\]|]+)\]\]", r"\1", text)  # [[a|b]]→b, [[a]]→a
    text = re.sub(r"\[\[[^\]]*\]\]", "", text)                 # 잔여 링크([[Category:..]])
    text = re.sub(r"'{2,}", "", text)                         # ''' '' 강조
    text = re.sub(r"\{\{[^{}]*\}\}", "", text)                 # 잔여 단순 템플릿
    text = SECTION_RE.sub("\n", text)                         # ==전기명== 헤더 드롭
    # 표(年表) 마크업: 표 시작/끝/행 구분 드롭, 셀 경계는 줄바꿈 → 셀 단위 문장 flush
    text = re.sub(r"^(?:\{\||\|\}|\|-).*$", "", text, flags=re.M)
    text = text.replace("||", "\n").replace("!!", "\n")
    text = re.sub(r"^[|!]", "", text, flags=re.M)
    # 판본 표기 행(편집 메타, 原文 아님) 드롭: 〔漢〕司馬遷撰〔宋〕裴駰集解〔唐〕司馬貞索隱 …
    text = re.sub(r"^[^\n]*〔[^〕\n]{1,8}〕[^\n]*(?:撰|集解|索隱|正義|箋|疏)[^\n]*$",
                  "", text, flags=re.M)
    # 잔여 인쇄 가능 ASCII 일괄 제거 — 표 셀 속성(colspan="2")·<p> 등 마크업 잔재가
    # 글자 토큰(c·o·l·s…)으로 새는 것을 차단. 고전한문 본문에 ASCII는 정당한 용처가 없다.
    text = re.sub(r"[!-~]+", "", text)
    # 위키소스 편집 부호 정규화(後漢書 志·傳 실측; 다른 사서엔 거의 없어 무해):
    text = text.translate(str.maketrans({
        "“": "「", "”": "」", "‘": "『", "’": "』",  # 곡선 인용부 → 파이프라인 인용부
        "﹑": "、",                                   # 작은 모점(열거 구분) → 、
        "〖": "", "〗": "",                          # 郡國志 縣名 묶음표 → 벗김(내용 보존)
        "◎": "", "○": "",                           # 郡國志 항목 圈點 → 제거
    }))
    text = re.sub(r"[０-９Ａ-Ｚａ-ｚ]+", "", text)   # 全角 영숫자(편집 잔재; 全角 ！？。는 보존)
    text = re.sub(r"[�-]", "", text)  # 대체문자·PUA 제거(缺字 □는 보존)
    text = text.replace("　", "").replace("​", "")
    return text


def to_sentences(cleaned: str, openers: set, closers: set) -> list[str]:
    """(이미 clean_wiki된) 텍스트를 문장 경계 표점으로 분할.

    - 닫는 부호(。！？」』): 그 '뒤'에서 분리 — 직전 세그먼트의 끝 토큰으로 붙는다.
      연속 닫는 부호(。」)는 함께 붙는다.
    - 여는 부호(「『): 그 '앞'에서 분리 — 다음(인용) 세그먼트의 첫 토큰으로 붙는다.
    - 문장 내 부호(，、；：)는 경계가 아니라 세그먼트에 남아 토큰 겸 병합 barrier가 된다.
    - 줄바꿈(문단·표 셀 경계)에서도 flush — 표점 없는 문단/셀이 이웃과 병합되지 않게.
    공백만 제거. 한자가 없는 세그먼트는 제외.
    """
    sents = []
    cur: list[str] = []

    def flush():
        kept = "".join("".join(cur).split())
        if HAN_ANY.search(kept):
            sents.append(kept)
        cur.clear()

    i, n = 0, len(cleaned)
    while i < n:
        ch = cleaned[i]
        if ch == "\n":                    # 문단/셀 경계에서 flush
            flush()
            i += 1
        elif ch in openers:               # 여는 부호 앞에서 분리
            flush()
            cur.append(ch)
            i += 1
        elif ch in closers:               # 연속 닫는 부호 뒤에서 분리
            while i < n and cleaned[i] in closers:
                cur.append(cleaned[i])
                i += 1
            flush()
        else:
            cur.append(ch)
            i += 1
    flush()
    return sents


def shu_of(cfg: dict, juan_no: int) -> str:
    """卷 번호 → 志/書 구분. config `corpus.shu_map`([[마지막卷, 이름], …]) 없으면 '本'."""
    for upto, name in cfg["corpus"].get("shu_map") or []:
        if juan_no <= int(upto):
            return name
    return "本"


def build_segments(cfg: dict) -> pd.DataFrame:
    openers = set(cfg["corpus"]["sentence_open"])
    closers = set(cfg["corpus"]["sentence_close"])
    cid = cfg["corpus"].get("id", "corpus")
    book = cfg["corpus"]["book_title"]
    titles = list_juan_pages(cfg)
    log.info("allpages 발견: %s 卷 페이지 %d개 (%s … %s)",
             book, len(titles), titles[0], titles[-1])
    rows = []
    for title in titles:
        wt = fetch_juan_wikitext(cfg, title)
        if not wt.strip():
            log.warning("%s 위키텍스트 비어있음", title)
            continue
        parts = split_main_note(preprocess_wikitext(wt))
        main_clean = clean_wiki("".join(t for k, t in parts if k == "main"))
        note_cleans = [clean_wiki(t) for k, t in parts if k == "peizhu"]

        jid = title.split("/", 1)[1] if "/" in title else title   # 예: 卷001
        m = re.search(r"卷(\d+)", title)
        shu = shu_of(cfg, int(m.group(1)) if m else 0)
        for si, sent in enumerate(to_sentences(main_clean, openers, closers)):
            rows.append(dict(segment_id=f"{cid}_{jid}_m{si:04d}", text=sent,
                             source=f"zh.wikisource:{book}", juan=jid, shu=shu,
                             kind="main", is_peizhu=False))
        ni = 0
        for nc in note_cleans:
            for sent in to_sentences(nc, openers, closers):
                rows.append(dict(segment_id=f"{cid}_{jid}_n{ni:05d}", text=sent,
                                 source=f"zh.wikisource:{book}", juan=jid, shu=shu,
                                 kind="peizhu", is_peizhu=True))
                ni += 1
    return pd.DataFrame(rows)


def main():
    cfg = load_config()
    df = build_segments(cfg)
    # 청대 考證·front matter는 wikisource 본문에 포함되지 않으므로 별도 제외 불필요.

    interim = ensure_dir(resolve(cfg, "interim"))
    df.to_parquet(interim / "segments.parquet", index=False)

    raw_dir = ensure_dir(resolve(cfg, "raw"))
    book = cfg["corpus"]["book_title"]
    juans = df["juan"].drop_duplicates().tolist()
    prov = {
        "corpus": cfg["corpus"].get("provenance", book),
        "source": f"中文維基文庫 (zh.wikisource.org) {book}",
        "api": cfg["corpus"]["wikisource_api"],
        "juan": f"{len(juans)}卷, allpages 발견 ({juans[0]}–{juans[-1]})",
        "license": "CC BY-SA 4.0 (Wikisource)",
        "note": "{{*|...}}=注(있는 경우), 그 외=本文. 표점으로 문장 분할, 한자만 토큰화.",
    }
    (raw_dir / "provenance.json").write_text(
        json.dumps(prov, ensure_ascii=False, indent=2), encoding="utf-8")

    by = df.groupby(["shu", "kind"]).agg(
        n=("text", "size"), chars=("text", lambda s: s.str.len().sum()))
    log.info("segments.parquet: %d 문장 세그먼트", len(df))
    for (shu, kind), r in by.iterrows():
        log.info("  %s/%-6s seg=%5d chars=%d", shu, kind, int(r.n), int(r.chars))


if __name__ == "__main__":
    main()
