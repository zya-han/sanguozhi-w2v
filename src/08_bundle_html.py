"""Stage 8 — 단일 HTML 번들.

`web/{index.html,style.css,app.js}` + `web/data/*` 를 외부 파일 의존 없는
**단일 HTML** 한 장으로 합친다. 벡터(.bin)는 base64로 내장 → `file://`에서
더블클릭만으로 열림(서버 불필요). 중복을 피하려 기존 소스를 그대로 인라인한다.

출력: web/sanguozhi_explorer.html  (Stage 7을 먼저 실행해 web/data/* 가 있어야 함)
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import get_logger, load_config, resolve  # noqa: E402

log = get_logger("08_bundle_html")


def main():
    cfg = load_config()
    web = resolve(cfg, "web")
    data = web / "data"

    html = (web / "index.html").read_text(encoding="utf-8")
    css = (web / "style.css").read_text(encoding="utf-8")
    js = (web / "app.js").read_text(encoding="utf-8")

    vocab = (data / "vocab.json").read_text(encoding="utf-8")
    readings = (data / "readings.json").read_text(encoding="utf-8")
    vectors_b64 = base64.b64encode((data / "vectors.bin").read_bytes()).decode("ascii")

    # <script> 안에 들어가는 내용물의 `</`만 escape → 태그 조기종료 방지.
    # (JS 문자열·코드에서 `<\/` 는 `</` 와 동일 의미라 동작 불변. base64는 `<` 없음.)
    def safe(s: str) -> str:
        return s.replace("</", "<\\/")

    # <link rel="stylesheet" ...> → 인라인 <style>
    html = html.replace(
        '<link rel="stylesheet" href="style.css">',
        f"<style>\n{css}\n</style>",
    )

    # 외부 <script src="app.js"> → 내장 데이터 + 인라인 app.js
    embed = (
        "<script>\n"
        "window.__W2V_EMBED__ = {\n"
        f"  vocab: {safe(vocab)},\n"
        f"  readings: {safe(readings)},\n"
        f'  vectorsB64: "{vectors_b64}"\n'
        "};\n"
        "</script>\n"
        f"<script>\n{safe(js)}\n</script>"
    )
    html = html.replace('<script src="app.js"></script>', embed)

    out = web / "sanguozhi_explorer.html"
    out.write_text(html, encoding="utf-8")
    mb = out.stat().st_size / 1e6
    log.info("단일 HTML 생성: %s (%.2f MB)", out, mb)


if __name__ == "__main__":
    main()
