# 《三國志》 Word2Vec

陳壽 정사 **《三國志》**(裴松之 注 포함) 단일 코퍼스에 대한 **개체 인식 토큰화 + Word2Vec** 파이프라인.
글자 단위 토큰화를 기본으로, 가제티어 기반 **결정론적 최장일치**로 다자 고유명사만 병합한다.
설계 원칙·비목표는 [`sanguozhi_word2vec_spec.md`](sanguozhi_word2vec_spec.md) 참조.

## 핵심 설계 (요약)
- **코퍼스 = 전체 65卷**(魏書30·蜀書15·吳書20), zh.wikisource 正字 번체, 표점 포함.
- **기본 토큰 = 글자**. 비개체 텍스트는 통계 분절·subword 하지 않음(허사 분포 보존).
- **개체 병합 = 가제티어 결정론적 최장일치**만. 학습형 NER 없음.
  - 가제티어: CBDB(인명·관직)+CHGIS 지명 + **코퍼스 전기 도입부 "{姓名}字{字}" 결정론적 추출**
    (CBDB가 누락한 三國 인물 周瑜·呂蒙·荀彧 등을 빈출 姓 앵커로 확보).
- **《書名》= 괄호 포함 단일 토큰**: 《魏略》·《江表傳》 등 인용 史書를 통째로 한 토큰.
- **별명 비정규화**: 諸葛亮(名)·孔明(字)·武侯(諡)는 각각 독립 토큰.
- **단자 名 미병합**: 亮·操·備는 동형이의라 글자 토큰 유지.
- **本文(陳壽)/裴注(裴松之)** 플래그 보존, 표점 문장 단위 segment → 윈도가 文/注·문장 경계를 넘지 않음.
- **자형**: 正字 번체 → OpenCC 미적용(于/於·云/雲 등 고전 의미 구분 보존).

## 환경 구축
```bash
conda create -p ./.conda python=3.12 -y
conda activate ./.conda
pip install -r requirements.txt
```

## 실행 (순서대로)
```bash
conda activate ./.conda
python src/01_fetch_corpus.py     # zh.wikisource 三國志 65卷 → 本文/裴注 분리·문장분할 → segments.parquet
python src/02_normalize.py        # 正字 보존 정규화 → normalized.parquet
python src/03_build_gazetteer.py  # CBDB+CHGIS+코퍼스 字추출 + 보충시드 + sanity 필터 → gazetteer.tsv
python src/04_tokenize.py         # 결정론적 최장일치 토큰화(《書名》 통째 토큰) → corpus.jsonl
python src/05_train_w2v.py        # Word2Vec 학습 → models/w2v_sanguozhi.model, vocab.tsv
python src/06_validate.py         # 정합성·커버리지 검증 → reports/validation.md
python src/07_export_web.py       # 모델 → web/data/{vectors.bin,vocab.json,readings.json} (웹 탐색기용)
```
> Stage 1은 wikisource API로 65卷을 받아 `data/raw/wikisource/卷NN.wiki`에 캐시한다(재실행 시 캐시 사용).
모든 설정은 [`config.yaml`](config.yaml) 한 곳에서 조정(소스 URL·시대경계·하이퍼파라미터·시드).

### CBDB SQLite 준비 (Stage 3 전제)
`src/03`은 `vendor/cbdb_sqlite/*.sqlite3`를 자동 탐색한다. 없으면:
```bash
git clone --depth 1 https://github.com/cbdb-project/cbdb_sqlite.git vendor/cbdb_sqlite
cd vendor/cbdb_sqlite
curl -L -o latest.zip "https://huggingface.co/datasets/cbdb/cbdb-sqlite/resolve/main/latest.zip"
unzip -q latest.zip && rm latest.zip
```

## 사용 예
```python
from gensim.models import Word2Vec
m = Word2Vec.load("models/w2v_sanguozhi.model")
m.wv.most_similar("丞相")   # → 倉曹·掾·令史 … (丞相府 속관)
m.wv.most_similar("劉備")   # → 關羽 …
```

## 웹 탐색기 (`web/`)
브라우저에서 임베딩을 직접 탐색하는 **서버리스 정적 페이지**. 백엔드 없이 벡터(L2 정규화 float32, ~2.4MB)를
통째로 받아 JS로 코사인을 계산한다. 기능: ① 유사어 검색(topn 조절) ② 두 단어 비교 ③ 스파이 찾기(`doesnt_match`).
한자(`荀彧`)·한글 음(`순욱`) 겸용 검색.

```bash
python src/07_export_web.py        # 모델 → web/data/* 추출 (먼저 1회)
cd web && python -m http.server    # http://localhost:8000 에서 확인
```
- `src/07_export_web.py`는 추출 직후 JS식 내적 결과를 `most_similar`와 대조해 일치를 검증한다.
- `web/`(HTML·JS·CSS)만 추적, `web/data/`는 모델 추출물이라 `.gitignore`(재생성 가능).
- **배포**: `web/`를 정적 호스팅(예: `zyahan.blog` 하위 경로)에 업로드. 상대경로라 어느 경로에서도 동작.
  `.bin`에 gzip/brotli 권장. CBDB 파생물이므로 페이지는 **비상업·CC BY-NC-SA 4.0** 표기를 유지한다.

### 단일 HTML 한 장 (`src/08`)
외부 파일 없이 CSS·JS·벡터(base64 내장)까지 한 파일에 담아 `file://`로 더블클릭만으로 열린다(서버 불필요).
```bash
python src/08_bundle_html.py       # web/{index.html,style.css,app.js}+data → web/sanguozhi_explorer.html (~3.4MB)
```
`web/index.html`·`style.css`·`app.js`를 그대로 인라인하므로 멀티파일 버전과 코드가 1:1 동일하다.

## 데이터 소스 / 라이선스 (인용 필수)
| 데이터 | 소스 | 라이선스 |
|---|---|---|
| 원문 三國志 (전체 65卷) | 中文維基文庫 zh.wikisource `三國志/卷01–卷65` | **CC BY-SA 4.0** |
| 인명·관직 | CBDB `cbdb-project/cbdb_sqlite` | **CC BY-NC-SA 4.0** (비상업, 동일조건 공유) |
| 지명 | CBDB ADDR_CODES (= CHGIS 연계) / TGAZ | CHGIS (Harvard & Fudan) |

> 당초 정본으로 검토한 Kanripo `KR2a0012`는 카탈로그 표기(65卷)와 달리 **魏志 30卷만** 담겨 있어
> (蜀·吳書 누락) zh.wikisource 전체본으로 교체했다.

> CBDB는 **CC BY-NC-SA 4.0**이므로 파생물(가제티어)도 동일 조건. `data/`·`vendor/`·`models/`는
> `.gitignore` 처리되어 재배포되지 않으며, 코드와 provenance만 추적된다.

## 산출물
- `data/interim/segments.parquet`, `normalized.parquet` — 추출·정규화 결과(本文/裴注/考證 플래그)
- `data/gazetteer/gazetteer.tsv` — 최종 가제티어(surface, type)
- `data/tokenized/corpus.jsonl` — 토큰화 코퍼스(`{segment_id, kind, is_peizhu, juan, tokens}`)
- `models/w2v_sanguozhi.model`, `models/vocab.tsv` — 모델·어휘
- `reports/validation.md` — 검증 리포트(정합성 13/13 통과, 커버리지, 분포 예시)
