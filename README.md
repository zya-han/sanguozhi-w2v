# 정사 《삼국지》 Word2Vec · 단어 탐색기

陳壽의 정사 **《三國志》**(裴松之 注 포함, 전체 65卷)로 학습한 **Word2Vec 단어 임베딩**과,
이를 브라우저에서 바로 탐험하는 **웹 탐색기**입니다. 인물·관직·지명을 하나의 토큰으로 인식하도록
한자 텍스트를 다듬어 학습했기 때문에, `荀彧`과 가까운 단어로 `程昱`·`荀攸` 같은 인물이 떠오르고
`丞相`의 이웃으로 그 속관(屬官)들이 나옵니다.

> **라이브 데모** — [zyahan.blog/sanguozhi-word-explorer](https://zyahan.blog/sanguozhi-word-explorer) 에서 탐색기를 직접 사용해 볼 수 있습니다.

## 웹 탐색기

서버 없이 **브라우저 안에서만** 동작합니다. 학습된 벡터(L2 정규화, 약 2.2MB)를 통째로 내려받아
코사인 유사도를 자바스크립트로 계산하므로, 정적 호스팅이면 어디서나 돌아갑니다.

- **두 단어 비교** — 두 단어의 코사인 유사도와, 각 단어의 이웃 단어를 순위와 함께 나란히 비교합니다.
  두 목록에 공통으로 나오는 단어는 같은 색으로 이어 보여줍니다.
- **🕵️ 스파이 찾기** — 단어 여럿 중 가장 이질적인 하나를 골라냅니다(`doesnt_match`).
- **비슷한 단어 찾기** — 한 단어와 가까운 단어들을 순위·유사도순으로 봅니다.

한자(`荀彧`)와 한글 독음(`순욱`) 양쪽으로 검색할 수 있고, 입력창은 빈도순 자동완성을 제공합니다.
결과에는 각 단어의 **코퍼스 출현 횟수**가 함께 표시되며, 처음 쓰는 분을 위한 **해석 방법·한계 안내**도 들어 있습니다.
독음은 한국 한자음 규칙(두음법칙, 僕射→복야·祭酒→좨주 같은 관직 특수음, 조사 받침 처리)을 반영합니다.

검색 결과는 **URL로 공유**할 수 있습니다(딥링크 예: `?task=compare&cmpA=荀彧&cmpB=郭嘉`). 검색할 때마다 주소가
갱신되고, **결과 내보내기** 버튼으로 현재 링크를 클립보드에 복사합니다. SNS 공유 시 Open Graph 카드가 표시됩니다.

## 모델 개요

| 항목 | 값 |
|---|---|
| 코퍼스 | 三國志 전체 65卷 (魏書30·蜀書15·吳書20), 本文 + 裴松之 注 |
| 어휘 | 5,585개 (그중 다자 고유명사 약 1,800) |
| 차원 / 구조 | 100차원 · skip-gram |
| 학습 | gensim Word2Vec (window 5, min_count 3, negative 10, epochs 15, 고정 시드) |

특징:

- **글자 단위 토큰화**가 기본이라 허사(虛詞)의 분포가 보존됩니다. 통계적 분절·subword를 쓰지 않습니다.
- **고유명사만 병합** — 인명·관직·지명 가제티어로 결정론적 최장일치 병합(학습형 NER 없음).
  CBDB가 누락한 三國 인물(周瑜·呂蒙·荀彧 등)은 코퍼스 전기 도입부의 `{姓名}字{字}` 패턴에서 보충합니다.
- **인용 史書는 통째로 한 토큰** — 《魏略》·《江表傳》 등 괄호 포함 단일 토큰.
- **별명을 합치지 않음** — 諸葛亮(名)·孔明(字)·武侯(諡)는 각각 독립 토큰.
- **本文/裴注·문장 경계 보존** — 학습 윈도가 文/注나 문장 경계를 넘지 않습니다.
- **자형 통일** — 원문의 간·번체 혼용을 번체로 정리하되, 고전 의미가 갈리는 글자(于≠於·云≠雲 등)는 보호하고
  일본 신자체 같은 변이형(呉→吳 등)만 통합합니다.

## 파이썬에서 사용

```python
from gensim.models import Word2Vec
m = Word2Vec.load("models/w2v_sanguozhi.model")

m.wv.most_similar("丞相")   # → 倉曹·掾·令史 … (丞相府 속관)
m.wv.most_similar("劉備")   # → 關羽 …
m.wv.doesnt_match(["周瑜", "魯肅", "呂蒙", "諸葛亮"])   # → 諸葛亮
```

## 직접 빌드하기

```bash
conda create -p ./.conda python=3.12 -y
conda activate ./.conda
pip install -r requirements.txt
```

파이프라인은 순서대로 실행합니다. 모든 설정은 [`config.yaml`](config.yaml) 한 곳에서 조정합니다.

```bash
python src/01_fetch_corpus.py     # zh.wikisource 三國志 65卷 수집 → 本文/裴注 분리·문장분할
python src/02_normalize.py        # 번체 통일(보호 s2t) + 異體字 정규화
python src/03_build_gazetteer.py  # CBDB+CHGIS+코퍼스 字추출 → 가제티어
python src/04_tokenize.py         # 결정론적 최장일치 토큰화
python src/05_train_w2v.py        # Word2Vec 학습 → models/w2v_sanguozhi.model
python src/06_validate.py         # 정합성·커버리지 검증 → reports/validation.md
python src/07_export_web.py       # 모델 → 웹 탐색기용 데이터(web/data/)
python src/08_bundle_html.py      # 단일 HTML 한 장으로 묶기(web/sanguozhi_explorer.html)
```

> **CBDB 준비 (Stage 3 전제)** — `src/03`은 `vendor/cbdb_sqlite/*.sqlite3`를 자동 탐색합니다. 없으면
> [`cbdb-project/cbdb_sqlite`](https://github.com/cbdb-project/cbdb_sqlite)를 클론해 최신 SQLite를 받아 두세요.

### 웹 탐색기 빌드

```bash
python src/07_export_web.py        # 벡터·어휘·독음 추출 (먼저 1회)
cd web && python -m http.server    # http://localhost:8000 에서 미리보기
```

- 프런트엔드 소스(`web/index.html`·`style.css`·`app.js`)만 저장소에 포함됩니다.
  추출 데이터(`web/data/`)와 단일 HTML 번들은 모델 산출물이라 위 스크립트로 재생성합니다.
- **배포**: `web/` 폴더를 정적 호스팅에 올리면 됩니다(상대경로라 어느 하위 경로에서도 동작).
- **단일 파일 배포**: `src/08_bundle_html.py`가 CSS·JS·벡터(base64)를 한 파일에 담아
  `web/sanguozhi_explorer.html`로 묶습니다. 더블클릭(`file://`)만으로 서버 없이 열립니다.

## 데이터 출처 · 라이선스

| 데이터 | 출처 | 라이선스 |
|---|---|---|
| 원문 三國志 (전체 65卷) | 中文維基文庫 zh.wikisource `三國志/卷01–卷65` | CC BY-SA 4.0 |
| 인명·관직 | CBDB [`cbdb-project/cbdb_sqlite`](https://github.com/cbdb-project/cbdb_sqlite) | CC BY-NC-SA 4.0 |
| 지명 | CBDB ADDR_CODES (CHGIS 연계) / TGAZ | CHGIS (Harvard & Fudan) |

가제티어가 CBDB에서 파생되므로, 모델과 탐색기를 비롯한 파생물은 **CC BY-NC-SA 4.0**(비상업·출처표기·동일조건)을
따릅니다. 이용 시 위 출처를 함께 표기해 주세요.

## 제작

[주아](https://zyahan.blog) & Claude Code
