# 《三國志》 Word2Vec

陳壽 정사 **《三國志》**(裴松之 注 포함) 단일 코퍼스에 대한 **개체 인식 토큰화 + Word2Vec** 파이프라인.
글자 단위 토큰화를 기본으로, 가제티어 기반 **결정론적 최장일치**로 다자 고유명사만 병합한다.
설계 원칙·비목표는 [`sanguozhi_word2vec_spec.md`](sanguozhi_word2vec_spec.md) 참조.

## 핵심 설계 (요약)
- **기본 토큰 = 글자**. 비개체 텍스트는 통계 분절·subword 하지 않음(허사 분포 보존).
- **개체 병합 = 가제티어 결정론적 최장일치**만. 학습형 NER 없음.
- **별명 비정규화**: 諸葛亮(名)·孔明(字)·武侯(諡)는 각각 독립 토큰.
- **단자 名 미병합**: 亮·操·備는 동형이의라 글자 토큰 유지.
- **本文(陳壽)/裴注(裴松之)** 플래그 보존, 윈도가 文/注·segment 경계를 넘지 않음.
- **자형**: 四庫 WYG본은 正字 번체 → OpenCC 미적용(于/於·云/雲 등 고전 의미 구분 보존).

## 환경 구축
```bash
conda create -p ./.conda python=3.12 -y
conda activate ./.conda
pip install -r requirements.txt
```

## 실행 (순서대로)
```bash
conda activate ./.conda
python src/01_fetch_corpus.py     # Kanripo KR2a0012 클론 → 本文/裴注 분리 → segments.parquet
python src/02_normalize.py        # 正字 보존 정규화 → normalized.parquet
python src/03_build_gazetteer.py  # CBDB(PER/OFI/LOC) + 보충시드 + sanity 필터 → gazetteer.tsv
python src/04_tokenize.py         # 결정론적 최장일치 토큰화 → corpus.jsonl
python src/05_train_w2v.py        # Word2Vec 학습 → models/w2v_sanguozhi.model, vocab.tsv
python src/06_validate.py         # 정합성·커버리지 검증 → reports/validation.md
```
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

## 데이터 소스 / 라이선스 (인용 필수)
| 데이터 | 소스 | 라이선스 |
|---|---|---|
| 원문 三國志 | Kanseki Repository `kanripo/KR2a0012` (文淵閣四庫全書本) | Kanripo 이용약관 (https://www.kanripo.org) |
| 인명·관직 | CBDB `cbdb-project/cbdb_sqlite` | **CC BY-NC-SA 4.0** (비상업, 동일조건 공유) |
| 지명 | CBDB ADDR_CODES (= CHGIS 연계) / TGAZ | CHGIS (Harvard & Fudan) |

> CBDB는 **CC BY-NC-SA 4.0**이므로 파생물(가제티어)도 동일 조건. `data/`·`vendor/`·`models/`는
> `.gitignore` 처리되어 재배포되지 않으며, 코드와 provenance만 추적된다.

## 산출물
- `data/interim/segments.parquet`, `normalized.parquet` — 추출·정규화 결과(本文/裴注/考證 플래그)
- `data/gazetteer/gazetteer.tsv` — 최종 가제티어(surface, type)
- `data/tokenized/corpus.jsonl` — 토큰화 코퍼스(`{segment_id, kind, is_peizhu, juan, tokens}`)
- `models/w2v_sanguozhi.model`, `models/vocab.tsv` — 모델·어휘
- `reports/validation.md` — 검증 리포트(정합성 13/13 통과, 커버리지, 분포 예시)
