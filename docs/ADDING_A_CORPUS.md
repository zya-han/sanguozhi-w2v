# 새 코퍼스 추가 가이드 — 史記·漢書·後漢書 (그리고 그 밖의 正史)

이 문서는 현재 《三國志》용으로 완성된 파이프라인을 **다른 사서에도 그대로 적용**하기 위한
핸드오프다. 코드(`src/`)와 CBDB(`vendor/`)는 공유하고, **코퍼스별 config + 데이터/모델
디렉터리**로 분리한다. 새 세션에서 이 문서만 보고 바로 시작할 수 있도록 작성했다.

> 설계 철학·확정 결정은 [`sanguozhi_word2vec_spec.md`](../sanguozhi_word2vec_spec.md), 현재
> 동작은 [`README.md`](../README.md) 참조. 아래 "재사용" 단계는 코퍼스 불문 그대로 동작한다.

---

## 1. 무엇이 공유되고 무엇이 코퍼스별인가

| 공유 (수정 없음) | 코퍼스별 (config·데이터) |
|---|---|
| `src/02·04·05·06·07·08` 전부 | `config/<id>.yaml` |
| `common.make_script_normalizer` (s2t+보호+異體字) | `data/<id>/…`, `models/<id>/`, `reports/<id>/` |
| Stage 3의 字-추출(`extract_per_from_zi`, 빈출 姓 앵커) | 가제티어 시드 `data/<id>/gazetteer/*.tsv` |
| 《書名》 토큰·부호 경계/토큰·단자 名 미병합 | CBDB 시대 필터(dynasty_codes·연도) |
| `vendor/cbdb_sqlite/*.sqlite3` (CBDB) | book_title, 年號·官職·地名·人名·user_dict 시드 |

`opencc_protect`(고전 충돌 글자 于云后里征…)는 고전한문 공통이라 그대로 재사용한다.

---

## 2. wikisource 사서별 특성 (실측 완료)

| 사서 | 페이지 | 卷 수 | 注 | 비고 |
|---|---|---|---|---|
| 三國志 | `三國志/卷01`–`卷65` | 65 | 裴注 `{{*|}}` | (현행) |
| **史記** | `史記/卷001`–`卷130` (3자리) | 130 | **없음**(三家注 미수록) | 本文만 |
| **漢書** | `漢書/卷001`–`卷100` | 100 | **없음**(顏注 미수록) | `卷NNN上/下`는 리다이렉트 |
| **後漢書** | `後漢書/卷1`–`卷120` (zero-pad 없음) | ~99 | **李賢注 `{{*|}}`**(卷마다 혼재: 卷1=266, 卷66=0) | 紀傳+志 |

→ **卷 네이밍이 제각각이라 range로 구성 불가. `allpages` API로 실페이지를 발견한다**(§4-C).
→ 注는 `{{*|}}` 유무로 자동 처리됨(있으면 peizhu 분리, 없으면 main만). 코드 변경 불필요.
→ 추가 템플릿 `{{header2|…}}`·`{{wikipedia|…}}` 제거는 기존 균형 `{{}}` 파서가 처리(검증할 것).
→ 본문은 正字 번체 + 표점 + 簡體 오염 → 기존 정규화(s2t+보호+異體字) 그대로.

---

## 3. CBDB 시대 필터 (DYNASTIES 실측값)

```
1=漢前(-1100~-206)  61=贏秦(-221~-206)  29=西漢(-206~9)  83=漢(-206~220)
46=新(9~25)  25=東漢(25~220)  2=秦漢(-221~220)  3=三國  23/82=晉
```

| 사서 | year_start | year_end | dynasty_codes |
|---|---|---|---|
| 後漢書 | 25 | 220 | `[25, 83]` |
| 漢書 (西漢+新) | -206 | 25 | `[29, 83, 46, 2]` |
| 史記 (黃帝~武帝) | -2100 | -90 | `[1, 61, 29, 2, 83]` |

> CBDB는 前漢 이전 인물 커버리지가 매우 빈약하다(三國 사망자도 62명뿐이었음). 史記·漢書는
> **字-추출 + 큐레이션 시드가 주력**이고 CBDB는 보조다. `require_corpus_occurrence: true`가
> 코퍼스 등장분만 남기므로 dynasty_codes를 다소 넓게 잡아도 노이즈는 제한적이다.

---

## 4. 코드 변경 (공유 `src/`, 1회만)

> **현재 상태**: 아래 **A·B 및 디렉터리 재편은 三國志 기준으로 이미 적용됨**.
> `config/sanguozhi.yaml`(기본), `data/sanguozhi/…`·`models/sanguozhi/`·`reports/sanguozhi/`,
> `paths.web: web`. 새 코퍼스는 **C(Stage 1 allpages)** 적용 + config 생성 + 큐레이션만 하면 됨.
> (三國志 config는 아직 `page_prefix/juan_count`로 동작 — C 적용 시 `book_title`로 전환.)

### A. CORPUS_CONFIG 디스패치 — `src/common.py` ✅적용됨
기본값은 `config/sanguozhi.yaml`. `CORPUS_CONFIG` 환경변수로 코퍼스 전환.
`load_config`가 환경변수를 읽게 한다(미설정 시 루트 `config.yaml` = 三國志 폴백, 하위호환):
```python
def load_config(path=None):
    chosen = path or os.environ.get("CORPUS_CONFIG")
    cfg_path = Path(chosen) if chosen else REPO_ROOT / "config.yaml"
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)
```
실행: `CORPUS_CONFIG=config/houhanshu.yaml python src/01_fetch_corpus.py` … 08까지 순차.
편의 래퍼(선택) `run.sh`:
```bash
#!/usr/bin/env bash
export CORPUS_CONFIG="config/$1.yaml"
shift; for s in "$@"; do python "src/${s}"*.py; done
# 예: ./run.sh houhanshu 01_ 02_ 03_ 04_ 05_ 06_
```

### B. 모델 파일명·웹 경로 파라미터화 — `config` + `src/05·06·07·08` ✅적용됨
`word2vec.model_name`(05 저장·06·07 로드)과 `paths.web`(07·08 출력 = `web/data`) config화 완료.
새 코퍼스는 config에 `model_name: w2v_<id>.model`, `paths.web: web/<id>`만 지정.

### C. Stage 1 페이지 발견 일반화 — `src/01_fetch_corpus.py`
`range(1, juan_count+1)` + zero-pad 구성을 **allpages 발견**으로 교체:
```python
def list_juan_pages(cfg):
    book = cfg["corpus"]["book_title"]
    api = cfg["corpus"]["wikisource_api"]
    pages, cont = [], None
    while True:
        params = {"action": "query", "list": "allpages",
                  "apprefix": f"{book}/卷", "apnamespace": 0,
                  "apfilterredir": "nonredirects", "aplimit": "max", "format": "json"}
        if cont: params["apcontinue"] = cont
        r = requests.get(api, params=params,
                         headers={"User-Agent": "han-w2v-research/1.0"}, timeout=30).json()
        pages += [p["title"] for p in r["query"]["allpages"]]
        cont = r.get("continue", {}).get("apcontinue")
        if not cont: break
    # 자연 정렬: 卷 뒤 숫자 + 上中下
    import regex as re
    ord_ = {"上": 0, "中": 1, "下": 2}
    def key(t):
        m = re.search(r"卷(\d+)([上中下])?", t)
        return (int(m.group(1)) if m else 9999, ord_.get(m.group(2) if m else None, -1))
    return sorted(pages, key=key)
```
- `fetch_juan_wikitext`는 페이지 **제목**으로 받게(개별 fetch에 `redirects=1` 유지, 캐시
  파일명은 제목 sanitize). 기존 429 백오프 보존.
- `build_segments`는 발견된 페이지를 순회하며 卷 id(`卷NN` 또는 제목)로 segment_id 생성.
- `shu_of()`(魏/蜀/吳)는 三國志 전용 → 일반화: config `shu_map` 없으면 `shu="本"`.
  (원하면 卷번호 구간→志名 매핑을 config로: 史記 本紀/世家/列傳, 漢書 紀/志/傳.)
- config에서 `page_prefix·juan_count·juan_zero_pad` 제거, `book_title` 추가.

### D. 그대로 재사용 (변경 없음)
Stage 2(정규화)·4(토큰화)·5(학습)·6(검증)·7·8(웹). 단 06의 검증 센티넬(`must_single`에
諸葛亮·周瑜 등 하드코딩)은 사서별로 다르므로 **config화하거나 사서별 리스트로 교체**(§7).

---

## 5. 코퍼스별 config 템플릿

`config/houhanshu.yaml` 예시(다른 사서는 값만 교체). 三國志 `config.yaml`을 복사해 아래만 수정:

```yaml
seed: 42
paths:
  vendor:    vendor                 # 공유
  raw:       data/houhanshu/raw
  interim:   data/houhanshu/interim
  gazetteer: data/houhanshu/gazetteer
  tokenized: data/houhanshu/tokenized
  models:    models/houhanshu
  reports:   reports/houhanshu
corpus:
  source:         "wikisource"
  wikisource_api: "https://zh.wikisource.org/w/api.php"
  book_title:     "後漢書"           # ← 사서명 (allpages 접두)
  sentence_close: "。！？」』"
  sentence_open:  "「『"
  include_peizhu_in_training: true  # 後漢書만 의미(注 있음). 史記·漢書는 注 없어 무관.
normalize:
  corpus_opencc:  "s2t"
  opencc_protect: "于云后里征咸并辟游范干余谷系凶沈表面松制致卜丑"   # 공통
  variant_map:    "data/houhanshu/gazetteer/variant_map.tsv"        # 코퍼스별 재생성
gazetteer:
  year_start:   25                  # ← §3 표
  year_end:     220
  dynasty_codes: [25, 83]           # ← §3 표
  min_surface_len: 2
  extract_names_from_zi: true
  surname_min_persons: 20
  require_corpus_occurrence: true
  stoplist:        "data/houhanshu/gazetteer/stoplist.txt"
  seed_supplement: "data/houhanshu/gazetteer/seed_supplement.tsv"
  user_dict:       "data/houhanshu/gazetteer/user_dict.tsv"
  opencc_config:   null
tokenize:
  punctuation_as_token: true
word2vec:
  sg: 1
  vector_size: 100
  window: 5
  min_count: 3
  negative: 10
  epochs: 15
  workers: 1
  seed: 42
  model_name: "w2v_houhanshu.model"
```

### 가제티어 시드 준비 (코퍼스별 `data/<id>/gazetteer/`)
- `stoplist.txt`: 三國志 것 복사(`data/gazetteer/stoplist.txt`) — 비개체 보통명사는 공통.
- `variant_map.tsv`: **사서마다 재생성**(JPVariants 역방향 + 빈도 다수형 + 블랙리스트
  才/御/予/弁/余). 三國志 생성 스니펫(아래) 재사용:
  ```python
  # opencc JPVariants 역방향에서 코퍼스 등장 異體字만, 다수형으로, 블랙리스트 제외
  # (전체 스니펫은 git log의 "Unify 異體字" 커밋 또는 기존 variant_map.tsv 헤더 참조)
  ```
- `seed_supplement.tsv`: §6 starter(年號·官職·地名·人名) + 반복 큐레이션.
- `user_dict.tsv`: 사서별 호칭어.

> `.gitignore`: `data/`·`models/`가 와일드카드로 이미 제외됨. 코퍼스별 `seed_supplement`·
> `user_dict`·`gazetteer.tsv`도 동일 정책(미추적). `stoplist`·`variant_map`은 추적 원하면
> 패턴 조정.

---

## 6. 큐레이션 starter — 年號(ERA) (복붙용, type=ERA)

가장 확실한 보강. `seed_supplement.tsv`에 `표면형⇥ERA⇥메모`로 추가. 코퍼스 출현 필터가 실등장분만 채택.

**後漢書** (三國志 seed의 後漢 연호 그대로 복사):
```
建武 中元 永平 建初 元和 章和 永元 元興 延平 永初 元初 永寧 建光 延光 永建 陽嘉 永和
漢安 建康 永嘉 本初 建和 和平 元嘉 永興 永壽 延熹 永康 建寧 熹平 光和 中平 初平 興平 建安
```
**漢書** (西漢 + 新):
```
建元 元光 元朔 元狩 元鼎 元封 太初 天漢 太始 征和 後元 始元 元鳳 元平 本始 地節 元康 神爵
五鳳 甘露 黃龍 初元 永光 建昭 竟寧 建始 河平 陽朔 鴻嘉 永始 元延 綏和 建平 元壽 元始 居攝
初始 始建國 天鳳 地皇
```
**史記** (武帝 이후만; 그 이전은 年號 없음):
```
建元 元光 元朔 元狩 元鼎 元封 太初
```

### 官職·地名·人名 (베이스 재사용 + 반복 큐레이션)
- **官職**: 三國志 office 시드 다수가 漢代 공통(三公·九卿·太守·刺史·郎官·尚書 등) → 복사 베이스.
  사서별 추가는 위음성(false negative) 보며 반복(三國志 작업 방식과 동일).
- **地名**: 郡縣이 시대마다 다름 — 三國志 州·郡 시드는 부분만 유효. 사서별 州·郡을 새로:
  史記=戰國 七雄·秦36郡, 漢書=西漢 郡國. 코퍼스 출현 필터가 스코핑.
- **人名(보충)**: 史記 黃帝·堯·舜·禹·項羽·劉邦·韓信·張良·蕭何…; 漢書 高祖·文帝·武帝·霍光·
  衛青·霍去病·董仲舒·司馬遷…; 後漢書 光武帝·班超·班固·馬援·鄧禹·竇憲…
- **user_dict(호칭)**: 史記 寡人·大王·足下 다수; 漢書·後漢書 陛下·主上 등.

---

## 7. 실행 & 검증 (사서별)

```bash
conda activate ./.conda
export CORPUS_CONFIG=config/houhanshu.yaml
for s in 01 02 03 04 05 06; do python src/${s}_*.py; done   # 웹: 07 08
```
1. **Stage 1 卷 수**: 史記 130, 漢書 100, 後漢書 ~99 발견되는지.
2. **정규화**: 簡體 잔존 0(`国书刘` 등 검색). 《書名》 이중(번/간) 통합.
3. **센티넬 개체** 단일 토큰 + 이웃 일관성:
   - 史記: 項羽·劉邦·黃帝 / 漢書: 高祖·霍光·武帝 / 後漢書: 光武帝·班超·馬援.
   - 年號 토큰끼리 군집(漢書: 元狩→元鼎·元朔), 단자 名 미병합 유지.
4. **검증 리포트**: `reports/<id>/validation.md` 통과. 06의 `must_single` 센티넬을 사서별로
   교체(諸葛亮·周瑜 → 項羽·劉邦 등). config `validate.sentinels`로 외부화 권장.

---

## 8. 새 세션에서 verify할 잔여 미지수
- 漢書/後漢書 `卷NNN上/下` 리다이렉트가 `apfilterredir=nonredirects`로 **중복 없이 단일 본문**
  되는지(allpages가 실내용 페이지만 주는지 확인. 漢書 卷100은 上/下가 본문일 수도 → 점검).
- 史記·漢書 일부 卷의 표점 완성도(오래된 전사는 표점 희소 → 문장 세그먼트 과대 가능).
- 後漢書 志(卷91~120, 司馬彪 續漢書 + 劉昭/李賢注) 포함 여부·일관성.
- 각 사서 첫 卷의 `{{header2}}`·`{{wikipedia}}`·기타 템플릿이 깨끗이 제거되는지(toc/저자 누출 점검).
```
