"""Stage 5 — Word2Vec 학습.

명세 §5:
  - gensim.models.Word2Vec. 기본값 config 노출: sg=1, vector_size=200, window=5,
    min_count=3, negative=10, epochs=15, seed=고정, workers=1(재현성).
  - 학습 입력 = 토큰 시퀀스 리스트. **윈도가 文/注·문장 경계를 넘지 않도록 segment 단위**로 학습.
    (각 segment = 한 문장(sentence). gensim은 sentence 내에서만 윈도를 잡으므로 경계 보존.)
  - 裴注 학습 포함 여부 = config.corpus.include_peizhu_in_training.

출력: models/w2v_sanguozhi.model, models/vocab.tsv (token, freq, is_entity).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO_ROOT, ensure_dir, get_logger, load_config, resolve, set_seed  # noqa: E402

log = get_logger("05_train_w2v")


def _read_corpus(path: Path, include_peizhu: bool, label: str = ""):
    """단일 corpus.jsonl → 토큰 시퀀스 리스트 (注는 include_peizhu에 따라 포함/제외)."""
    sents, n_main, n_pei = [], 0, 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["is_peizhu"]:
                if not include_peizhu:
                    continue
                n_pei += 1
            else:
                n_main += 1
            sents.append(rec["tokens"])
    log.info("  %s文 %d (本文 %d, 注 %d, include_peizhu=%s)",
             f"{label}: " if label else "", len(sents), n_main, n_pei, include_peizhu)
    return sents


def load_sentences(cfg: dict):
    """corpus.jsonl → segment 단위 토큰 시퀀스 리스트. 文/注 경계는 segment 분리로 보존.

    통합 모델(config.corpus.combine): 여러 사서의 corpus.jsonl 을 각자의 include_peizhu
    로 풀링(segment_id 접두로 출처 추적 가능). 일반 모델: 단일 paths.tokenized/corpus.jsonl.
    """
    combine = cfg["corpus"].get("combine")
    if combine:
        log.info("통합 학습: %d개 코퍼스 풀링", len(combine))
        sents = []
        for src in combine:
            p = Path(src["tokenized"])
            if not p.is_absolute():
                p = REPO_ROOT / p
            sents.extend(_read_corpus(p / "corpus.jsonl",
                                      bool(src.get("include_peizhu", True)),
                                      label=src.get("id", p.name)))
        log.info("통합 학습 문장(segment) 합계: %d", len(sents))
        return sents

    include_peizhu = cfg["corpus"].get("include_peizhu_in_training", True)
    return _read_corpus(resolve(cfg, "tokenized") / "corpus.jsonl", include_peizhu)


def main():
    cfg = load_config()
    set_seed(cfg)
    from gensim.models import Word2Vec

    w = cfg["word2vec"]
    sentences = load_sentences(cfg)

    log.info("Word2Vec 학습 시작: sg=%s size=%s window=%s min_count=%s neg=%s epochs=%s workers=%s",
             w["sg"], w["vector_size"], w["window"], w["min_count"],
             w["negative"], w["epochs"], w["workers"])

    model = Word2Vec(
        sentences=sentences,
        sg=int(w["sg"]),
        vector_size=int(w["vector_size"]),
        window=int(w["window"]),
        min_count=int(w["min_count"]),
        negative=int(w["negative"]),
        epochs=int(w["epochs"]),
        seed=int(w["seed"]),
        workers=int(w["workers"]),
    )

    models_dir = ensure_dir(resolve(cfg, "models"))
    model_path = models_dir / w.get("model_name", "w2v_sanguozhi.model")
    model.save(str(model_path))

    # vocab.tsv (token, freq, is_entity)
    rows = []
    for tok in model.wv.index_to_key:
        rows.append((tok, model.wv.get_vecattr(tok, "count"), int(len(tok) > 1)))
    vocab = pd.DataFrame(rows, columns=["token", "freq", "is_entity"])
    vocab.to_csv(models_dir / "vocab.tsv", sep="\t", index=False)

    # 학습 메타
    meta = {
        "vocab_size": len(model.wv),
        "entity_vocab": int(vocab["is_entity"].sum()),
        "total_corpus_tokens": int(model.corpus_total_words),
        "params": {k: w[k] for k in w},
        "include_peizhu": cfg["corpus"].get("include_peizhu_in_training", True),
    }
    (models_dir / "train_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    log.info("모델 저장: %s | vocab %d (개체 %d) | 코퍼스 %d토큰",
             model_path, len(model.wv), int(vocab["is_entity"].sum()),
             int(model.corpus_total_words))

    # 스모크 테스트
    for probe in ["曹操", "諸葛亮", "丞相"]:
        if probe in model.wv:
            sims = ", ".join(f"{t}({s:.2f})" for t, s in model.wv.most_similar(probe, topn=5))
            log.info("  most_similar(%s): %s", probe, sims)


if __name__ == "__main__":
    main()
