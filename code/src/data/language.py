"""언어 과제 데이터: Brown 코퍼스 품사(universal tagset 12종) 시퀀스.

과제: 직전 CONTEXT개 단어의 품사로 다음 단어를 예측한다.
  - task="tag12"  : 다음 단어의 품사 12종 분류
  - task="content": 다음 단어가 내용어(NOUN/VERB/ADJ/ADV)인지 기능어인지 이진 분류

입력은 직전 품사들의 one-hot을 이어붙인 벡터. 문장 시작 앞은 BOS 품사로 채운다.
분할은 문장 단위(같은 문장이 train/test에 걸치지 않음)로 하고, 분할 자체는 시드 고정.

캐시: data/brown_tags.npz (품사 id 시퀀스 + 문장 경계). nltk 코퍼스가 없어도 캐시만 있으면 동작한다.
"""
from pathlib import Path

import numpy as np

TAGS = ["NOUN", "VERB", ".", "ADP", "DET", "ADJ", "ADV", "PRON", "CONJ", "PRT", "NUM", "X"]
BOS = len(TAGS)            # 문장 시작 패딩용 가짜 품사
N_SYMBOLS = len(TAGS) + 1  # 입력 쪽 one-hot 크기 (BOS 포함)
CONTENT_TAGS = {"NOUN", "VERB", "ADJ", "ADV"}

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "brown_tags.npz"
SPLIT_SEED = 0  # 문장 분할 시드는 모든 실험에서 고정


def _build_cache():
    from nltk.corpus import brown  # 캐시가 없을 때만 필요
    tag_id = {t: i for i, t in enumerate(TAGS)}
    ids, starts = [], []
    for sent in brown.tagged_sents(tagset="universal"):
        starts.append(len(ids))
        ids.extend(tag_id[t] for _, t in sent)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, ids=np.array(ids, dtype=np.uint8), starts=np.array(starts, dtype=np.int64))


def load_sentences():
    """품사 id 시퀀스를 문장 리스트로 반환."""
    if not CACHE.exists():
        _build_cache()
    z = np.load(CACHE)
    ids, starts = z["ids"], z["starts"]
    ends = np.append(starts[1:], len(ids))
    return [ids[s:e] for s, e in zip(starts, ends)]


def _windows(sentences, context):
    """(이전 품사들, 다음 품사) 쌍을 만든다."""
    ctx, nxt = [], []
    for sent in sentences:
        padded = np.concatenate([np.full(context, BOS, dtype=np.int64), sent.astype(np.int64)])
        for i in range(len(sent)):
            ctx.append(padded[i:i + context])
            nxt.append(sent[i])
    return np.array(ctx), np.array(nxt)


def _one_hot(ctx):
    n, k = ctx.shape
    x = np.zeros((n, k * N_SYMBOLS), dtype=np.float32)
    rows = np.repeat(np.arange(n), k)
    cols = (np.arange(k) * N_SYMBOLS)[None, :].repeat(n, 0).ravel() + ctx.ravel()
    x[rows, cols] = 1.0
    return x


def make_task(task="content", context=3, n_train=20000, n_val=5000, n_test=5000, seed=0):
    """과제 데이터를 만든다.

    문장 분할(train/val/test = 80/10/10)은 SPLIT_SEED로 고정되고,
    각 분할 안에서 표본을 뽑는 것만 seed에 따라 달라진다.

    반환: dict(X_train, y_train, X_val, y_val, X_test, y_test, ctx_*, n_classes, task, context)
      - X_*: float32 one-hot (N, context * N_SYMBOLS)
      - y_*: int64 (N,)  — 1차원. (N,1) 형태를 쓰지 않는다 (브로드캐스팅 채점 버그 방지)
      - ctx_*: int64 (N, context) 원래 품사 id (n-gram 기준선용)
    """
    sents = load_sentences()
    order = np.random.RandomState(SPLIT_SEED).permutation(len(sents))
    a, b = int(len(sents) * 0.8), int(len(sents) * 0.9)
    parts = {"train": order[:a], "val": order[a:b], "test": order[b:]}
    sizes = {"train": n_train, "val": n_val, "test": n_test}
    rng = np.random.RandomState(seed)

    out = {"task": task, "context": context}
    for name, idx in parts.items():
        ctx, nxt = _windows([sents[i] for i in idx], context)
        n = sizes[name]
        if n is not None and n < len(nxt):
            pick = rng.choice(len(nxt), n, replace=False)
            ctx, nxt = ctx[pick], nxt[pick]
        if task == "content":
            content_ids = [TAGS.index(t) for t in CONTENT_TAGS]
            y = np.isin(nxt, content_ids).astype(np.int64)
        elif task == "tag12":
            y = nxt.astype(np.int64)
        else:
            raise ValueError(task)
        out[f"X_{name}"] = _one_hot(ctx)
        out[f"y_{name}"] = y
        out[f"ctx_{name}"] = ctx
    out["n_classes"] = 2 if task == "content" else len(TAGS)
    return out



# ---------------------------------------------------------------------------
# 어려운 과제 (효율 단계): 단어 정보 포함
# ---------------------------------------------------------------------------
WORD_CACHE = ROOT / "data" / "brown_words.npz"
WORD_VOCAB_MAX = 5000      # 캐시에 저장하는 빈도 상위 단어 수 (그 밖은 OOV = WORD_VOCAB_MAX)


def _build_word_cache():
    from collections import Counter
    from nltk.corpus import brown
    words = [w.lower() for sent in brown.tagged_sents(tagset="universal") for w, _ in sent]
    vocab = [w for w, _ in Counter(words).most_common(WORD_VOCAB_MAX)]
    wid = {w: i for i, w in enumerate(vocab)}
    ids = np.array([wid.get(w, WORD_VOCAB_MAX) for w in words], dtype=np.uint16)
    np.savez_compressed(WORD_CACHE, ids=ids, vocab=np.array(vocab))


def load_word_ids():
    """품사 캐시와 같은 순서의 단어 id 시퀀스 (빈도 순위, OOV = WORD_VOCAB_MAX)."""
    if not WORD_CACHE.exists():
        _build_word_cache()
    return np.load(WORD_CACHE)["ids"].astype(np.int64)


def make_word_task(task="wtag", context=4, vocab=300, n_train=50000, n_val=5000, n_test=5000, seed=0, n_words=50):
    """단어 + 품사 문맥으로 다음 품사(12종)를 예측하는 과제.

    입력: 직전 context개 위치마다 [단어 one-hot (빈도 상위 vocab개, 그 밖은 0) | 품사 one-hot (13)].
      → 입력 차원 = context × (vocab + 13). 품사 n-gram 표로는 단어 조합을 다 외울 수 없어서
        기존 tag12(입력 39차원, n-gram 표 = 천장)보다 천장이 높고 모델 크기가 의미를 갖는다.
    분할: make_task와 같은 문장 단위 80/10/10 (SPLIT_SEED), 표본 추출만 seed에 따라 다름.
    반환 키는 make_task와 같다. ctx_*는 (N, 2*context): [단어 id들 | 품사 id들] (OOV는 vocab).
    task="nextword": 같은 입력으로 다음 단어 자체를 맞힌다. 다음 단어가 빈도 상위 n_words개 안에 드는 위치만 쓰고,
      라벨 = 그 단어의 빈도 순위 (0 ~ n_words−1). 품사보다 훨씬 세밀해서 모델 크기가 필요할 것으로 기대.
    """
    assert task in ("wtag", "nextword")
    tag_sents = load_sentences()
    raw = load_word_ids()
    wids = np.where(raw < vocab, raw, vocab)
    lens = np.array([len(t) for t in tag_sents])
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    word_sents = [wids[s:s + n] for s, n in zip(starts, lens)]
    raw_sents = [raw[s:s + n] for s, n in zip(starts, lens)]
    order = np.random.RandomState(SPLIT_SEED).permutation(len(tag_sents))
    a, b = int(len(tag_sents) * 0.8), int(len(tag_sents) * 0.9)
    parts = {"train": order[:a], "val": order[a:b], "test": order[b:]}
    sizes = {"train": n_train, "val": n_val, "test": n_test}
    rng = np.random.RandomState(seed)
    per = vocab + N_SYMBOLS
    out = {"task": task, "context": context, "vocab": vocab}
    for name, idx in parts.items():
        tctx, nxt = _windows([tag_sents[i] for i in idx], context)
        wctx, _ = _windows([word_sents[i] for i in idx], context)
        wctx = np.where(tctx == BOS, vocab, wctx)          # 문장 시작 패딩 자리는 단어 없음
        nxt_tag = nxt
        if task == "nextword":
            nxt = np.concatenate([raw_sents[i] for i in idx]).astype(np.int64)
            keep = nxt < n_words
            tctx, wctx, nxt, nxt_tag = tctx[keep], wctx[keep], nxt[keep], nxt_tag[keep]
        n = sizes[name]
        if n is not None and n < len(nxt):
            pick = rng.choice(len(nxt), n, replace=False)
            tctx, wctx, nxt, nxt_tag = tctx[pick], wctx[pick], nxt[pick], nxt_tag[pick]
        N = len(nxt)
        X = np.zeros((N, context * per), dtype=np.float32)
        rows = np.arange(N)
        for k in range(context):
            known = wctx[:, k] < vocab
            X[rows[known], k * per + wctx[known, k]] = 1.0
            X[rows, k * per + vocab + tctx[:, k]] = 1.0
        out[f"X_{name}"] = X
        out[f"y_{name}"] = nxt.astype(np.int64)
        out[f"ctx_{name}"] = np.concatenate([wctx, tctx], 1)
        out[f"tag_{name}"] = nxt_tag.astype(np.int64)          # 다음 요소의 품사 (wtag에서는 y와 같음)
    out["n_classes"] = len(TAGS) if task == "wtag" else n_words
    return out


if __name__ == "__main__":
    for task in ["content", "tag12"]:
        d = make_task(task)
        counts = np.bincount(d["y_train"], minlength=d["n_classes"])
        print(task, d["X_train"].shape, "train class ratio:", np.round(counts / counts.sum(), 3))
