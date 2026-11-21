"""Three search tools: vector, keyword (BM25), and hybrid (both merged with RRF)."""
import re

from rank_bm25 import BM25Okapi

import llm
import store

_bm25 = {"n": -1, "index": None, "ids": [], "docs": [], "metas": []}


def tokenize(text):
    """Split snake_case and camelCase but keep the whole identifier too."""
    text = text or ""
    whole = [t.lower() for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text)]
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    parts = [t.lower() for t in re.findall(r"[A-Za-z]+|\d+", spaced)]
    return whole + parts


def _chunk(id_, doc, meta, score):
    return {"id": id_, "text": doc, "file": meta["file"], "start": meta["start"],
            "end": meta["end"], "score": float(score)}


def _load_bm25():
    col = store.collection()
    n = col.count()
    if n == 0:
        raise RuntimeError("The index is empty. Run: python index_repo.py <path-to-repo>")
    if _bm25["n"] != n:
        data = col.get(include=["documents", "metadatas"])
        _bm25.update(
            n=n, ids=data["ids"], docs=data["documents"], metas=data["metadatas"],
            index=BM25Okapi([tokenize(d) for d in data["documents"]]),
        )
    return _bm25


def vector_search(query, k):
    col = store.collection()
    if col.count() == 0:
        raise RuntimeError("The index is empty. Run: python index_repo.py <path-to-repo>")
    qv = llm.embed([query], "RETRIEVAL_QUERY")[0]
    res = col.query(query_embeddings=[qv], n_results=min(k, col.count()))
    return [
        _chunk(i, d, m, 1 - dist)
        for i, d, m, dist in zip(res["ids"][0], res["documents"][0],
                                 res["metadatas"][0], res["distances"][0])
    ]


def keyword_search(query, k):
    b = _load_bm25()
    toks = tokenize(query)
    if not toks:
        return []
    scores = b["index"].get_scores(toks)
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [_chunk(b["ids"][i], b["docs"][i], b["metas"][i], scores[i])
            for i in order if scores[i] > 0]


def hybrid_search(query, k):
    """Reciprocal Rank Fusion: score = sum of 1/(60 + rank) across both result lists."""
    fused, by_id = {}, {}
    for results in (vector_search(query, k * 2), keyword_search(query, k * 2)):
        for rank, c in enumerate(results):
            fused[c["id"]] = fused.get(c["id"], 0) + 1 / (60 + rank)
            by_id[c["id"]] = c
    top = sorted(fused, key=fused.get, reverse=True)[:k]
    return [dict(by_id[i], score=fused[i]) for i in top]


def retrieve(query, strategy, k):
    if strategy == "keyword":
        return keyword_search(query, k)
    if strategy == "vector":
        return vector_search(query, k)
    return hybrid_search(query, k)
