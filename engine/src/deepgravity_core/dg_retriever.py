#!/usr/bin/env python3
"""dg_retriever — the mechanical vector-retrieval pass for DeepGravity.

License: Apache-2.0. Depends only on numpy (BSD-3-Clause) and, when
available, faiss-cpu (MIT) for the index backend.

What this is: the first, mechanical half of retrieval. Given a query, it
embeds the query, finds the most similar stored vectors, and returns the
middle similarity band — nothing more. It is a sieve, not a judge.

What this is not: it does not decide relevance. Relevance judgment is the
cognitive eval, a separate step that reads the hits this module returns
and decides what actually matters. This module never stores plaintext
(it keeps item ids, tags, and vectors only — the content lives with the
caller, looked up by item id), invents no content, and makes no network
calls.

The band: John's stated band from the Oct 28 pipeline, unverified —
keep similarities strictly between BAND_LOW (0.45) and BAND_HIGH (0.75).
Near-identical echoes above the band are likely duplicates; distant noise
below it is likely irrelevant. The edges are parameters, not canon:
tune them, don't hard-code a new truth.

The embedder is injected, never imported. The real embedder
(Qwen3-Embedding-0.6B, Apache 2.0) is NOT on the box yet — a future
acquisition. In production the injected function calls the local
embedding daemon; in tests it is a fake returning fixed vectors.

The index has two backends. FAISS (faiss-cpu, MIT) when the faiss
package imports, numpy brute-force cosine otherwise. Both backends must
agree on the same inputs; the tests check that.

CCF tags: retrieved items carry tags. A cloud reader must NEVER receive
an NT-tagged item — NT means never leaves the device. NK is withheld
from cloud too, matching dg_turns.render_log: NK was never to be
retained, so it must never travel either. An unknown reader kind fails
closed to cloud. CO passes through here; conversation scoping is the
gate's job, not this module's.
"""

from __future__ import annotations

import math

import numpy as np

try:
    import faiss  # MIT, faiss-cpu — the fast index backend, when installed.
    FAISS_AVAILABLE = True
except ImportError:  # pragma: no cover - the numpy fallback path
    faiss = None
    FAISS_AVAILABLE = False

# John's stated band (Oct 28 pipeline), unverified. Strict inequalities:
# the edges themselves are out. Tunable per Retriever; these are defaults.
BAND_LOW = 0.45
BAND_HIGH = 0.75

READER_LOCAL = "local"
READER_CLOUD = "cloud"

# Tags a cloud reader must never receive (same rule as dg_turns.render_log).
_CLOUD_WITHHELD_TAGS = frozenset({"NT", "NK"})


def cosine(a, b) -> float:
    """Cosine similarity between two vectors. 0.0 on degenerate input."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))


def _normalize(vector) -> np.ndarray:
    """L2-normalize a vector. A zero vector stays zero (cosine 0.0)."""
    arr = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return arr
    return arr / norm


def in_band(similarity: float, low: float = BAND_LOW, high: float = BAND_HIGH) -> bool:
    """Strict band check: low < similarity < high. The edges are out."""
    return low < similarity < high


def band_filter(scored, low: float = BAND_LOW, high: float = BAND_HIGH):
    """Keep the (index, similarity) pairs whose similarity is in the band."""
    return [(i, s) for (i, s) in scored if in_band(s, low, high)]


class VectorIndex:
    """A small vector index with two backends: FAISS when available, else
    numpy brute-force cosine. Both backends store L2-normalized vectors,
    so inner product equals cosine similarity, and both must agree on the
    same inputs (see tests/test_retriever.py)."""

    def __init__(self, backend: str = "auto"):
        if backend == "auto":
            backend = "faiss" if FAISS_AVAILABLE else "numpy"
        if backend == "faiss" and not FAISS_AVAILABLE:
            raise RuntimeError("faiss backend requested but faiss is not installed")
        if backend not in ("faiss", "numpy"):
            raise ValueError(f"unknown backend: {backend!r}")
        self.backend = backend
        self.dim = None
        self._vectors = []  # normalized float64 rows, in add order
        self._faiss_index = None

    def __len__(self) -> int:
        return len(self._vectors)

    def add(self, vector) -> int:
        """Add one vector. Returns its position. First add sets the dimension."""
        arr = _normalize(vector)
        if self.dim is None:
            if arr.size == 0:
                raise ValueError("cannot add an empty vector")
            self.dim = int(arr.size)
            if self.backend == "faiss":
                self._faiss_index = faiss.IndexFlatIP(self.dim)
        if int(arr.size) != self.dim:
            raise ValueError(f"dimension mismatch: got {arr.size}, index is {self.dim}")
        self._vectors.append(arr)
        if self.backend == "faiss":
            self._faiss_index.add(
                np.ascontiguousarray(arr, dtype=np.float32).reshape(1, -1))
        return len(self._vectors) - 1

    def search(self, query_vector, k: int) -> list:
        """Top-k (index, cosine similarity), best first. Empty list on an
        empty index or k <= 0. FAISS returns -1 slots when k exceeds the
        count; those are dropped."""
        if not self._vectors or k <= 0:
            return []
        q = _normalize(query_vector)
        if int(q.size) != self.dim:
            raise ValueError(f"dimension mismatch: got {q.size}, index is {self.dim}")
        k = min(k, len(self._vectors))
        if self.backend == "faiss":
            q32 = np.ascontiguousarray(q, dtype=np.float32).reshape(1, -1)
            distances, indices = self._faiss_index.search(q32, k)
            return [(int(i), float(d)) for i, d in zip(indices[0], distances[0])
                    if i != -1]
        mat = np.vstack(self._vectors)
        scores = mat @ q
        order = np.argsort(-scores, kind="stable")[:k]
        return [(int(i), float(scores[i])) for i in order]


class Retriever:
    """The mechanical retrieval pass. embed_fn is injected: text -> vector.
    It is passed in, never imported, never fetched from a network. In
    production it calls the local embedding daemon; in tests it is a fake."""

    def __init__(self, embed_fn, band_low: float = BAND_LOW,
                 band_high: float = BAND_HIGH, backend: str = "auto"):
        if band_low >= band_high:
            raise ValueError(f"band_low ({band_low}) must be below band_high ({band_high})")
        self.embed_fn = embed_fn
        self.band_low = float(band_low)
        self.band_high = float(band_high)
        self.index = VectorIndex(backend=backend)
        # Parallel to index rows: {"item_id", "tags"}. No text is kept —
        # the caller owns content and looks it up by item_id.
        self._items = []

    @property
    def backend(self) -> str:
        return self.index.backend

    def __len__(self) -> int:
        return len(self._items)

    def add_item(self, item_id: str, text: str, tags=()) -> int:
        """Embed text and index it. Only the item id, tags, and vector are
        kept — the text itself is NOT stored."""
        vector = self.embed_fn(text)
        pos = self.index.add(vector)
        self._items.append({"item_id": item_id, "tags": tuple(tags)})
        return pos

    def retrieve(self, query: str, top_k: int = 5, reader: str = "local") -> list:
        """Embed the query, search, band-filter, tag-filter by reader kind,
        return ranked hits: {"item_id", "similarity", "tags"}.
        Unknown reader kinds fail closed to cloud. The candidate pool is
        the whole index: the band and tag filters can only remove hits, so
        searching just top_k would starve the result. At personal-braid
        scale this is cheap; a larger deployment would oversample instead."""
        if reader not in (READER_LOCAL, READER_CLOUD):
            reader = READER_CLOUD
        if len(self._items) == 0 or top_k <= 0:
            return []
        query_vector = self.embed_fn(query)
        scored = self.index.search(query_vector, len(self._items))
        hits = []
        for idx, sim in band_filter(scored, self.band_low, self.band_high):
            item = self._items[idx]
            if reader == READER_CLOUD and _CLOUD_WITHHELD_TAGS.intersection(item["tags"]):
                continue
            hits.append({"item_id": item["item_id"], "similarity": sim,
                         "tags": list(item["tags"])})
            if len(hits) >= top_k:
                break
        return hits
