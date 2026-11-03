"""test_retriever.py — tests for deepgravity_core.dg_retriever.

Uses injected FAKE vectors, deterministic, no model, no network, no live
engine — the Qwen3-Embedding-0.6B weights are not on the box yet. Covers:
cosine basics; the band keeps the middle and drops both extremes (and the
edges themselves, strict); tunable band edges; NT present for local but
withheld for cloud (NK too, matching dg_turns.render_log); unknown reader
kinds fail closed to cloud; empty index returns no hits; top_k respected
and ranked; FAISS and numpy backends agree on the same inputs (FAISS
cases skipped, not faked, if faiss is unavailable); no content stored or
returned — ids, tags, and scores only.

Run: /srv/dg-dev/.venv/bin/python tests/test_retriever.py (plain asserts).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine" / "src"))

from deepgravity_core.dg_retriever import (
    BAND_HIGH,
    BAND_LOW,
    FAISS_AVAILABLE,
    Retriever,
    VectorIndex,
    band_filter,
    cosine,
    in_band,
)


def _vec(sim: float):
    """Unit vector whose cosine against [1, 0, 0] is `sim`."""
    return [sim, math.sqrt(max(0.0, 1.0 - sim * sim)), 0.0]


_TABLE = {
    "query": [1.0, 0.0, 0.0],
    "echo": [1.0, 0.0, 0.0],        # sim 1.00 — above the band
    "near echo": _vec(0.8),         # sim 0.80 — above the band
    "good one": _vec(0.7),          # in the band
    "good two": _vec(0.6),          # in the band
    "good three": _vec(0.5),        # in the band
    "far": _vec(0.2),               # sim 0.20 — below the band
    "unrelated": [0.0, 1.0, 0.0],   # sim 0.00 — below the band
}


def _fake_embed(text):
    return list(_TABLE.get(text, [0.0, 0.0, 1.0]))


def _retriever_with_items(**kwargs):
    r = Retriever(_fake_embed, **kwargs)
    r.add_item("echo", "echo", tags=["UF"])
    r.add_item("near echo", "near echo", tags=["UF"])
    r.add_item("good one", "good one", tags=["UF"])
    r.add_item("good two", "good two", tags=["NT"])
    r.add_item("good three", "good three", tags=["NK"])
    r.add_item("far", "far", tags=["UF"])
    r.add_item("unrelated", "unrelated", tags=["UF"])
    return r


def test_cosine_basics():
    assert cosine([1, 0, 0], [1, 0, 0]) == 1.0
    assert cosine([1, 0, 0], [0, 1, 0]) == 0.0
    assert cosine([1, 0, 0], [-1, 0, 0]) == -1.0
    assert cosine([1, 0, 0], [0, 0, 0]) == 0.0
    assert cosine([1, 0], [1, 0, 0]) == 0.0
    assert abs(cosine([2, 0, 0], [1, 0, 0]) - 1.0) < 1e-12  # scale-invariant
    print("ok: cosine basics")


def test_in_band_edges_are_strict():
    # Literal floats: no vector float-noise involved at this level.
    assert in_band(0.6) is True
    assert in_band(BAND_LOW) is False, "0.45 itself is out (strict band)"
    assert in_band(BAND_HIGH) is False, "0.75 itself is out (strict band)"
    assert in_band(0.44) is False
    assert in_band(0.76) is False
    assert band_filter([(0, 0.6), (1, 0.45), (2, 0.75), (3, 0.2)]) == [(0, 0.6)]
    print("ok: band edges are strict (0.45 and 0.75 themselves are out)")


def test_band_keeps_middle_drops_extremes():
    r = _retriever_with_items()
    hits = r.retrieve("query", top_k=10, reader="local")
    ids = [h["item_id"] for h in hits]
    assert ids == ["good one", "good two", "good three"], ids
    sims = [h["similarity"] for h in hits]
    assert sims == sorted(sims, reverse=True), "ranked best first"
    assert all(0.45 < s < 0.75 for s in sims)
    print("ok: band keeps the middle (0.5/0.6/0.7), drops 1.0/0.8/0.2/0.0")


def test_band_edges_tunable():
    r = _retriever_with_items(band_low=0.1, band_high=0.9)
    ids = [h["item_id"] for h in r.retrieve("query", top_k=10, reader="local")]
    assert ids == ["near echo", "good one", "good two", "good three", "far"], ids
    print("ok: tunable band edges (0.1/0.9 admits 0.8 and 0.2, still drops 1.0/0.0)")


def test_nt_present_for_local_withheld_for_cloud():
    r = _retriever_with_items()
    local_ids = [h["item_id"] for h in r.retrieve("query", top_k=10, reader="local")]
    assert "good two" in local_ids, "NT item present for local"
    cloud_ids = [h["item_id"] for h in r.retrieve("query", top_k=10, reader="cloud")]
    assert "good two" not in cloud_ids, "NT item must never reach a cloud reader"
    assert "good one" in cloud_ids
    print("ok: NT present for local, withheld for cloud")


def test_nk_withheld_for_cloud_too():
    # Same rule as dg_turns.render_log: NK was never to be retained, so it
    # must never travel either.
    r = _retriever_with_items()
    cloud_ids = [h["item_id"] for h in r.retrieve("query", top_k=10, reader="cloud")]
    assert "good three" not in cloud_ids, "NK item withheld for cloud"
    local_ids = [h["item_id"] for h in r.retrieve("query", top_k=10, reader="local")]
    assert "good three" in local_ids
    print("ok: NK withheld for cloud too (render_log rule)")


def test_unknown_reader_fails_closed_to_cloud():
    r = _retriever_with_items()
    for weird in ("banana", "", None, "CLOUD", "Local"):
        ids = [h["item_id"] for h in r.retrieve("query", top_k=10, reader=weird)]
        assert "good two" not in ids, "unknown reader must fail closed to cloud"
        assert "good three" not in ids, "unknown reader must fail closed to cloud"
        assert "good one" in ids
    print("ok: unknown reader kinds fail closed to cloud")


def test_empty_index_returns_no_hits():
    r = Retriever(_fake_embed)
    assert r.retrieve("query", reader="local") == []
    assert r.retrieve("query", reader="cloud") == []
    assert len(r) == 0
    print("ok: empty index returns no hits")


def test_top_k_respected_and_ranked():
    r = _retriever_with_items()
    hits = r.retrieve("query", top_k=2, reader="local")
    assert [h["item_id"] for h in hits] == ["good one", "good two"]
    assert hits[0]["similarity"] > hits[1]["similarity"]
    assert r.retrieve("query", top_k=0, reader="local") == []
    print("ok: top_k respected, best first")


def test_backends_agree():
    if not FAISS_AVAILABLE:
        print("SKIP: faiss not installed — FAISS backend cases skipped, numpy only")
        r = Retriever(_fake_embed, backend="numpy")
        assert r.backend == "numpy"
        assert [h["item_id"] for h in r.retrieve("query", top_k=10, reader="local")] == [
            "good one", "good two", "good three"]
        return
    faiss_r = _retriever_with_items(backend="faiss")
    numpy_r = _retriever_with_items(backend="numpy")
    assert faiss_r.backend == "faiss" and numpy_r.backend == "numpy"
    faiss_hits = faiss_r.retrieve("query", top_k=10, reader="local")
    numpy_hits = numpy_r.retrieve("query", top_k=10, reader="local")
    assert [h["item_id"] for h in faiss_hits] == [h["item_id"] for h in numpy_hits]
    for f, n in zip(faiss_hits, numpy_hits):
        assert abs(f["similarity"] - n["similarity"]) < 1e-5, (f, n)
    # Same agreement at the raw index level, including full ranking.
    fi = VectorIndex(backend="faiss")
    ni = VectorIndex(backend="numpy")
    for text in ("echo", "near echo", "good one", "good two", "far", "unrelated"):
        fi.add(_TABLE[text])
        ni.add(_TABLE[text])
    f_top = fi.search([1.0, 0.0, 0.0], 6)
    n_top = ni.search([1.0, 0.0, 0.0], 6)
    assert [i for i, _ in f_top] == [i for i, _ in n_top]
    for (_, fs), (_, ns) in zip(f_top, n_top):
        assert abs(fs - ns) < 1e-5, (fs, ns)
    print("ok: FAISS and numpy backends agree on the same inputs")


def test_no_content_stored_or_returned():
    r = _retriever_with_items()
    for item in r._items:
        assert set(item.keys()) == {"item_id", "tags"}, item
    hits = r.retrieve("query", top_k=10, reader="local")
    for h in hits:
        assert set(h.keys()) == {"item_id", "similarity", "tags"}, h
    # The embedded texts are never stored: nothing kept but ids and tags.
    kept = [v for item in r._items for v in (item["item_id"], item["tags"])]
    assert all(isinstance(v, (str, tuple)) for v in kept)
    print("ok: no content stored or returned — ids, tags, and scores only")


def test_dimension_mismatch_raises():
    idx = VectorIndex(backend="numpy")
    idx.add([1.0, 0.0, 0.0])
    try:
        idx.add([1.0, 0.0])
    except ValueError:
        pass
    else:
        raise AssertionError("dimension mismatch must raise ValueError")
    try:
        idx.search([1.0, 0.0], 1)
    except ValueError:
        pass
    else:
        raise AssertionError("query dimension mismatch must raise ValueError")
    print("ok: dimension mismatch raises ValueError")


def test_embedder_is_injected_not_imported():
    r = Retriever(_fake_embed)
    assert r.embed_fn is _fake_embed, "embedder must be the injected function"
    print("ok: embedder is the injected function, never imported")


def test_bad_band_params_raise():
    for low, high in ((0.75, 0.45), (0.5, 0.5)):
        try:
            Retriever(_fake_embed, band_low=low, band_high=high)
        except ValueError:
            pass
        else:
            raise AssertionError("inverted band must raise ValueError")
    try:
        VectorIndex(backend="nope")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown backend must raise ValueError")
    print("ok: bad band params and unknown backend raise ValueError")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed")
