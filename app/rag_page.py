"""Streamlit page: Ad-Reader Search & RAG. All logic lives in adreader/rag.py.

Hybrid retrieval (BM25 + multilingual-e5-small embeddings, fused with Reciprocal Rank Fusion) over every
record source, a self-query step that turns explicit metadata in the question into filters, grounded answers
with validated citations, a MANDATORY judge verification of every AI answer, and semantic similar-ad search.
Read-only over the existing records.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from adreader import rag

EXAMPLES = ["Find hindi ads",
            "Automobile ads with offers or horsepower",
            "Find advertisements mentioning zero-interest EMI",
            "Milk or health drinks for children"]
ALL = "All"
FILTER_KEYS = {"language": "rag:lang", "category": "rag:cat", "ad_type": "rag:at", "brand": "rag:brand"}
BADGE = {True: "🟢 VERIFIED", False: "🟡 UNVERIFIED"}
EXAMPLE_LABELS = ["Hindi ads", "Cars: offers / horsepower", "Zero-interest EMI", "Kids' health drinks"]


def _run_example(q: str):
    """Example button: put the query in the box, clear every filter, run the search right away."""
    st.session_state["rag:q"] = q
    for k in FILTER_KEYS.values():
        st.session_state[k] = ALL
    st.session_state["rag:verified_only"] = False
    st.session_state["rag:run_now"] = True


@st.cache_resource(show_spinner=False)
def _load(path: str, mtime: float):
    """Cached per index-file version: reloaded only when the file changes (after a refresh)."""
    return rag.load_index(Path(path))


@st.cache_resource(show_spinner=False)
def _embedder():
    return rag.get_embedder(download=True)


def _index():
    p = rag.INDEX_FILE
    return _load(str(p), p.stat().st_mtime) if p.is_file() else None


def _val(v) -> str:
    if v in (None, "", []):
        return "—"
    return ", ".join(map(str, v)) if isinstance(v, list) else str(v)


def _build(include_jobs: bool = True, include_eval: bool = True) -> str | None:
    """(Re)build the index from the sources; returns an error message or None."""
    try:
        emb = _embedder()
        new = rag.refresh_index(include_jobs=include_jobs, include_eval=include_eval, embedder=emb)
        st.session_state["rag:refreshed"] = (f"Index built: {new.counts['total']} ads ({new.counts['verified']} verified, "
                                             f"{new.counts['unverified']} unverified)" +
                                             ("" if new.vectors is not None else " - semantic layer unavailable, lexical only"))
        return None
    except rag.RefreshError as e:
        return str(e)


# ============================================================================ cards
def _why(h: rag.Hit) -> list[str]:
    d, out = h.details or {}, []
    if d.get("listing"):
        out.append("Listed because it matches the filters (no ranking words left in the question).")
    if b := d.get("bm25"):
        rel = f"relative overlap {b['relative']:.2f}" if "relative" in b else f"BM25 {b.get('score')}"
        out.append(f"**Keywords (BM25):** rank {b.get('rank')} · {rel}" +
                   (f" · matched: {', '.join(b.get('matched', [])[:12])}" if b.get("matched") else ""))
    if s := d.get("semantic"):
        rank = f"rank {s['rank']} · " if s.get("rank") else "not in the semantic top list · "
        out.append(f"**Meaning (e5 embeddings):** {rank}cosine {s.get('cosine')}" +
                   (f" · z {s['z']}" if s.get("z") is not None else ""))
    if "rrf" in d:
        out.append(f"**Fused (RRF):** {d['rrf']}")
    return out


def _image_file(path) -> Path | None:
    """The record's image: the stored path if it exists, else the same file under this checkout's data/ folder
    (the index stores absolute paths from the machine that built it, e.g. a Windows laptop vs. a cloud host)."""
    if not path:
        return None
    p = Path(path)
    if p.is_file():
        return p
    s = str(path).replace("\\", "/")
    if "/data/" in s:
        q = Path(__file__).resolve().parent.parent / "data" / s.split("/data/", 1)[1]
        if q.is_file():
            return q
    return None


def _card(h: rag.Hit, score_text: str | None):
    """Compact result card: thumbnail left, key fields right, evidence in one collapsed expander."""
    d, m = h.doc, h.doc.meta
    with st.container(border=True):
        c1, c2 = st.columns([0.16, 0.84])
        with c1:
            if (img := _image_file(d.image_path)) is not None:
                st.image(str(img), width="stretch")
            else:
                st.caption("no image")
        with c2:
            st.markdown(f"**`{d.record_id}`** · {BADGE[d.verified]}" + (f" · {score_text}" if score_text else ""))
            st.markdown(f"**{_val(m.get('brand'))}** — {_val(m.get('product'))} · _{_val(m.get('headline'))}_")
            st.caption(f"{_val(m.get('category'))} · {_val(m.get('language'))} · {_val(m.get('ad_type'))} · "
                       f"price {_val(m.get('price'))} · client {_val(m.get('client'))} · {d.label}")
            with st.expander("Why this result · indexed text & provenance", expanded=False):
                for line in _why(h):
                    st.markdown(line)
                for k, v in d.text.items():
                    st.markdown(f"- **{k}:** {v[:400]}")
                st.json(d.provenance)


def _verification(ans: rag.Answer):
    j = ans.verification
    kind, text = rag.verification_label(j)
    {"ok": st.success, "flagged": st.warning, "unverified": st.warning}[kind](
        text + ("" if kind == "ok" else " - read the answer with care; the source records below are authoritative."))
    if j is not None and j.ok:
        if j.answer_relevant is False:
            st.warning(f"Judge: the answer may not address the question - {j.relevance_reason or ''}")
        bad = [c for c in j.claims if not c.supported]
        if bad:
            st.markdown("**Unsupported claims:**\n" + "\n".join(f"- ❌ {c.claim} — _{c.reason}_" for c in bad))
        with st.expander("Claim-by-claim evidence (judge model)", expanded=False):
            for c in j.claims:
                st.markdown(f"{'✅' if c.supported else '❌'} {c.claim}" + (f"  \n_{c.reason}_" if c.reason else ""))
            st.caption(f"Faithfulness {j.faithfulness:.0%} = supported ÷ total claims, computed by the app · "
                       f"relevance: {j.relevance_reason or '—'} · judge model {j.model or '?'}. "
                       "A second AI opinion, not a guarantee." if j.faithfulness is not None else
                       f"Judge model {j.model or '?'}.")


# ============================================================================ sections
def _bento(key: str, label: str, value: str, sub: str) -> None:
    with st.container(key=key):
        st.markdown(f'<p class="rb-k">{label}</p><p class="rb-v">{value}</p><p class="rb-s">{sub}</p>',
                    unsafe_allow_html=True)


def _index_panel(idx):
    counts = idx.counts if idx else {"total": 0, "verified": 0, "unverified": 0}
    c1, c2, c3, c4 = st.columns(4, gap="small")
    with c1:                                         # real metrics (indexed / verified / unverified)
        with st.container(key="rb_0"):
            st.metric("Indexed ads", counts["total"])
            st.markdown('<p class="rb-s">searchable ad records</p>', unsafe_allow_html=True)
    with c2:
        if idx is not None and idx.vectors is not None:
            short = str(idx.emb_model or "embeddings").replace("multilingual-", "")
            _bento("rb_1", "Hybrid retrieval", f"BM25 + {short}", "keyword + semantic · RRF fusion")
        else:
            why = rag.embedding_status() or "index built without embedding vectors - refresh it"
            _bento("rb_1", "Retrieval", "BM25 only", f"semantic layer unavailable: {why}")
    with c3:
        _bento("rb_2", "Fact-checking", "Judge LLM", "every AI answer verified claim-by-claim")
    with c4:
        _bento("rb_3", "Query engine", "Auto-filters", "natural language → structured filters")
    stale = idx is not None and rag.is_stale(idx)
    with st.container(key="rag_index"):
        s1, s2 = st.columns([5, 1], vertical_alignment="center")
        built = (f"Index built {idx.built_at if idx else 'never'} · "
                 f"{'semantic vectors ready' if idx is not None and idx.vectors is not None else 'keyword index only'}")
        s1.markdown(f'<p class="ri-line">{built}' + (' <span class="ri-stale"><i>•</i> new or changed records since this build'
                                                       '</span>' if stale else "") + '</p>', unsafe_allow_html=True)
        if stale and s2.button("🔄 Refresh index", key="rag:refresh_quick", width="stretch",
                               help="Re-reads all sources and recomputes embeddings (about 1-2 minutes on CPU)."):
            with st.spinner("Reading records, computing embeddings (about 1-2 minutes on CPU)..."):
                err = _build(bool(st.session_state.get("rag:jobs", True)), bool(st.session_state.get("rag:eval", True)))
            if err:
                st.error(err)
            else:
                st.rerun()
        with st.expander("Index sources & refresh", expanded=False):
            m1, m2, _ = st.columns([1, 1, 2])            # what the index holds (moved off the headline card)
            m1.metric("🟢 Verified", counts["verified"], help="Saved by a person (Saved records)")
            m2.metric("🟡 Unverified", counts["unverified"], help="AI extractions: processed jobs and evaluation runs")
            o1, o2 = st.columns(2)
            inc_jobs = o1.checkbox("Include processed jobs (🟡 unverified)", value=bool(idx is None or idx.sources.get("processed_jobs", True)),
                                   key="rag:jobs")
            inc_eval = o2.checkbox("Include evaluation-run extractions (🟡 unverified)",
                                   value=bool(idx is None or idx.sources.get("evaluation_runs", True)), key="rag:eval",
                                   help="Pipeline extractions of the 200 bench200 ads; the provisional answer keys are not used.")
            st.caption("🟢 Verified saved records are always included. Narrow results with **Verified records only** "
                       "under Filters instead of removing sources.")
            if st.button("🔄 Refresh index", key="rag:refresh"):
                with st.spinner("Reading records, computing embeddings (about 1-2 minutes on CPU)..."):
                    err = _build(inc_jobs, inc_eval)
                if err:
                    st.error(err)
                else:
                    st.rerun()
    if msg := st.session_state.pop("rag:refreshed", None):
        st.success(msg)
    if idx is not None and idx.skipped:
        with st.expander(f"{len(idx.skipped)} source row(s) skipped while indexing"):
            st.json(idx.skipped)


def _search_tab(idx):
    fac = rag.facets(idx)
    ws = st.container(key="rag_workspace")
    ws.markdown('<p class="rw-k">Try an example</p>', unsafe_allow_html=True)
    ex = ws.columns(len(EXAMPLES), gap="small")
    for i, q in enumerate(EXAMPLES):
        ex[i].button(EXAMPLE_LABELS[i], key=f"rag:ex{i}", on_click=_run_example, args=(q,), width="stretch", help=q)
    with ws:
        query, filters, auto, top_k, ask_ai, clicked = _search_controls(fac)
    _search_results(idx, query, filters, auto, top_k, ask_ai, clicked)


def _search_controls(fac):
    query = st.text_input("Ask a question or search", key="rag:q",
                          placeholder="e.g. Which car ads offer exchange benefits?")
    filters: dict = {}
    with st.expander("Filters (optional)", expanded=False):
        f1, f2, f3, f4 = st.columns([1, 1.4, 1, 1])
        for col, (name, label) in zip((f1, f2, f3, f4), (("language", "Language"), ("category", "Category"),
                                                         ("ad_type", "Ad type"), ("brand", "Brand"))):
            opts = [ALL] + fac[name]
            key = FILTER_KEYS[name]
            if st.session_state.get(key) not in opts:      # default (or a value no longer in the index) -> All
                st.session_state[key] = ALL
            v = col.selectbox(label, opts, key=key)
            filters[name] = None if v == ALL else v
        g1, g2, g3 = st.columns([1, 1.4, 1])
        filters["verified_only"] = g1.toggle("Verified records only", key="rag:verified_only")
        auto = g2.toggle("Understand filters from my question", value=True, key="rag:auto",
                         help="'Hindi ads' -> Language = Hindi, 'automobile ads' -> Category = Automobile. "
                              "Explicit filters above always win.")
        top_k = g3.number_input("Results", 1, 50, 10, key="rag:k")
    a1, a2 = st.columns([3, 1], vertical_alignment="center")
    ask_ai = a1.toggle("Generate an AI answer (always verified by a judge model)", value=True, key="rag:ai")
    clicked = a2.button("🔎 Search", type="primary", key="rag:go", width="stretch")
    return query, filters, auto, top_k, ask_ai, clicked


def _search_results(idx, query, filters, auto, top_k, ask_ai, clicked):
    run_now = st.session_state.pop("rag:run_now", False)
    if (clicked or run_now) and (query.strip() or any(v for k, v in filters.items() if k != "verified_only")):
        with st.spinner("Retrieving (keywords + meaning)..."):
            res = rag.hybrid_search(idx, query, filters, int(top_k), embedder=_embedder(), auto_filters=auto)
        ans = None
        if ask_ai and query.strip() and res.hits:
            with st.spinner("Writing a grounded answer, then verifying every claim with the judge model..."):
                ans = rag.answer_and_verify(query, res.hits)
        st.session_state["rag:result"] = (query, res, ans)

    stored = st.session_state.get("rag:result")
    if not stored:
        return
    q, res, ans = stored
    st.markdown(f"**Results for:** _{q or '(filters only)'}_ · retrieval: `{res.mode}`")
    if res.interpreted:
        st.info("Understood from your question: " + " · ".join(f"**{k.replace('_', ' ')} = {v}**"
                                                              for k, v in res.interpreted.items())
                + ". Turn off *Understand filters from my question* (Filters) to search without it.")
    if ans is not None:
        with st.container(border=True):
            if ans.mode == "llm":
                st.markdown(f"**AI answer** · model `{ans.model}` · every statement cites a retrieved record")
                st.markdown(ans.text)
                _verification(ans)
            else:
                st.warning(f"No AI answer: {ans.reason}. {ans.text}")
    if res.note:
        st.caption(res.note)
    if not res.hits:
        st.info("No matching advertisements.")
    for h in res.hits:
        _card(h, f"Match score: {h.score:.1f}" if not h.details.get("listing") else "filter match")


def _similar_tab(idx):
    ids = [d.record_id for d in idx.docs]

    def lab(rid):
        d = idx.by_id(rid)
        return f"{rid} · {_val(d.meta.get('brand'))} · {_val(d.meta.get('product'))} · {BADGE[d.verified]}"
    with st.container(key="rag_similar"):
        st.markdown('<p class="rw-k">Pick an advertisement</p>', unsafe_allow_html=True)
        c1, c2 = st.columns([0.8, 0.2])
        pick = c1.selectbox("Advertisement", ids, format_func=lab, key="rag:pick")
        k2 = c2.number_input("How many", 1, 20, 6, key="rag:k2")
        sel = idx.by_id(pick)
        if sel:
            _card(rag.Hit(sel, 0.0, []), None)
        if st.button("Find similar advertisements", key="rag:sim", type="primary"):
            hits, msg = rag.similar(idx, pick, int(k2))
            st.session_state["rag:simres"] = (pick, hits, msg)
    sr = st.session_state.get("rag:simres")
    if sr and sr[0] == pick:
        _, hits, msg = sr
        if msg:
            st.info(msg)
        st.caption(("Similar by meaning (multilingual embeddings) and wording (BM25), fused with RRF. " if idx.vectors is not None
                    else "Similar by wording (BM25) - semantic layer unavailable. ") +
                   "Compares extracted text, not images - not visual similarity. Scores are a ranking signal, not a probability.")
        for h in hits:
            _card(h, (f"Semantic similarity: {h.score:.2f}" if idx.vectors is not None else f"Text overlap: {h.score:.0%}"))


def render():
    st.title("Ad-Reader Search & RAG")
    st.caption("Search previously processed advertisements by keywords and meaning, discover similar creatives and "
               "ask questions answered only from your records - every AI answer is verified claim by claim.")
    idx = _index()
    if idx is None:                                  # first run (or index from an older version): build it now
        with st.spinner("First run: building the search index over all records (about 1-2 minutes on CPU)..."):
            err = _build(True, True)
        if err:
            st.error(err)
            return
        st.rerun()
    _index_panel(idx)
    if not idx.docs:
        st.warning("No records to search yet.")
        return
    t1, t2 = st.tabs(["🔍 Ask & Search", "🔁 Find Similar Ads"])
    with t1:
        _search_tab(idx)
    with t2:
        _similar_tab(idx)
