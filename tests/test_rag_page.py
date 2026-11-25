"""Headless UI test of the Search & RAG page (temp index file, no model API call)."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from adreader import rag

APP = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")


@pytest.fixture()
def temp_index(tmp_path, monkeypatch):
    from adreader import jobs, opsdb, usage
    monkeypatch.setattr(opsdb, "OPS_DB", tmp_path / "ops.sqlite")
    monkeypatch.setattr(jobs, "JOBS_DIR", tmp_path / "jobs")
    usage._local.conn = None

    def doc(rid, verified, **m):
        meta = {k: m.get(k) for k in rag.META_KEYS}
        meta["language"] = m.get("language", [])
        return rag.AdDoc(rid, "submission" if verified else "job", verified, "t" if verified else None, meta,
                         {"headline": m.get("headline", ""), "offers": m.get("offers", "")}, None, {"source_id": rid})
    idx = rag.build_index([
        doc("submission:1", True, brand="Kia", category="Automobile > Cars", language=["English"],
            headline="Sorento 230 horsepower", offers="offer: benefits up to inr 50000"),
        doc("job:2", False, brand="Hyundai", category="Automobile > Cars", language=["Hindi"],
            headline="Creta offer", offers="offer: 0% EMI")])
    p = tmp_path / "rag" / "index.json"
    rag.save_index(idx, p)
    monkeypatch.setattr(rag, "INDEX_FILE", p)
    monkeypatch.setattr(rag, "is_stale", lambda *a, **k: False)
    yield p
    usage._local.conn = None


def test_search_page_flow(temp_index):
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.sidebar.radio(key="page").set_value("Search & RAG").run()
    assert not at.exception, at.exception
    assert any("Ad-Reader Search & RAG" in t.value for t in at.title)
    assert [m.value for m in at.metric][:3] == ["2", "1", "1"]

    at.toggle(key="rag:ai").set_value(False)
    at.button(key="rag:ex1").click().run()                  # example query fills the box and runs
    assert at.text_input(key="rag:q").value.startswith("Automobile ads")
    at.button(key="rag:go").click().run()
    assert not at.exception, at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "submission:1" in text and "🟢 VERIFIED" in text

    at.selectbox(key="rag:pick").set_value("submission:1").run()
    at.button(key="rag:sim").click().run()
    assert not at.exception, at.exception
    sim_text = " ".join(m.value for m in at.markdown)
    assert "job:2" in sim_text and "🟡 UNVERIFIED" in sim_text
    assert any("not visual similarity" in c.value for c in at.caption)


def test_filters_default_all_in_collapsed_expander_and_examples_reset_and_run(temp_index):
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.sidebar.radio(key="page").set_value("Search & RAG").run()
    assert not at.exception, at.exception
    exp = next(e for e in at.expander if e.label == "Filters (optional)")
    assert exp.proto.expanded is False
    for k in ("rag:lang", "rag:cat", "rag:at", "rag:brand"):
        assert at.selectbox(key=k).value == "All" and at.selectbox(key=k).options[0] == "All"

    # set a filter that would exclude the Kia ad, then click an example: filters reset + search runs at once
    at.toggle(key="rag:ai").set_value(False)
    at.selectbox(key="rag:lang").set_value("Hindi").run()
    at.button(key="rag:ex1").click().run()                  # no Search click
    assert not at.exception, at.exception
    for k in ("rag:lang", "rag:cat", "rag:at", "rag:brand"):
        assert at.selectbox(key=k).value == "All"
    text = " ".join(m.value for m in at.markdown)
    assert "Results for:" in text and "submission:1" in text


def test_tabs_banner_scores_and_judge(temp_index, monkeypatch):
    import json as _json

    def fake_llm(prompt, system):            # one stub for both the answer and the judge call
        if "fact-checking judge" in system:
            return _json.dumps({"claims": [{"claim": "Kia offers benefits", "supported": True, "reason": "submission:1"},
                                           {"claim": "Hyundai is free", "supported": False, "reason": "not in records"}],
                                "answer_relevant": True, "relevance_reason": "ok"}), "judge-stub"
        return "Kia offers benefits up to ₹50,000 [submission:1].", "answer-stub"
    monkeypatch.setattr(rag, "default_llm", fake_llm)
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.sidebar.radio(key="page").set_value("Search & RAG").run()
    assert not at.exception, at.exception
    assert [t.label for t in at.tabs] == ["🔍 Ask & Search", "🔁 Find Similar Ads"]
    assert not any(c.key == "rag:judge" for c in at.checkbox)                   # judge is not optional any more
    at.button(key="rag:ex1").click().run()
    assert not at.exception, at.exception
    md = " ".join(m.value for m in at.markdown)
    assert "Match score:" in md and "relevance (BM25" not in md
    assert any("Flagged: 1/2 claims grounded" in w.value for w in at.warning)  # mandatory verification shown
    assert any("Hyundai is free" in m.value for m in at.markdown)               # unsupported claim listed
    assert not any("self:" in m.value for m in at.markdown)
