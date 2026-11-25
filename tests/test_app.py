"""Headless UI test of the Streamlit POC (uses cached sample results - no API calls)."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "app" / "streamlit_app.py")
KIA = ROOT / "data/ads/_kia_from_screenshot.jpg"


@pytest.fixture(autouse=True)
def isolated_jobs(tmp_path, monkeypatch):
    """App tests must never write to the real job store or call a model API."""
    from adreader import jobs, opsdb, usage
    monkeypatch.setattr(opsdb, "OPS_DB", tmp_path / "ops.sqlite")
    monkeypatch.setattr(jobs, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setenv("NVIDIA_VERIFIER_MODEL", "")       # cached extraction + no verifier call
    usage._local.conn = None
    yield
    usage._local.conn = None


def _at():
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception, at.exception
    return at


def test_app_loads():
    at = _at()
    assert any("Ad Data Entry" in t.value for t in at.sidebar.title)


def test_kia_flow_review_gate_and_save():
    if not (ROOT / "data/outputs/ocr_cache/_kia_from_screenshot.latin-devanagari.pkl").exists():
        pytest.skip("cached Kia results missing")
    at = _at()
    sb = at.sidebar.selectbox(key="sample")
    sb.set_value(next(o for o in sb.options if "_kia_from_screenshot" in str(o))).run()
    assert not at.exception, at.exception
    at.button(key="run").click().run()                  # creates a persistent job
    assert not at.exception, at.exception
    here = next(b for b in at.button if b.key and b.key.startswith("here:"))   # no worker in tests
    here.click().run()
    assert not at.exception, at.exception
    assert any("Review and save" in s.value for s in at.subheader)

    # flagged items exist -> save is blocked until reviewed
    save = next(b for b in at.button if b.key and b.key.endswith(":save"))
    assert save.disabled
    assert any("still need your check" in e.value for e in at.error)

    # tick every 'verified' checkbox for flagged scalar fields
    for cb in at.checkbox:
        if cb.key and cb.key.endswith(":chk"):
            cb.check()
    at.text_input(key="reviewer").input("test-reviewer")
    at.run()
    assert not at.exception, at.exception
    # advertiser ID must not be pre-selected (it is only a suggestion)
    adv = next(s for s in at.selectbox if s.key and s.key.endswith(":advertiser:id"))
    assert adv.value is None

    # 'Use suggestion' fills the dealer names but the reviewer must still verify
    use = next(b for b in at.button if b.key and b.key.endswith(":advertiser:use"))
    use.click().run()
    assert not at.exception, at.exception
    adv_text = next(t for t in at.text_input if t.key and t.key.endswith(":advertiser"))
    assert adv_text.value == "Allied Kia / West Coast Kia"
    chk = next(c for c in at.checkbox if c.key and c.key.endswith(":advertiser:chk"))
    assert chk.value is False


def test_non_default_scripts_warn():
    at = _at()
    at.sidebar.multiselect(key="langs").unselect("devanagari").run()
    assert not at.exception, at.exception
    assert any("Not the default setting" in w.value for w in at.sidebar.warning)


def test_overview_and_form_preview_render():
    """Workstream B panels: field status + DB matching overview and the live data-entry form."""
    if not (ROOT / "data/outputs/ocr_cache/_kia_from_screenshot.latin-devanagari.pkl").exists():
        pytest.skip("cached Kia results missing")
    at = _at()
    assert any("candidate_v4" in c.value for c in at.sidebar.caption)
    sb = at.sidebar.selectbox(key="sample")
    sb.set_value(next(o for o in sb.options if "_kia_from_screenshot" in str(o))).run()
    at.button(key="run").click().run()
    next(b for b in at.button if b.key and b.key.startswith("here:")).click().run()
    assert not at.exception, at.exception
    labels = [e.label for e in at.expander]
    assert any("Field status & database matching" in l for l in labels)
    assert any("Data-entry form - live preview" in l for l in labels)
    assert any("still need review" in w.value for w in at.warning)          # draft lists pending items
    overview = at.dataframe[0].value
    assert set(overview["Field"]) >= {"Brand", "Product", "Client / advertiser", "Headline", "Description",
                                      "Price", "Offers", "Contacts: phones", "Language", "Category", "Ad type"}
    brand = overview[overview["Field"] == "Brand"].iloc[0]
    assert brand["Database match"].startswith("✅") and brand["DB ID / candidates"].startswith("#")
    at.checkbox(key=next(c.key for c in at.checkbox if c.key and ":showjson:" in c.key)).check().run()
    assert not at.exception, at.exception
    assert at.json, "JSON payload not shown"
