"""Auto script detection for OCR (langs="auto"): pure logic with fake engines + one real Telugu ad."""
from pathlib import Path

import numpy as np
import pytest

from adreader import ocr

ROOT = Path(__file__).resolve().parent.parent
TELUGU_AD = ROOT / "data/eval/set50/images/040_Andhra_Jyothi_80.jpg"

TE, KA, DV = "తెలుగు ప్రకటన", "ಕನ್ನಡ ಜಾಹೀರಾತು", "हिंदी विज्ञापन"


def _crops(n=8):
    return [np.zeros((30, 200, 3), np.uint8) for _ in range(n)]


@pytest.fixture()
def fake_models(monkeypatch):
    """Each fake 'model' returns fixed readings per crop; loads are counted to prove one-at-a-time loading."""
    readings: dict[str, list[tuple[str, float]]] = {}
    loaded: list[str] = []

    def engine(lang):
        loaded.append(lang)
        if lang == "tamil":
            raise RuntimeError("model not installed")
        return lang

    engine.__wrapped__ = engine
    monkeypatch.setattr(ocr, "_engine", engine)
    monkeypatch.setattr(ocr, "_recognize", lambda lang, crops, eng=None: readings[lang][:len(crops)])
    return readings, loaded


def test_is_auto_variants():
    assert ocr.is_auto("auto") and ocr.is_auto(["auto"]) and ocr.is_auto(("auto", "latin"))
    assert not ocr.is_auto(ocr.DEFAULT_LANGS) and not ocr.is_auto(["latin"]) and not ocr.is_auto(None)


def test_probe_picks_telugu_and_skips_failed_model(fake_models):
    readings, loaded = fake_models
    readings["devanagari"] = [("खराब", 0.3)] * 8
    readings["telugu"] = [(TE, 0.95)] * 8
    readings["kannada"] = [(KA, 0.7)] * 8          # look-alike script, lower confidence on every box
    found, info = ocr._probe_scripts(_crops())
    assert found == ("telugu",)
    assert info["scores"]["telugu"] == 1.0 and info["scores"]["kannada"] == 0.0
    assert info["failed"] == ["tamil"]             # a model that cannot load is skipped, not fatal
    assert loaded == list(ocr.PROBE_SCRIPTS)       # each script model loaded once, in turn


def test_probe_mixed_scripts_and_nothing_found(fake_models):
    readings, _ = fake_models
    readings["devanagari"] = [(DV, 0.9)] * 4 + [("", 0.0)] * 4
    readings["telugu"] = [("", 0.0)] * 4 + [(TE, 0.9)] * 4
    readings["kannada"] = [("", 0.0)] * 8
    assert ocr._probe_scripts(_crops())[0] == ("devanagari", "telugu")
    readings["devanagari"] = readings["telugu"] = [("abc", 0.99)] * 8      # nothing in its own script
    assert ocr._probe_scripts(_crops())[0] == ()
    assert ocr._probe_scripts([])[0] == ()


def test_default_mode_unchanged_signature():
    """Backward compatibility: the default is still latin + devanagari and no detection metadata."""
    import inspect
    assert inspect.signature(ocr.run_ocr).parameters["langs"].default == ocr.DEFAULT_LANGS
    assert ocr.OCRResult.__dataclass_fields__["script_detection"].default is None


@pytest.mark.skipif(not TELUGU_AD.exists(), reason="Telugu sample ad not available")
def test_real_telugu_ad_is_auto_detected():
    from adreader import preprocess as pp
    try:
        res = ocr.run_ocr(pp.load_image(TELUGU_AD.read_bytes()), langs=["auto"])
    except RuntimeError as e:                      # OCR models not downloadable in this environment
        pytest.skip(str(e))
    assert res.langs == ("latin", "telugu")
    assert res.script_detection["detected"] == ["latin", "telugu"] and not res.script_detection["fallback"]
    assert any(l.script == "telugu" for l in res.lines)           # Telugu text actually read
