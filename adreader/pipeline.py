"""End-to-end pipeline: image -> [ad finder] -> OCR -> VLM extraction -> verification ->
master-data linking -> AdRecord(s).

    from adreader.pipeline import process_image
    records = process_image("data/ads/page.png", find_ads_on_page=True)
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import normalize as nz
from . import ocr as ocr_mod
from . import preprocess as pp
from .extract import extract
from .db_link import safe_link_records
from .schema import AdRecord, RawExtraction
from .segment import AdRegion, find_ads
from .validate import validate
from .verify import verify
from .vlm import VLMClient


@dataclass
class AdResult:
    record: AdRecord
    raw: RawExtraction
    ocr: ocr_mod.OCRResult
    image: np.ndarray            # the (cropped) ad image
    region: AdRegion | None


def ocr_corpus_key(ocr: ocr_mod.OCRResult) -> str:
    parts = [nz.key(l.norm) for l in ocr.lines] + [nz.key(a[0]) for l in ocr.lines for a in l.alts]
    return "|".join(parts)


def process_ad(img: np.ndarray, name: str = "", langs=ocr_mod.DEFAULT_LANGS,
               client: VLMClient | None = None, verifier_model: str | None = None,
               ocr: ocr_mod.OCRResult | None = None, raw: RawExtraction | None = None,
               region: AdRegion | None = None) -> AdResult:
    """One advertisement image -> verified, linked record. `ocr`/`raw` may be passed in to
    reuse cached results (evaluation)."""
    client = client or VLMClient()
    t0 = time.perf_counter()
    timings: dict[str, float] = {}

    if ocr is None:
        ocr = ocr_mod.run_ocr(img, langs=langs)
    timings["ocr"] = round(time.perf_counter() - t0, 2)

    models = []
    if raw is None:
        run = extract(img, ocr, client=client)
        raw = run.raw
        timings["extract"] = run.seconds
        models.append(run.model)
    else:
        models.append("cached")

    verifier_raw = None
    verifier_model = verifier_model if verifier_model is not None else client.cfg.verifier_model
    if verifier_model:
        try:   # independent second opinion: image only - no OCR text, no primary answer
            v = verify(img, client, model=verifier_model)
            verifier_raw = v.raw
            timings["verify_model"] = v.seconds
            models.append(f"verifier: {v.model}")
        except Exception as e:  # verifier is optional; its failure must not block the record
            models.append(f"verifier failed: {e}")

    t = time.perf_counter()
    rec = validate(raw, ocr, image=img, image_name=name, verifier=verifier_raw)
    safe_link_records(rec, ocr_corpus_key(ocr))   # DB failure -> unresolved, never invented IDs
    timings["validate_link"] = round(time.perf_counter() - t, 2)
    timings["total"] = round(time.perf_counter() - t0, 2)
    rec.models_used = models
    rec.timings_s = timings
    return AdResult(rec, raw, ocr, img, region)


def process_image(src: str | Path | bytes | np.ndarray, name: str | None = None,
                  find_ads_on_page: bool = False, regions: list[AdRegion] | None = None,
                  **kwargs) -> list[AdResult]:
    """Image (single ad or full page) -> list of AdResults (one per advertisement)."""
    img = src if isinstance(src, np.ndarray) else pp.load_image(src)
    name = name or (Path(src).name if isinstance(src, (str, Path)) else "upload")
    client = kwargs.pop("client", None) or VLMClient()
    if regions is None and find_ads_on_page:
        regions, single = find_ads(img, client)
        if single:
            regions = None
    if not regions:
        return [process_ad(img, name, client=client, **kwargs)]
    out = []
    for i, r in enumerate(regions, 1):
        out.append(process_ad(r.crop(img), f"{name}#ad{i}", client=client, region=r, **kwargs))
    return out
