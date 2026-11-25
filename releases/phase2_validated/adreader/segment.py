"""Ad finder: locate individual advertisements on a full newspaper page.

The VLM proposes boxes (normalised 0-1000 coordinates); code then validates and cleans them.
Proposals are always shown to the user to confirm/adjust in the app - a wrong crop would
silently corrupt every downstream field, so this step is never fully automatic.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .vlm import VLMClient, parse_json

PROMPT = """This image is a scanned newspaper page or a part of one.
Find every PAID ADVERTISEMENT on it (product, brand, dealer, property, education, classified,
recruitment or public-notice ads). Do NOT include news articles, news photos, the newspaper's
masthead/logo, page headers, or the newspaper's own promotional strips.

Return ONLY JSON:
{"page_is_single_ad": true|false,
 "ads": [{"label": "<brand or short description>", "box": [x1, y1, x2, y2]}]}
Coordinates are integers normalised to 0-1000 (0,0 = top-left, 1000,1000 = bottom-right) and the
box must cover the whole advertisement including its border and small print.
If the whole image is one advertisement, set page_is_single_ad to true and return it as one box."""


@dataclass
class AdRegion:
    label: str
    box: tuple[int, int, int, int]     # pixel coords x1, y1, x2, y2 in the input image

    def crop(self, img: np.ndarray, pad: int = 6) -> np.ndarray:
        h, w = img.shape[:2]
        x1, y1, x2, y2 = self.box
        return img[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)]


def find_ads(img: np.ndarray, client: VLMClient | None = None) -> tuple[list[AdRegion], bool]:
    client = client or VLMClient()
    reply = client.ask([img], PROMPT, max_tokens=1500, max_side=2048)
    data = parse_json(reply.text)
    single = bool(data.get("page_is_single_ad"))
    h, w = img.shape[:2]
    regions: list[AdRegion] = []
    for ad in data.get("ads", []):
        try:
            x1, y1, x2, y2 = (float(v) for v in ad["box"])
        except (KeyError, TypeError, ValueError):
            continue
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        box = (int(x1 / 1000 * w), int(y1 / 1000 * h), int(x2 / 1000 * w), int(y2 / 1000 * h))
        bw, bh = box[2] - box[0], box[3] - box[1]
        if bw < 0.05 * w or bh < 0.03 * h:      # implausibly small -> discard
            continue
        regions.append(AdRegion(str(ad.get("label") or "ad"), box))
    if single or not regions:
        return [AdRegion("whole image", (0, 0, w, h))], True
    return regions, False
