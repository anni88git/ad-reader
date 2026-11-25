"""Indy - the Indycium Ad-Reader mascot: an astronaut with a magnifying glass, floating above a rotating Earth.

Pure inline SVG + CSS: no images, no JS, no CDN, no file paths - safe offline and on Hugging Face Spaces.
`astronaut_svg(size)` is the compact bust (sidebar), `hero_html()` the landing composition (astronaut + Earth +
speech caption), `sidebar_assistant(page)` the collapsible page-aware tip card.
"""
from __future__ import annotations

import streamlit as st

CSS = """
<style>
.indy { display:inline-block; line-height:0; }
/* zero-gravity bob + slight tilt */
.indy-bob { transform-box:fill-box; transform-origin:50% 50%; animation:indy-bob 5.5s ease-in-out infinite; }
@keyframes indy-bob { 0%, 100% { transform:translateY(0) rotate(-2.5deg); } 50% { transform:translateY(-9px) rotate(2.5deg); } }
/* waving arm pivots at the shoulder (astronaut viewBox coordinates) */
.indy-wave { transform-box:view-box; transform-origin:74px 112px; animation:indy-wave 1.5s ease-in-out infinite alternate; }
@keyframes indy-wave { from { transform:rotate(10deg); } to { transform:rotate(-26deg); } }
.indy-eye { transform-box:fill-box; transform-origin:50% 50%; animation:indy-blink 5s infinite; }
@keyframes indy-blink { 0%, 92%, 100% { transform:scaleY(1); } 94.5% { transform:scaleY(.1); } }
.indy-glint { animation:indy-glint 4.5s ease-in-out infinite; }
@keyframes indy-glint { 0%, 60%, 100% { opacity:.18; } 75% { opacity:.75; } }
.indy-lens { animation:indy-lens 3.2s ease-in-out infinite; }
@keyframes indy-lens { 0%, 100% { opacity:.25; } 50% { opacity:.9; } }
/* Earth: continents scroll horizontally inside a circular clip - one full strip per loop, seamless */
.indy-spin { animation:indy-spin 36s linear infinite; }
@keyframes indy-spin { from { transform:translateX(0); } to { transform:translateX(-240px); } }
.indy-atmo { animation:indy-atmo 6s ease-in-out infinite; }
@keyframes indy-atmo { 0%, 100% { opacity:.55; } 50% { opacity:.85; } }

.indy-hero { position:relative; display:inline-block; width:272px; max-width:80vw; line-height:0; }
.indy-hero > svg { width:100%; height:auto; overflow:visible; }
.indy-bubble { position:absolute; left:63%; top:9%; padding:8px 13px; border-radius:14px 14px 14px 4px;
  background:rgba(245,245,247,.08); border:1px solid rgba(245,245,247,.16);
  -webkit-backdrop-filter:blur(10px); backdrop-filter:blur(10px); white-space:nowrap; line-height:1.2;
  font:500 13px/1.2 -apple-system, BlinkMacSystemFont, "SF Pro Text", "Inter", "Segoe UI", sans-serif;
  color:#E5E5EA; letter-spacing:-0.005em; animation:indy-pop 1s cubic-bezier(0.16, 1, 0.3, 1) .6s both; }
@keyframes indy-pop { from { opacity:0; transform:translateY(6px) scale(.96); } to { opacity:1; transform:none; } }

@media (prefers-reduced-motion: reduce) {
  .indy-bob, .indy-wave, .indy-eye, .indy-glint, .indy-lens, .indy-spin, .indy-atmo, .indy-bubble { animation:none !important; }
}

.indy-card { display:block; padding:16px 16px 14px; border-radius:14px; background:#0B0B0E; border:1px solid rgba(255,255,255,.08);
  font-family:-apple-system, BlinkMacSystemFont, "SF Pro Display", "Inter", "Segoe UI", Roboto, sans-serif; }
.indy-head { display:flex; align-items:center; gap:12px; margin:0 0 12px; }
.indy-name { margin:0 !important; font-size:14px !important; font-weight:600; letter-spacing:-0.01em; color:#F5F5F7 !important; line-height:1.2; }
.indy-page { margin:2px 0 0 !important; font-size:11px !important; letter-spacing:.12em; text-transform:uppercase; color:#8E8E93 !important; }
.indy-lead { margin:0 0 10px !important; font-size:13px !important; line-height:1.5; color:#D1D1D6 !important; }
.indy-steps { margin:0 !important; padding:0 !important; list-style:none; counter-reset:indy; }
.indy-steps li { position:relative; margin:0 0 8px !important; padding:0 0 0 24px !important; font-size:12.5px !important;
  line-height:1.5; color:#AEAEB2 !important; counter-increment:indy; }
.indy-steps li::before { content:counter(indy, decimal-leading-zero); position:absolute; left:0; top:1px;
  font-size:10.5px; letter-spacing:.04em; color:#636366; font-variant-numeric:tabular-nums; }
.indy-steps b { color:#F5F5F7; font-weight:600; }
</style>
"""

# ---------------------------------------------------------------- astronaut (own viewBox 0 0 200 220)
_ASTRONAUT = """
<defs>
<linearGradient id="suit{u}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#FAFAFC"/><stop offset="1" stop-color="#C7C7CC"/></linearGradient>
<linearGradient id="visor{u}" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#3A3A3C"/><stop offset=".55" stop-color="#151517"/><stop offset="1" stop-color="#0A0A0C"/></linearGradient>
<radialGradient id="glass{u}" cx=".35" cy=".35" r=".8"><stop offset="0" stop-color="#DCE8FF" stop-opacity=".35"/><stop offset="1" stop-color="#9DB4E0" stop-opacity=".08"/></radialGradient>
</defs>
<g class="indy-bob">
<rect x="64" y="100" width="72" height="62" rx="16" fill="#AEAEB2"/>
<rect x="76" y="150" width="20" height="40" rx="9" fill="url(#suit{u})"/>
<rect x="104" y="150" width="20" height="40" rx="9" fill="url(#suit{u})"/>
<rect x="74" y="182" width="24" height="12" rx="6" fill="#8E8E93"/>
<rect x="102" y="182" width="24" height="12" rx="6" fill="#8E8E93"/>
<rect x="70" y="102" width="60" height="58" rx="18" fill="url(#suit{u})"/>
<rect x="86" y="118" width="28" height="18" rx="5" fill="#E5E5EA" stroke="#C7C7CC"/>
<circle cx="94" cy="127" r="2.6" fill="#8E8E93"/><circle cx="101" cy="127" r="2.6" fill="#AEAEB2"/><circle cx="108" cy="127" r="2.6" fill="#636366"/>
<g class="indy-wave">
<path d="M74 112 L54 84" stroke="url(#suit{u})" stroke-width="15" stroke-linecap="round"/>
<circle cx="51" cy="79" r="8.5" fill="#E5E5EA" stroke="#C7C7CC"/>
</g>
<path d="M126 114 L144 134" stroke="url(#suit{u})" stroke-width="15" stroke-linecap="round"/>
<path d="M150 131 L159 119" stroke="#8E8E93" stroke-width="5" stroke-linecap="round"/>
<circle cx="167" cy="108" r="13.5" fill="url(#glass{u})" stroke="#D1D1D6" stroke-width="4"/>
<path class="indy-lens" d="M159 104 A9 9 0 0 1 166 98.5" stroke="#FFFFFF" stroke-width="2.2" stroke-linecap="round" fill="none"/>
<circle cx="146" cy="136" r="8.5" fill="#E5E5EA" stroke="#C7C7CC"/>
<path d="M100 32 V22" stroke="#AEAEB2" stroke-width="2.5" stroke-linecap="round"/><circle cx="100" cy="19" r="3.5" fill="#E5E5EA"/>
<circle cx="100" cy="68" r="38" fill="url(#suit{u})" stroke="#D1D1D6"/>
<rect x="72" y="50" width="56" height="40" rx="20" fill="url(#visor{u})"/>
<ellipse class="indy-eye" cx="90" cy="71" rx="3.4" ry="4.4" fill="#E5E5EA"/>
<ellipse class="indy-eye" cx="110" cy="71" rx="3.4" ry="4.4" fill="#E5E5EA"/>
<path class="indy-glint" d="M80 58 Q88 52 98 53" stroke="#FFFFFF" stroke-width="3" stroke-linecap="round" fill="none"/>
<circle cx="62" cy="70" r="5" fill="#D1D1D6"/><circle cx="138" cy="70" r="5" fill="#D1D1D6"/>
</g>"""

# ---------------------------------------------------------------- Earth (local coords: centre 70,70 r 70)
_LAND = ('<path d="M10 40 C25 25 55 28 62 45 C70 62 52 70 48 88 C44 104 30 112 24 97 C18 82 4 66 10 40 Z"/>'
         '<path d="M22 108 C30 104 38 112 34 124 C30 132 22 130 20 120 Z"/>'
         '<path d="M95 30 C115 17 150 21 170 33 C184 43 177 58 160 60 C146 62 140 74 128 70 C112 66 100 52 95 30 Z"/>'
         '<path d="M117 78 C130 73 143 84 141 100 C139 117 126 127 118 118 C109 108 107 86 117 78 Z"/>'
         '<path d="M190 96 C203 89 221 93 223 106 C225 117 208 121 198 117 C190 113 184 102 190 96 Z"/>'
         '<path d="M200 30 C210 24 226 28 228 38 C230 46 216 50 206 46 C198 42 194 35 200 30 Z"/>'
         '<circle cx="80" cy="64" r="3"/><circle cx="176" cy="74" r="2.5"/><circle cx="232" cy="80" r="2"/>')
_EARTH = """
<defs>
<radialGradient id="ocean{u}" cx=".38" cy=".32" r=".85"><stop offset="0" stop-color="#22344F"/><stop offset=".6" stop-color="#132037"/><stop offset="1" stop-color="#0A1222"/></radialGradient>
<radialGradient id="shade{u}" cx=".3" cy=".25" r="1"><stop offset="0" stop-color="#FFFFFF" stop-opacity=".16"/><stop offset=".55" stop-color="#000000" stop-opacity="0"/><stop offset="1" stop-color="#000000" stop-opacity=".6"/></radialGradient>
<radialGradient id="halo{u}" cx=".5" cy=".5" r=".5"><stop offset=".78" stop-color="#7FA8FF" stop-opacity="0"/><stop offset=".9" stop-color="#7FA8FF" stop-opacity=".28"/><stop offset="1" stop-color="#7FA8FF" stop-opacity="0"/></radialGradient>
<clipPath id="globe{u}"><circle cx="70" cy="70" r="70"/></clipPath>
</defs>
<circle class="indy-atmo" cx="70" cy="70" r="88" fill="url(#halo{u})"/>
<circle cx="70" cy="70" r="70" fill="url(#ocean{u})"/>
<g clip-path="url(#globe{u})">
<g class="indy-spin" fill="#3E5A63">
<g>{land}</g><g transform="translate(240 0)">{land}</g>
</g>
<path d="M0 70 H140 M8 38 H132 M8 102 H132" stroke="#FFFFFF" stroke-opacity=".05"/>
<circle cx="70" cy="70" r="70" fill="url(#shade{u})"/>
</g>
<circle cx="70" cy="70" r="69.5" fill="none" stroke="#9DB8FF" stroke-opacity=".35"/>"""


def astronaut_svg(size: int = 56, uid: str = "a", bust: bool = True) -> str:
    """The astronaut alone; `bust` crops to helmet, arms and magnifying glass (compact, for the sidebar)."""
    vb = "28 8 162 150" if bust else "0 0 200 220"
    body = _ASTRONAUT.replace("{u}", uid)
    return (f'<span class="indy" role="img" aria-label="Indy the astronaut, waving and holding a magnifying glass">'
            f'<svg width="{size}" height="{size}" viewBox="{vb}" fill="none" xmlns="http://www.w3.org/2000/svg">'
            f'{body}</svg></span>').replace("\n", "")


svg = astronaut_svg          # backwards-compatible name


def hero_html(uid: str = "hero") -> str:
    """Landing composition: floating astronaut above a rotating Earth, with a speech caption."""
    earth = _EARTH.replace("{land}", _LAND).replace("{u}", uid)
    astro = _ASTRONAUT.replace("{u}", uid)
    return (f'<div class="indy-hero" role="img" aria-label="Indy the astronaut waving above a rotating Earth">'
            f'<svg viewBox="0 0 340 345" fill="none" xmlns="http://www.w3.org/2000/svg">'
            f'<g transform="translate(100 188)">{earth}</g>'
            f'<svg x="78" y="0" width="184" height="202" viewBox="0 0 200 220" overflow="visible">{astro}</svg>'
            f'</svg><span class="indy-bubble">Hello! I’m Indy</span></div>').replace("\n", "")


# ---------------------------------------------------------------- peeking Indy (sidebar)
# Helmet + two gloved hands over the top edge of the tip card. viewBox 0 0 120 64; the card's top edge sits at
# y = 52 (the SVG overlaps the card by 12px), so the helmet is clipped there and the hands rest on the edge.
_PEEK = """<svg class="indy-peek-svg" width="104" height="56" viewBox="0 0 120 64" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
<defs>
<linearGradient id="pkSuit" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#FAFAFC"/><stop offset="1" stop-color="#C7C7CC"/></linearGradient>
<linearGradient id="pkVisor" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#3A3A3C"/><stop offset=".55" stop-color="#151517"/><stop offset="1" stop-color="#0A0A0C"/></linearGradient>
<clipPath id="pkEdge"><rect x="0" y="0" width="120" height="52"/></clipPath>
</defs>
<g clip-path="url(#pkEdge)"><g class="indy-peek">
<path d="M60 20 V11" stroke="#AEAEB2" stroke-width="2" stroke-linecap="round"/><circle cx="60" cy="9" r="3" fill="#E5E5EA"/>
<circle cx="60" cy="48" r="28" fill="url(#pkSuit)" stroke="#D1D1D6"/>
<circle cx="32.5" cy="48" r="4" fill="#D1D1D6"/><circle cx="87.5" cy="48" r="4" fill="#D1D1D6"/>
<rect x="40" y="33" width="40" height="28" rx="14" fill="url(#pkVisor)"/>
<ellipse class="indy-eye" cx="52" cy="44" rx="2.6" ry="3.4" fill="#E5E5EA"/>
<ellipse class="indy-eye" cx="68" cy="44" rx="2.6" ry="3.4" fill="#E5E5EA"/>
<path class="indy-glint" d="M46 39 Q51 35 57 35.5" stroke="#FFFFFF" stroke-width="2" stroke-linecap="round" fill="none"/>
</g></g>
<g fill="#ECECF0" stroke="#C7C7CC">
<rect x="31" y="47" width="20" height="12" rx="6"/><rect x="69" y="47" width="20" height="12" rx="6"/>
</g>
<path d="M37 49.5 V53 M41 49 V53 M45 49.5 V53 M75 49.5 V53 M79 49 V53 M83 49.5 V53" stroke="#C7C7CC" stroke-width="1.2" stroke-linecap="round"/>
</svg>"""

PEEK_CSS = """
<style>
.indy-peekwrap { position:relative; margin:2px 0 6px; font-family:-apple-system, BlinkMacSystemFont, "SF Pro Text", "Inter", "Segoe UI", sans-serif; }
.indy-peek-svg { display:block; position:relative; z-index:2; margin:0 auto -12px; }
.indy-peek { animation:indy-peek 4.8s ease-in-out infinite; }
@keyframes indy-peek { 0%, 100% { transform:translateY(0); } 45% { transform:translateY(-3px); } 60% { transform:translateY(-3px); } }
.indy-tip { position:relative; z-index:1; padding:15px 13px 11px; border-radius:12px; background:rgba(255,255,255,.045);
  border:1px solid rgba(255,255,255,.09); box-shadow:0 1px 0 rgba(255,255,255,.04) inset; }
.indy-tip p { margin:0 !important; font-size:12.5px !important; line-height:1.45 !important; color:#D1D1D6 !important; letter-spacing:-0.003em; }
.indy-tip b { color:#F5F5F7 !important; font-weight:600; }
.indy-tip .indy-who { display:block; margin-bottom:3px; font-size:10.5px !important; letter-spacing:.08em; text-transform:uppercase; color:#8E8E93 !important; }
@media (prefers-reduced-motion: reduce) { .indy-peek { animation:none !important; } }
</style>
"""

# two short lines max per page
TIPS: dict[str, str] = {
    "Process advertisement": "Pick or upload an ad, hit <b>Extract &amp; verify</b>, then clear any 🟠 flags and save.",
    "Jobs": "Retry 🔁 failed jobs, or open a completed one to review and save it.",
    "Saved records": "Human-verified records. They show as 🟢 <b>VERIFIED</b> in Search &amp; RAG.",
    "Database additions": "Only reviewers add brands and products here, never the AI. Undo if unused.",
    "Audit answers": "Check each drafted answer against the image. Your fixes measure label errors.",
    "Search & RAG": "Search 206 ads by keywords + meaning. A judge fact-checks every AI answer.",
}


def peek_html(page: str) -> str:
    tip = TIPS.get(page) or "I’m here to help."
    return (f'<div class="indy-peekwrap" role="note" aria-label="Indy says">{_PEEK.replace(chr(10), "")}'
            f'<div class="indy-tip"><p><span class="indy-who">Indy</span>{tip}</p></div></div>')


def sidebar_assistant(page: str) -> None:
    """Compact peeking-Indy tip card for the active page (no expander, two lines max)."""
    st.sidebar.markdown(CSS + PEEK_CSS + peek_html(page), unsafe_allow_html=True)
