"""Landing screen - restrained Apple / OpenAI-style, pure HTML/CSS/SVG (no external JS, fonts or packages).

`gate()` shows the landing screen until the user clicks "Let's Get Started" and returns True once the
user has entered the app. It steps aside automatically under pytest (existing UI tests render the app
directly) or when ADREADER_SKIP_LANDING=1 is set.
"""
from __future__ import annotations

import os

import streamlit as st

import mascot

KEY = "entered_app"
BRAND = "Indycium Ad-Reader"

CSS = """
<style>
/* ---------- hide Streamlit chrome while the landing screen is shown ---------- */
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"],
header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"], footer { display:none !important; }
[data-testid="stAppViewContainer"], .stApp { background:#09090B !important; }
[data-testid="stMain"] .block-container, .main .block-container {
  padding-top:0 !important; padding-bottom:0 !important; max-width:1120px !important; }

/* ---------- surface: obsidian + a whisper of neutral light behind the hero ---------- */
.ix-bg { position:fixed; inset:0; z-index:0; pointer-events:none; background:#09090B; }
.ix-bg::before { content:""; position:absolute; inset:0;
  background:radial-gradient(ellipse 55% 45% at 50% 18%, rgba(255,255,255,.055), rgba(255,255,255,0) 70%); }
.ix-bg::after { content:""; position:absolute; left:50%; top:0; width:min(900px, 80vw); height:1px; transform:translateX(-50%);
  background:linear-gradient(90deg, transparent, rgba(255,255,255,.12), transparent); }

/* ---------- type ---------- */
.ix, .ix * { font-family:-apple-system, BlinkMacSystemFont, "SF Pro Display", "Inter", "Segoe UI", Roboto,
  "Helvetica Neue", Arial, sans-serif; -webkit-font-smoothing:antialiased; }
.ix-hero { position:relative; z-index:1; text-align:center; padding:3vh 16px 0; }
.ix-in { opacity:0; animation:ix-rise 1.2s cubic-bezier(0.16, 1, 0.3, 1) forwards; animation-delay:var(--d, 0s); }
@keyframes ix-rise { from { opacity:0; transform:translateY(14px); } to { opacity:1; transform:none; } }
.ix-eyebrow { margin:0 0 18px !important; font-size:12px !important; font-weight:500; letter-spacing:.2em;
  text-transform:uppercase; color:#A1A1AA !important; }
.ix-eyebrow b { color:#FAFAFA; font-weight:600; }
/* macOS titanium: soft off-white falling to graphite. Every property !important so Streamlit's theme
   can't override it; `color` is the off-white fallback where background-clip:text is unsupported. */
.ix .hero-title, div.hero-title { margin:0 auto !important; max-width:1000px; padding:0 0 .08em !important;
  font-family:-apple-system, BlinkMacSystemFont, "SF Pro Display", "Inter", "Segoe UI", sans-serif !important;
  font-size:clamp(46px, 7vw, 88px) !important; font-weight:600 !important; line-height:1.05 !important;
  letter-spacing:-0.03em !important; color:#E5E5EA !important; opacity:1 !important;
  background-image:linear-gradient(180deg, #F5F5F7 0%, #E5E5EA 38%, #8E8E93 100%) !important;
  -webkit-background-clip:text !important; background-clip:text !important; -webkit-text-fill-color:transparent !important;
  animation:ix-lift 1.2s cubic-bezier(0.16, 1, 0.3, 1) .14s both; }
@supports not ((-webkit-background-clip:text) or (background-clip:text)) {
  .ix .hero-title, div.hero-title { -webkit-text-fill-color:#E5E5EA !important; background-image:none !important; } }
@keyframes ix-lift { from { transform:translateY(14px); } to { transform:none; } }
.ix-indy { margin:0 0 6px; }
.ix-hello { margin:0 0 26px !important; font-size:14px !important; color:#A1A1AA !important; letter-spacing:-0.005em; }
.ix-sub { max-width:700px; margin:16px auto 0 !important; font-size:19px !important; font-weight:400; line-height:1.55;
  letter-spacing:-0.011em; color:#A1A1AA !important; text-align:center; }

/* ---------- CTA: the native Streamlit button as a solid white pill ---------- */
.st-key-landing_cta { position:relative; z-index:2; display:flex; justify-content:center; width:fit-content !important;
  margin:26px auto 0 !important; opacity:0; animation:ix-rise 1.2s cubic-bezier(0.16, 1, 0.3, 1) .42s forwards; }
.st-key-landing_cta button, .st-key-landing_cta [data-testid="stButton"] > button,
.st-key-landing_cta button[kind="secondary"], .st-key-landing_cta [data-testid="stBaseButton-secondary"] {
  min-height:0 !important; padding:13px 30px !important; border-radius:999px !important;
  background-color:#F5F5F7 !important; background-image:none !important; color:#1D1D1F !important;
  border:1px solid #F5F5F7 !important; box-shadow:none !important; opacity:1 !important;
  transition:background-color .25s cubic-bezier(0.16, 1, 0.3, 1), transform .25s cubic-bezier(0.16, 1, 0.3, 1) !important; }
.st-key-landing_cta button p, .st-key-landing_cta button span, .st-key-landing_cta button div {
  font-size:16px !important; font-weight:500 !important; letter-spacing:-0.01em; color:#1D1D1F !important;
  -webkit-text-fill-color:#1D1D1F !important;
  font-family:-apple-system, BlinkMacSystemFont, "SF Pro Display", "Inter", "Segoe UI", sans-serif !important; }
.st-key-landing_cta button:hover, .st-key-landing_cta [data-testid="stButton"] > button:hover,
.st-key-landing_cta button[kind="secondary"]:hover {
  background-color:#E5E5EA !important; border-color:#E5E5EA !important; color:#1D1D1F !important; transform:translateY(-1px); }
.st-key-landing_cta button:active { transform:none; background-color:#D1D1D6 !important; }
.st-key-landing_cta button:focus-visible { outline:2px solid #FAFAFA !important; outline-offset:3px; }

/* ---------- feature strip: hairline bento ---------- */
.ix-strip { position:relative; z-index:1; display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:14px;
  max-width:1000px; margin:72px auto 0; padding:0 16px; }
@media (max-width: 860px) { .ix-strip { grid-template-columns:1fr; max-width:440px; margin-top:52px; } }
.ix-card { padding:24px 24px 26px; border-radius:14px; background:rgba(255,255,255,.025);
  border:1px solid rgba(255,255,255,.08); transition:border-color .4s cubic-bezier(0.16, 1, 0.3, 1); }
.ix-card:hover { border-color:rgba(255,255,255,.18); }
.ix-idx { display:flex; align-items:center; gap:10px; margin:0 0 18px !important; font-size:12px !important;
  font-weight:500; letter-spacing:.08em; color:#71717A !important; font-variant-numeric:tabular-nums; }
.ix-idx svg { flex:none; }
.ix-ct { margin:0 !important; font-size:16px !important; font-weight:600; letter-spacing:-0.015em; color:#FAFAFA !important; line-height:1.35; }
.ix-cd { margin:8px 0 0 !important; font-size:14px !important; line-height:1.5; color:#A1A1AA !important; }
.ix-foot { position:relative; z-index:1; text-align:center; margin:56px 0 32px !important; font-size:12px !important;
  letter-spacing:.02em; color:#52525B !important; }

@media (prefers-reduced-motion: reduce) {
  .ix-in, .st-key-landing_cta { animation:none !important; opacity:1 !important; }
}
</style>
"""

_SVG = ('<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#A1A1AA" stroke-width="1.5" '
        'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{}</svg>')
ICONS = {
    "read": _SVG.format('<rect x="4" y="3.5" width="16" height="17" rx="2"/><path d="M8 8h8M8 12h8M8 16h5"/>'),
    "match": _SVG.format('<circle cx="12" cy="12" r="8.5"/><path d="M8.5 12.2l2.4 2.4 4.6-4.9"/>'),
    "search": _SVG.format('<circle cx="11" cy="11" r="6.5"/><path d="M20 20l-4.4-4.4"/>'),
}
FEATURES = [
    ("01", "read", "OCR + Vision Extraction", "Reads English and Indian-script advertisements into structured fields."),
    ("02", "match", "Zero-Hallucination DB Matching", "IDs come only from real database records, confirmed by a human reviewer."),
    ("03", "search", "Hybrid RAG + Judge LLM", "Search 206 indexed ads; every answer is fact-checked claim by claim."),
]

HERO = (
    '<div class="ix-bg" aria-hidden="true"></div>'
    '<div class="ix ix-hero">'
    f'<div class="ix-indy ix-in" style="--d:0s">{mascot.hero_html()}</div>'
    '<p class="ix-eyebrow ix-in" style="--d:.08s"><b>Indycium</b> &middot; Advertisement Intelligence</p>'
    f'<div class="hero-title" role="heading" aria-level="1">{BRAND}</div>'
    '<p class="ix-sub ix-in" style="--d:.26s">Turn any print, newspaper, or digital advertisement into verified, '
    'database&#8209;ready records &mdash; read, matched, reviewed, and searchable in one place.</p>'
    '</div>'
)


def _strip_html() -> str:
    cards = "".join(
        f'<div class="ix-card ix-in" style="--d:{.6 + i * .09:.2f}s">'
        f'<p class="ix-idx">{ICONS[ic]}<span>{n}</span></p>'
        f'<p class="ix-ct">{t}</p><p class="ix-cd">{d}</p></div>'
        for i, (n, ic, t, d) in enumerate(FEATURES))
    return (f'<div class="ix ix-strip">{cards}</div>'
            '<p class="ix ix-foot ix-in" style="--d:.95s">Indycium &middot; Advertisement Data Entry System</p>')


def skip_landing() -> bool:
    return bool(os.environ.get("PYTEST_CURRENT_TEST")) or os.environ.get("ADREADER_SKIP_LANDING") == "1"


VIEW_PARAM, VIEW_APP = "view", "app"          # ?view=app keeps the user inside the app across a browser refresh


def enter() -> None:
    st.session_state[KEY] = True
    st.query_params[VIEW_PARAM] = VIEW_APP


def go_home() -> None:
    st.session_state[KEY] = False
    st.query_params.pop(VIEW_PARAM, None)


def entered() -> bool:
    """In the app if this session clicked "Let's Get Started", or the URL says ?view=app (survives a refresh)."""
    if st.query_params.get(VIEW_PARAM) == VIEW_APP:
        st.session_state[KEY] = True
    return bool(st.session_state.get(KEY, False))


def render() -> None:
    # st.markdown (not st.html): st.html's sanitiser drops the <style> block and the page renders unstyled
    st.markdown(CSS + mascot.CSS, unsafe_allow_html=True)
    st.markdown(HERO, unsafe_allow_html=True)
    if st.button("Let's Get Started →", key="landing_cta"):
        enter()
        st.rerun()
    st.markdown(_strip_html(), unsafe_allow_html=True)


def gate() -> bool:
    """True when the main app should render; otherwise draws the landing screen."""
    if skip_landing() or entered():
        return True
    render()
    return False


def home_button() -> None:
    """Subtle way back to the landing screen, shown at the top of the sidebar."""
    if skip_landing():
        return
    if st.sidebar.button("← Home", key="landing_home", help="Back to the welcome screen"):
        go_home()
        st.rerun()
