"""macOS dark-mode sidebar styling for the main app (pure CSS, no external assets - Hugging Face Spaces safe).

Only presentation: the underlying widgets (the `page` st.radio, multiselects, inputs) are unchanged, so the
AppTest suites keep driving them by key.
"""
from __future__ import annotations

import streamlit as st

SIDEBAR_CSS = """
<style>
/* ---------- surface ---------- */
section[data-testid="stSidebar"] { background:#141416 !important; border-right:1px solid rgba(255,255,255,.08) !important;
  color:#E5E5EA; font-family:-apple-system, BlinkMacSystemFont, "SF Pro Text", "Inter", "Segoe UI", Roboto, sans-serif; }
section[data-testid="stSidebar"] > div, [data-testid="stSidebarContent"] { background:#141416 !important; }
[data-testid="stSidebarHeader"] { padding:8px 14px 0 !important; min-height:0 !important; height:auto !important; }
[data-testid="stSidebarHeader"] button, [data-testid="stSidebarCollapseButton"] button { color:#8E8E93 !important; }
[data-testid="stSidebarUserContent"] { padding:4px 16px 24px !important; }
[data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"] { gap:.4rem !important; }
/* the style-only markdown blocks take no room */
[data-testid="stSidebar"] .stElementContainer:has(.sb-css), [data-testid="stSidebar"] .element-container:has(.sb-css) { display:none !important; }

/* ---------- type ---------- */
[data-testid="stSidebar"] h1 { font-size:15px !important; font-weight:600 !important; letter-spacing:-0.01em !important;
  color:#F5F5F7 !important; padding:2px 0 0 !important; line-height:1.3 !important; }
[data-testid="stSidebar"] h1 span[class*="stMarkdownColoredText"], [data-testid="stSidebar"] h1 span[style*="color"] {
  display:block; margin-top:2px; font-size:11.5px !important; font-weight:400 !important; color:#8E8E93 !important; letter-spacing:0; }
[data-testid="stSidebar"] h1 a, [data-testid="stSidebar"] [data-testid="stHeaderActionElements"] { display:none !important; }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { font-size:11px !important; font-weight:600 !important; letter-spacing:.08em !important;
  text-transform:uppercase; color:#8E8E93 !important; padding:8px 0 2px !important; margin:0 !important; }
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { font-size:11px !important; font-weight:500 !important; letter-spacing:.07em;
  text-transform:uppercase; color:#8E8E93 !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p, [data-testid="stSidebar"] label p { color:#D1D1D6; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"], [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { color:#636366 !important; }
[data-testid="stSidebar"] code { background:rgba(255,255,255,.06) !important; color:#8E8E93 !important; font-size:10.5px !important; }

/* ---------- Page navigation: vertical macOS rows with icons (Page radio ONLY) ----------
   Streamlit 1.67 radio markup: label[data-testid=stRadioOption][data-selected] > span(input) + div > [div(circle), markdown] */
.st-key-page, .st-key-page [data-testid="stRadio"] { width:100% !important; }
.st-key-page { padding-bottom:10px; border-bottom:1px solid rgba(255,255,255,.08); }
.st-key-page [role="radiogroup"] { gap:1px !important; width:100% !important; }
.st-key-page [role="radiogroup"] > div { width:100%; margin:0 !important; }
[data-testid="stSidebar"] .st-key-page [data-testid="stRadio"] label[data-testid="stRadioOption"] {
  display:flex; align-items:center; gap:10px; width:100%; box-sizing:border-box; margin:0 !important; padding:7px 10px !important;
  border-radius:8px; border:1px solid transparent; cursor:pointer; color:#A1A1AA; transition:background-color .12s ease, border-color .12s ease; }
[data-testid="stSidebar"] .st-key-page label[data-testid="stRadioOption"]::before { content:""; flex:none; width:16px; height:16px;
  background-color:currentColor; -webkit-mask-repeat:no-repeat; mask-repeat:no-repeat; -webkit-mask-size:contain; mask-size:contain;
  -webkit-mask-position:center; mask-position:center; opacity:.85; }
[data-testid="stSidebar"] .st-key-page label[data-testid="stRadioOption"] > div > div:first-child:not([data-testid]) { display:none !important; }
[data-testid="stSidebar"] .st-key-page label[data-testid="stRadioOption"] > div { gap:0 !important; padding:0 !important; }
[data-testid="stSidebar"] .st-key-page label[data-testid="stRadioOption"] p { color:inherit !important; font-size:13.5px !important; line-height:1.35 !important; }
/* hover: unmistakable */
[data-testid="stSidebar"] .st-key-page [data-testid="stRadio"] label[data-testid="stRadioOption"]:hover {
  background-color:rgba(255,255,255,.10) !important; color:#FFFFFF !important; }
/* active: crisp translucent pill (checked input, or Streamlit's data-selected flag) */
[data-testid="stSidebar"] .st-key-page [data-testid="stRadio"] label[data-testid="stRadioOption"]:has(input:checked),
[data-testid="stSidebar"] .st-key-page [data-testid="stRadio"] label[data-testid="stRadioOption"][data-selected="true"] {
  background-color:rgba(255,255,255,.16) !important; color:#FFFFFF !important; border:1px solid rgba(255,255,255,.14) !important; }
[data-testid="stSidebar"] .st-key-page label[data-testid="stRadioOption"]:has(input:checked) p,
[data-testid="stSidebar"] .st-key-page label[data-testid="stRadioOption"][data-selected="true"] p { font-weight:600 !important; }
[data-testid="stSidebar"] .st-key-page label[data-testid="stRadioOption"]:has(input:checked)::before,
[data-testid="stSidebar"] .st-key-page label[data-testid="stRadioOption"][data-selected="true"]::before { opacity:1; }
[data-testid="stSidebar"] .st-key-page label[data-testid="stRadioOption"]:has(input:focus-visible) { outline:2px solid rgba(255,255,255,.55); outline-offset:1px; }
.st-key-page [role="radiogroup"] > :nth-child(1) label[data-testid="stRadioOption"]::before { -webkit-mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='5' y='3.5' width='14' height='17' rx='2'/%3E%3Cpath d='M8.5 8h7M8.5 12h7M8.5 16h4'/%3E%3C/svg%3E"); mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='5' y='3.5' width='14' height='17' rx='2'/%3E%3Cpath d='M8.5 8h7M8.5 12h7M8.5 16h4'/%3E%3C/svg%3E"); }
.st-key-page [role="radiogroup"] > :nth-child(2) label[data-testid="stRadioOption"]::before { -webkit-mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M4 7h16M4 12h16M4 17h10'/%3E%3C/svg%3E"); mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M4 7h16M4 12h16M4 17h10'/%3E%3C/svg%3E"); }
.st-key-page [role="radiogroup"] > :nth-child(3) label[data-testid="stRadioOption"]::before { -webkit-mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='8.5'/%3E%3Cpath d='M8.5 12.2l2.4 2.4 4.6-4.9'/%3E%3C/svg%3E"); mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='8.5'/%3E%3Cpath d='M8.5 12.2l2.4 2.4 4.6-4.9'/%3E%3C/svg%3E"); }
.st-key-page [role="radiogroup"] > :nth-child(4) label[data-testid="stRadioOption"]::before { -webkit-mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round'%3E%3Cellipse cx='12' cy='6' rx='7' ry='2.5'/%3E%3Cpath d='M5 6v12c0 1.4 3.1 2.5 7 2.5M19 6v5'/%3E%3Cpath d='M17 15v6M14 18h6'/%3E%3C/svg%3E"); mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round'%3E%3Cellipse cx='12' cy='6' rx='7' ry='2.5'/%3E%3Cpath d='M5 6v12c0 1.4 3.1 2.5 7 2.5M19 6v5'/%3E%3Cpath d='M17 15v6M14 18h6'/%3E%3C/svg%3E"); }
.st-key-page [role="radiogroup"] > :nth-child(5) label[data-testid="stRadioOption"]::before { -webkit-mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='5.5' y='4.5' width='13' height='16' rx='2'/%3E%3Cpath d='M9 3.5h6v3H9z'/%3E%3Cpath d='M9 13l2 2 4-4'/%3E%3C/svg%3E"); mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='5.5' y='4.5' width='13' height='16' rx='2'/%3E%3Cpath d='M9 3.5h6v3H9z'/%3E%3Cpath d='M9 13l2 2 4-4'/%3E%3C/svg%3E"); }
.st-key-page [role="radiogroup"] > :nth-child(6) label[data-testid="stRadioOption"]::before { -webkit-mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='11' cy='11' r='6.5'/%3E%3Cpath d='M20 20l-4.4-4.4'/%3E%3C/svg%3E"); mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='11' cy='11' r='6.5'/%3E%3Cpath d='M20 20l-4.4-4.4'/%3E%3C/svg%3E"); }

/* ---------- Source: horizontal 2-tab segmented control (clearly not navigation) ---------- */
.st-key-src, .st-key-src [data-testid="stRadio"] { width:100% !important; }
.st-key-src [role="radiogroup"] { display:flex !important; flex-direction:row !important; flex-wrap:nowrap !important; gap:2px !important;
  width:100% !important; padding:3px; box-sizing:border-box; border-radius:9px; background:rgba(255,255,255,.05); border:1px solid rgba(255,255,255,.08); }
.st-key-src [role="radiogroup"] > div { flex:1 1 0; min-width:0; margin:0 !important; }
.st-key-src label[data-testid="stRadioOption"] { display:flex; justify-content:center; width:100%; box-sizing:border-box; margin:0 !important;
  padding:5px 8px !important; border-radius:7px; cursor:pointer; transition:background-color .12s ease; }
.st-key-src label[data-testid="stRadioOption"] > div > div:first-child:not([data-testid]) { display:none !important; }
.st-key-src label[data-testid="stRadioOption"] > div { gap:0 !important; padding:0 !important; justify-content:center; }
.st-key-src label[data-testid="stRadioOption"] p { color:#A1A1AA !important; font-size:12.5px !important; white-space:nowrap; text-align:center; }
.st-key-src label[data-testid="stRadioOption"]:hover p { color:#FFFFFF !important; }
.st-key-src label[data-testid="stRadioOption"]:has(input:checked), .st-key-src label[data-testid="stRadioOption"][data-selected="true"] {
  background-color:rgba(255,255,255,.16) !important; box-shadow:0 1px 2px rgba(0,0,0,.35); }
.st-key-src label[data-testid="stRadioOption"]:has(input:checked) p, .st-key-src label[data-testid="stRadioOption"][data-selected="true"] p {
  color:#FFFFFF !important; font-weight:500 !important; }

/* ---------- inputs: neutral glass instead of red (react-aria markup: span[data-tag] chips, div[role=group] controls) ---------- */
[data-testid="stSidebar"] span[data-tag] { background:rgba(255,255,255,.10) !important; border:1px solid rgba(255,255,255,.12) !important;
  color:#F5F5F7 !important; border-radius:6px !important; }
[data-testid="stSidebar"] span[data-tag] span, [data-testid="stSidebar"] span[data-tag] button { color:#F5F5F7 !important; }
[data-testid="stSidebar"] span[data-tag] svg { color:#C7C7CC !important; fill:currentColor !important; }
[data-testid="stSidebar"] [data-testid="stMultiSelect"] div[role="group"], [data-testid="stSidebar"] [data-testid="stSelectbox"] div[role="group"],
[data-testid="stSidebar"] [data-testid="stTextInput"] input { background:rgba(255,255,255,.05) !important; border-color:rgba(255,255,255,.10) !important; }
[data-testid="stSidebar"] input[role="combobox"], [data-testid="stSidebar"] [data-testid="stTextInput"] input { color:#E5E5EA !important; -webkit-text-fill-color:#E5E5EA !important; }
[data-testid="stSidebar"] div[role="group"] button svg { color:#8E8E93 !important; }
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] { background:rgba(255,255,255,.04) !important;
  border:1px dashed rgba(255,255,255,.16) !important; color:#A1A1AA !important; }
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] span, [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] small { color:#A1A1AA !important; }
/* checkbox: label text stays sentence case; box is neutral, white when ticked */
[data-testid="stSidebar"] [data-testid="stCheckbox"] [data-testid="stWidgetLabel"] p { text-transform:none !important; letter-spacing:0 !important;
  font-size:13px !important; font-weight:400 !important; color:#D1D1D6 !important; }
/* checkboxes everywhere (Streamlit 1.67 markup: label > span(hidden input) + div(box) + div[stWidgetLabel]).
   Unchecked: dark surface + crisp light border. Checked: solid #F5F5F7 with a dark check (inline SVG). */
[data-testid="stCheckbox"]:not(:has(input[role="switch"])) label > div:not([data-testid]) {
  width:18px !important; height:18px !important; min-width:18px; flex:none; box-sizing:border-box;
  border-radius:5px !important; background-color:#1C1C1E !important; background-image:none !important;
  border:1.5px solid rgba(255,255,255,.38) !important; box-shadow:none !important;
  transition:border-color .15s ease, background-color .15s ease; }
[data-testid="stCheckbox"]:not(:has(input[role="switch"])) label > div:not([data-testid])::before,
[data-testid="stCheckbox"]:not(:has(input[role="switch"])) label > div:not([data-testid])::after { display:none !important; }
[data-testid="stCheckbox"]:not(:has(input[role="switch"])) label:hover > div:not([data-testid]) { border-color:rgba(255,255,255,.70) !important; background-color:#242426 !important; }
[data-testid="stCheckbox"]:not(:has(input[role="switch"])) label:has(input:checked) > div:not([data-testid]) {
  background-color:#F5F5F7 !important; border-color:#F5F5F7 !important;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 12 12'%3E%3Cpath d='M2.5 6.3l2.3 2.3 4.7-5' fill='none' stroke='%230E0E10' stroke-width='1.9' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E") !important;
  background-repeat:no-repeat !important; background-position:center !important; background-size:12px 12px !important; }
[data-testid="stCheckbox"]:not(:has(input[role="switch"])) label:has(input:checked):hover > div:not([data-testid]) { background-color:#FFFFFF !important; border-color:#FFFFFF !important; }
[data-testid="stCheckbox"]:not(:has(input[role="switch"])) label:has(input:checked) > div:not([data-testid]) * { display:none !important; }
[data-testid="stCheckbox"]:not(:has(input[role="switch"])) label:has(input:focus-visible) > div:not([data-testid]) { outline:2px solid #0A84FF; outline-offset:2px; }
[data-testid="stCheckbox"]:not(:has(input[role="switch"])) label:has(input:disabled) > div:not([data-testid]) { opacity:.45; }
[data-testid="stSidebar"] [data-testid="stTooltipIcon"] svg { color:#636366 !important; }
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] { margin-bottom:2px; }

/* ---------- ghost Home button + metadata footer ---------- */
.st-key-landing_home button, .st-key-landing_home button * { white-space:nowrap !important; }
.st-key-landing_home button { width:auto !important; background:transparent !important; border:none !important; box-shadow:none !important;
  min-height:0 !important; padding:3px 8px !important; margin-left:-8px; border-radius:6px !important; }
.st-key-landing_home button p { color:#8E8E93 !important; font-size:12.5px !important; }
.st-key-landing_home button:hover { background:rgba(255,255,255,.06) !important; }
.st-key-landing_home button:hover p { color:#F5F5F7 !important; }
.st-key-sb_meta { margin-top:18px; padding-top:10px; border-top:1px solid rgba(255,255,255,.08); }
.st-key-sb_meta p { font-size:10.5px !important; line-height:1.6 !important; color:#636366 !important; margin:0 !important; }
</style>
"""


READY_CARD_CSS = """
<style>
.rc-card { background:#141416; border:1px solid rgba(255,255,255,.08); border-radius:12px; padding:22px 22px 16px;
  font-family:-apple-system, BlinkMacSystemFont, "SF Pro Text", "Inter", "Segoe UI", Roboto, sans-serif; }
.rc-card p { margin:0; }
.rc-card .rc-h { font-size:17px !important; font-weight:600; letter-spacing:-0.015em; color:#F5F5F7 !important; margin:0 0 6px !important; }
.rc-card .rc-t { font-size:13.5px !important; line-height:1.55; color:#A1A1AA !important; margin:0 0 18px !important; }
.rc-card .rc-t b { color:#F5F5F7; font-weight:600; }
.rc-card .rc-k { font-size:10.5px !important; font-weight:600; letter-spacing:.09em; text-transform:uppercase; color:#8E8E93 !important; margin:0 0 8px !important; }
.rc-rows { border:1px solid rgba(255,255,255,.07); border-radius:10px; overflow:hidden; margin:0 0 18px; }
.rc-row { display:grid; grid-template-columns:150px 1fr; gap:12px; padding:9px 12px; border-top:1px solid rgba(255,255,255,.06); }
.rc-row:first-child { border-top:none; }
.rc-row:nth-child(odd) { background:rgba(255,255,255,.02); }
.rc-f { font-size:12.5px; font-weight:500; color:#E5E5EA; }
.rc-v { font-size:12.5px; color:#A1A1AA; line-height:1.45; }
.rc-badges { display:flex; flex-direction:column; gap:6px; margin:0 0 14px; }
.rc-badges span { font-size:12.5px; color:#E5E5EA; }
.rc-badges i { font-style:normal; color:#8E8E93; margin-left:4px; }
.rc-card .rc-n { font-size:11.5px !important; color:#636366 !important; padding-top:10px; border-top:1px solid rgba(255,255,255,.06); }
.st-key-rc_prev { margin-top:10px; padding:12px 14px; border-radius:12px; border:1px solid rgba(255,255,255,.08); background:rgba(255,255,255,.03); }
.st-key-rc_prev button { width:100%; justify-content:flex-start; }
@media (max-width: 1100px) { .rc-row { grid-template-columns:1fr; gap:2px; } }
/* picker card on the main page (neutral translucent surface: readable on the light and the dark theme) */
.st-key-proc_input { padding:22px 22px 18px; border-radius:12px; border:1px solid rgba(127,127,127,.22); background:rgba(127,127,127,.06); }
.st-key-proc_input .pc-h { margin:0 0 2px !important; font-size:17px !important; font-weight:600; letter-spacing:-0.015em; }
.st-key-proc_input .pc-t { margin:0 0 6px !important; font-size:13.5px !important; opacity:.7; }
.pc-or { margin:-2px 0 -6px !important; text-align:center; font-size:11px !important; letter-spacing:.12em; text-transform:uppercase; opacity:.5; }
/* ================= macOS card system (one surface for every card in the app) =================
   Dark values follow the spec; on a light system theme the same cards fall back to neutral greys. */
/* the app is always dark (.streamlit/config.toml base = "dark"), so the dark card tokens are the only ones */
:root { --mc-bg:rgba(255,255,255,.03); --mc-bd:rgba(255,255,255,.08); --mc-bd-h:rgba(255,255,255,.16);
        --mc-sh:inset 0 1px 0 rgba(255,255,255,.05), 0 8px 24px rgba(0,0,0,.28); --mc-dz:rgba(255,255,255,.14); }
.st-key-proc_input, .st-key-proc_bar, .st-key-rc_prev,
[data-testid="stMain"] [data-testid="stExpander"] details,
[data-testid="stMain"] [data-testid="stVerticalBlock"][class*="border"],
[data-testid="stMain"] div[data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stMain"] [data-testid="stDataFrame"] {
  background:var(--mc-bg) !important; border:1px solid var(--mc-bd) !important; border-radius:14px !important;
  box-shadow:var(--mc-sh) !important; transition:border-color .25s ease; }
.st-key-proc_input:hover, .st-key-proc_bar:hover, .st-key-rc_prev:hover,
[data-testid="stMain"] [data-testid="stExpander"] details:hover,
[data-testid="stMain"] [data-testid="stVerticalBlock"][class*="border"]:hover,
[data-testid="stMain"] div[data-testid="stVerticalBlockBorderWrapper"]:hover,
[data-testid="stMain"] [data-testid="stDataFrame"]:hover { border-color:var(--mc-bd-h) !important; }
[data-testid="stMain"] [data-testid="stDataFrame"] { overflow:hidden; padding:2px; }
[data-testid="stMain"] [data-testid="stExpander"] details summary { border-radius:14px; }
.rc-card, .lp-card { border-radius:14px !important; box-shadow:inset 0 1px 0 rgba(255,255,255,.05), 0 8px 24px rgba(0,0,0,.28);
  transition:border-color .25s ease; }
.rc-card:hover, .lp-card:hover { border-color:rgba(255,255,255,.16) !important; }

/* ---- "How it works": pipeline strip, 2x3 tiles, legend ---- */
.hw-pipe { display:flex; align-items:center; gap:8px; margin:0 0 18px; flex-wrap:wrap; }
.hw-pipe span { padding:5px 11px; border-radius:999px; font-size:12px; color:#E5E5EA; background:rgba(255,255,255,.06);
  border:1px solid rgba(255,255,255,.10); }
.hw-pipe b { font-weight:600; color:#8E8E93; margin-right:4px; font-variant-numeric:tabular-nums; }
.hw-pipe i { font-style:normal; color:#636366; font-size:12px; }
.hw-grid { display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:8px; margin:0 0 16px; }
.hw-tile { display:flex; gap:10px; align-items:flex-start; padding:11px 12px; border-radius:10px; background:rgba(255,255,255,.03);
  border:1px solid rgba(255,255,255,.07); transition:border-color .2s ease, background-color .2s ease; }
.hw-tile:hover { border-color:rgba(255,255,255,.16); background:rgba(255,255,255,.05); }
.hw-ic { flex:none; width:26px; height:26px; border-radius:7px; display:grid; place-items:center; color:#D1D1D6;
  background:rgba(255,255,255,.06); border:1px solid rgba(255,255,255,.08); }
.rc-card .hw-tt { margin:0 !important; font-size:12.5px !important; font-weight:600; color:#F5F5F7 !important; line-height:1.3; }
.rc-card .hw-td { margin:3px 0 0 !important; font-size:11.5px !important; color:#8E8E93 !important; line-height:1.4; }
.hw-legend { display:flex; flex-wrap:wrap; gap:6px; margin:0 0 12px; }
.hw-legend span { padding:4px 10px; border-radius:999px; font-size:11.5px; color:#D1D1D6; background:rgba(255,255,255,.04);
  border:1px solid rgba(255,255,255,.08); }
.hw-legend .hw-zero { color:#F5F5F7; border-color:rgba(255,255,255,.16); }
@media (max-width: 1100px) { .hw-grid { grid-template-columns:1fr; } }

/* ---- live processing card: deliberately STATIC (no animation) with a fixed height, so the 3-second
   refreshes never make it shift, shake or flash ---- */
.lp-card { background:#141416; border:1px solid rgba(255,255,255,.08); padding:18px 20px 14px; margin:0 0 10px;
  min-height:178px; box-sizing:border-box; contain:layout paint;
  font-family:-apple-system, BlinkMacSystemFont, "SF Pro Text", "Inter", "Segoe UI", Roboto, sans-serif; }
.lp-card, .lp-card * { animation:none !important; transition:none !important; }
.lp-card p { margin:0 !important; }
.lp-head { display:flex; align-items:center; gap:10px; height:22px; }
.lp-card .lp-title { font-size:15px !important; font-weight:600; color:#F5F5F7 !important; letter-spacing:-0.01em;
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis; flex:1; min-width:0; }
.lp-time { flex:none; width:52px; text-align:right; font-size:13px; color:#AEAEB2; font-variant-numeric:tabular-nums; }
.lp-card .lp-name { font-size:12.5px !important; color:#8E8E93 !important; margin:3px 0 14px !important; height:18px;
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.lp-pulse { width:9px; height:9px; border-radius:50%; background:#F5F5F7; flex:none; box-shadow:0 0 0 4px rgba(245,245,247,.12); }
.lp-completed .lp-pulse { background:#30D158; box-shadow:0 0 0 4px rgba(48,209,88,.15); }
.lp-failed .lp-pulse { background:#FF6B6B; box-shadow:0 0 0 4px rgba(255,107,107,.15); }
.lp-steps { display:grid; grid-template-columns:repeat(4, minmax(0, 1fr)); gap:6px; margin:0 0 12px; }
.lp-step { display:flex; flex-direction:column; gap:7px; padding:9px 8px 8px; border-radius:9px; background:rgba(255,255,255,.03);
  border:1px solid rgba(255,255,255,.07); height:52px; box-sizing:border-box; }
.lp-dot { width:100%; height:3px; border-radius:2px; background:rgba(255,255,255,.10); }
.lp-lbl { font-size:11.5px; color:#8E8E93; line-height:1.25; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.lp-step.done .lp-dot { background:#F5F5F7; }
.lp-step.done .lp-lbl { color:#D1D1D6; }
.lp-step.now { border-color:rgba(255,255,255,.22); background:rgba(255,255,255,.07); }
.lp-step.now .lp-dot { background:linear-gradient(90deg, #F5F5F7 0 45%, rgba(255,255,255,.18) 45% 100%); }
.lp-step.now .lp-lbl { color:#F5F5F7; font-weight:600; }
.lp-card .lp-sub { font-size:11.5px !important; color:#636366 !important; height:16px; white-space:nowrap; overflow:hidden;
  text-overflow:ellipsis; }
.lp-failed { border-color:rgba(255,99,99,.35) !important; }
/* the polling fragment must never dim / fade the card while it refreshes */
[data-stale="true"]:has(.lp-card), [data-stale="true"] .lp-card { opacity:1 !important; transition:none !important; }
@media (max-width: 1100px) { .lp-steps { grid-template-columns:repeat(2, minmax(0, 1fr)); } }

/* ---- analyze card: dropzone + OR divider ---- */
.st-key-proc_input [data-testid="stFileUploaderDropzone"], .st-key-proc_bar [data-testid="stFileUploaderDropzone"] {
  background:rgba(127,127,127,.04) !important; border:1px dashed var(--mc-dz) !important; border-radius:12px !important;
  transition:background-color .2s ease, border-color .2s ease; }
.st-key-proc_input [data-testid="stFileUploaderDropzone"]:hover { background:rgba(127,127,127,.09) !important; border-color:var(--mc-bd-h) !important; }
.pc-or { display:flex; align-items:center; gap:10px; }
.pc-or::before, .pc-or::after { content:""; flex:1; height:1px; background:var(--mc-bd); }

/* ---- results: one silky reveal when they first mount; the scroll helper takes no space ---- */
#extraction-result-anchor { scroll-margin-top:72px; }       /* clear Streamlit's top toolbar when scrolled to */
.st-key-proc_back button, .st-key-proc_back button * { white-space:nowrap !important; }
.st-key-proc_back, .st-key-proc_back button { width:auto !important; }
.st-key-result_wrap { animation:resultFadeUp .8s cubic-bezier(0.16, 1, 0.3, 1) both; }
@keyframes resultFadeUp { from { opacity:0; transform:translateY(14px); } to { opacity:1; transform:none; } }
.st-key-scroll_js { position:absolute !important; width:0 !important; height:0 !important; overflow:hidden !important; }
@media (prefers-reduced-motion: reduce) {
  .st-key-result_wrap { animation:none !important; } }

/* ================= Saved records: verification & benchmark card + record card ================= */
.vb-card, .vr-card { margin:0 0 14px; }
.st-key-vb_card { padding:20px 20px 14px; margin:0 0 6px; background:#141416 !important; border:1px solid rgba(255,255,255,.08) !important;
  border-radius:14px !important; box-shadow:inset 0 1px 0 rgba(255,255,255,.05), 0 8px 24px rgba(0,0,0,.28) !important; gap:10px !important; }
.st-key-vb_card .vb-title { margin:0 0 4px !important; font-size:17px !important; font-weight:600; letter-spacing:-0.015em; color:#F5F5F7 !important; }
.st-key-vb_card .vb-intro { margin:0 !important; font-size:13.5px !important; line-height:1.55; color:#A1A1AA !important; }
/* Streamlit's markdown block carries a negative bottom margin; inside this card it pulled the banner over the intro */
.st-key-vb_card [data-testid="stMarkdown"], .st-key-vb_card .stMarkdown,
.st-key-vb_card [data-testid="stMarkdownContainer"] { margin-bottom:0 !important; }
.st-key-vb_card [data-testid="stElementContainer"] { margin:0 !important; }
.st-key-vb_card .vb-caveat { margin:0 !important; padding-top:10px; border-top:1px solid rgba(255,255,255,.06);
  font-size:11.5px !important; line-height:1.5; color:#636366 !important; }
.st-key-vb_card .vb-caveat b { color:#8E8E93; }
.st-key-vb_card iframe { display:block; border:0; }
/* ================= toggles (st.toggle = stCheckbox with input[role=switch]): high-contrast macOS switch ================= */
[data-testid="stCheckbox"]:has(input[role="switch"]) label > div:not([data-testid]) {
  background-color:#3A3A3C !important; border:1px solid rgba(255,255,255,.22) !important; box-shadow:none !important; }
[data-testid="stCheckbox"]:has(input[role="switch"]) label > div:not([data-testid]) > div { background-color:#D1D1D6 !important; }
[data-testid="stCheckbox"]:has(input[role="switch"]:checked) label > div:not([data-testid]) {
  background-color:#F5F5F7 !important; border-color:#F5F5F7 !important; }
[data-testid="stCheckbox"]:has(input[role="switch"]:checked) label > div:not([data-testid]) > div { background-color:#0E0E10 !important; }
[data-testid="stCheckbox"]:has(input[role="switch"]:focus-visible) label > div:not([data-testid]) { outline:2px solid #0A84FF; outline-offset:2px; }

/* ================= Search & RAG ================= */
.st-key-rb_0, .st-key-rb_1, .st-key-rb_2, .st-key-rb_3, .st-key-rag_workspace, .st-key-rag_similar {
  background:#141416 !important; border:1px solid rgba(255,255,255,.08) !important; border-radius:14px !important;
  box-shadow:inset 0 1px 0 rgba(255,255,255,.05), 0 8px 24px rgba(0,0,0,.28) !important; transition:border-color .25s ease; }
.st-key-rb_0:hover, .st-key-rb_1:hover, .st-key-rb_2:hover, .st-key-rb_3:hover,
.st-key-rag_workspace:hover, .st-key-rag_similar:hover { border-color:rgba(255,255,255,.16) !important; }
.st-key-rb_0, .st-key-rb_1, .st-key-rb_2, .st-key-rb_3 { padding:14px 16px 12px; height:100%; min-height:104px; box-sizing:border-box; gap:0 !important; }
/* compact page chrome: less top padding on every app page, smaller title on Search & RAG */
[data-testid="stMain"] .block-container { padding-top:2.75rem !important; }
[data-testid="stMain"]:has(.st-key-rb_0) h1 { font-size:2.15rem !important; padding:0 0 .15rem !important; }
[data-testid="stMain"]:has(.st-key-rb_0) [data-testid="stCaptionContainer"] p { margin-bottom:0 !important; }
.st-key-rb_0 [data-testid="stMetricValue"] { font-size:22px !important; line-height:1.15 !important; }
.st-key-rb_0 [data-testid="stMetricValue"] > div { padding:6px 0 4px !important; }
/* slim status line: build time + muted stale notice + small refresh button */
.st-key-rag_index > div > [data-testid="stHorizontalBlock"]:first-child,
.st-key-rag_index [data-testid="stHorizontalBlock"]:has(.ri-line) {
  background:#141416; border:1px solid rgba(255,255,255,.08); border-radius:10px; padding:4px 6px 4px 14px; margin:2px 0 6px; }
.ri-line { margin:0 !important; font-size:12.5px !important; color:#8E8E93 !important; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.ri-stale { margin-left:10px; color:#C9A45C; }
.ri-stale i { font-style:normal; color:#FFB340; margin-right:4px; }
[class*="st-key-rag:refresh_quick"] button, .st-key-rag_index [data-testid="stHorizontalBlock"] button {
  min-height:0 !important; padding:4px 10px !important; border-radius:8px !important; }
.st-key-rag_index [data-testid="stHorizontalBlock"] button p { font-size:12.5px !important; white-space:nowrap; }
.st-key-rag_index [data-testid="stExpander"] details summary { padding-top:8px !important; padding-bottom:8px !important; }
[data-testid="stTabs"] [role="tablist"] { margin-bottom:6px !important; }
.rb-k, .st-key-rb_0 .rb-k { margin:0 !important; font-size:10.5px !important; font-weight:600; letter-spacing:.08em; text-transform:uppercase; color:#8E8E93 !important; }
.rb-v { margin:8px 0 6px !important; font-size:22px !important; font-weight:600; letter-spacing:-0.02em; color:#F5F5F7 !important; line-height:1.15; }
.rb-s { margin:0 !important; font-size:12px !important; line-height:1.45; color:#A1A1AA !important; }
.st-key-rb_0 [data-testid="stMetric"] { padding:0 !important; background:transparent !important; }
.st-key-rb_0 [data-testid="stMetricValue"] { font-size:30px !important; font-weight:600 !important; line-height:1.15 !important; color:#F5F5F7 !important; }
.st-key-rb_0 [data-testid="stHorizontalBlock"] [data-testid="stMetricValue"] { font-size:14px !important; font-weight:600 !important; color:#D1D1D6 !important; }
.st-key-rb_0 [data-testid="stHorizontalBlock"] { gap:6px !important; margin-top:2px; }
.st-key-rag_index { margin-top:2px; }
.st-key-rag_workspace, .st-key-rag_similar { padding:18px 20px 16px; margin:4px 0 14px; }
.rw-k { margin:0 0 2px !important; font-size:10.5px !important; font-weight:600; letter-spacing:.08em; text-transform:uppercase; color:#8E8E93 !important; }
/* example query pills (the only secondary buttons in the workspace): full labels, clean wrapping, no ellipsis */
.st-key-rag_workspace button[kind="secondary"] { border-radius:999px !important; min-height:36px !important; padding:6px 14px !important;
  background:rgba(255,255,255,.05) !important; border:1px solid rgba(255,255,255,.12) !important; }
.st-key-rag_workspace button[kind="secondary"]:hover { border-color:rgba(255,255,255,.28) !important; background:rgba(255,255,255,.09) !important; }
.st-key-rag_workspace button[kind="secondary"] p { white-space:normal !important; overflow:visible !important; text-overflow:clip !important;
  line-height:1.25 !important; font-size:13px !important; }
/* Indexed-ads card: the metric labels styled like the other cards' eyebrows */
.st-key-rb_0 [data-testid="stMetricLabel"] p { font-size:10.5px !important; font-weight:600 !important; letter-spacing:.08em;
  text-transform:uppercase; color:#8E8E93 !important; }
.st-key-rb_0 [data-testid="stHorizontalBlock"] [data-testid="stMetricLabel"] p { letter-spacing:.04em; text-transform:none; }
/* tabs: segmented control - pill on the active tab, no red underline */
[data-testid="stTabs"] [role="tablist"] { gap:4px !important; padding:4px !important; width:fit-content; border-radius:11px;
  background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.08); margin-bottom:10px; }
[data-testid="stTabs"] [data-testid="stTab"] { padding:7px 16px !important; border-radius:8px !important; height:auto !important;
  border:1px solid transparent !important; transition:background-color .15s ease; }
[data-testid="stTabs"] [data-testid="stTab"] p { color:#A1A1AA !important; font-size:13.5px !important; font-weight:500 !important; }
[data-testid="stTabs"] [data-testid="stTab"]:hover { background:rgba(255,255,255,.06) !important; }
[data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"] { background:rgba(255,255,255,.14) !important;
  border-color:rgba(255,255,255,.14) !important; }
[data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"] p { color:#FFFFFF !important; font-weight:600 !important; }
[data-testid="stTabs"] [data-testid="stTab"] > div:not([data-testid]) { display:none !important; }      /* the red underline */
/* Database additions: empty state */
.dba-empty { margin:0 0 14px; }
.rc-card .dba-steps { margin:0 0 12px !important; padding-left:20px !important; color:#D1D1D6; }
.rc-card .dba-steps li { margin:0 0 6px !important; font-size:13px !important; line-height:1.5; color:#D1D1D6 !important; }
.rc-card .dba-steps b { color:#F5F5F7; }
.vb-grid { display:grid; grid-template-columns:repeat(4, minmax(0, 1fr)); gap:10px; margin:4px 0 14px; }
@media (max-width: 1200px) { .vb-grid { grid-template-columns:repeat(2, minmax(0, 1fr)); } }
.vb-tile { padding:14px 14px 12px; border-radius:11px; background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.07); }
.rc-card .vb-k { margin:0 !important; font-size:10.5px !important; font-weight:600; letter-spacing:.08em; text-transform:uppercase; color:#8E8E93 !important; }
.rc-card .vb-v { margin:6px 0 4px !important; font-size:24px !important; font-weight:600; letter-spacing:-0.02em; color:#F5F5F7 !important;
  font-variant-numeric:tabular-nums; }
.rc-card .vb-u { font-size:12px; font-weight:500; letter-spacing:0; color:#8E8E93; margin-left:2px; }
.rc-card .vb-s { margin:0 !important; font-size:11.5px !important; line-height:1.5; color:#A1A1AA !important; }
.rc-card .vb-s b { color:#E5E5EA; font-weight:600; }
.rc-card .vb-ok { color:#30D158 !important; }
.rc-card .vb-bad { color:#FF6B6B !important; }
.rc-card .vb-na { font-size:14px; font-weight:500; color:#636366; }
.vr-head { display:flex; flex-direction:column; gap:2px; margin:0 0 12px; }
.rc-card .vr-meta { margin:0 !important; font-size:12.5px !important; color:#8E8E93 !important; }
.rc-card .vr-meta b { color:#E5E5EA; }
.vr-grid { display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:8px; }
.vr-cell { padding:10px 12px; border-radius:10px; background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.07); min-width:0; }
.rc-card .vr-v { margin:4px 0 0 !important; font-size:13.5px !important; color:#F5F5F7 !important; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.rc-card .vr-id { font-size:11px; color:#8E8E93; font-variant-numeric:tabular-nums; }

/* ================= Jobs dashboard ================= */
.st-key-jobs_status, .st-key-jobs_open_card, .st-key-jobs_retry_card {
  padding:18px 20px 16px; background:var(--mc-bg) !important; border:1px solid var(--mc-bd) !important; border-radius:14px !important;
  box-shadow:var(--mc-sh) !important; transition:border-color .25s ease; }
.st-key-jobs_status:hover, .st-key-jobs_open_card:hover, .st-key-jobs_retry_card:hover { border-color:var(--mc-bd-h) !important; }
.jb-stats { display:flex; gap:36px; flex-wrap:wrap; }
.jb-stats p { margin:0 !important; }
.jb-k { font-size:10.5px !important; font-weight:600; letter-spacing:.09em; text-transform:uppercase; color:#8E8E93 !important; }
.jb-v { margin-top:4px !important; font-size:16px !important; font-weight:600; color:#F5F5F7 !important; display:flex; align-items:center; gap:8px; }
.jb-dot { width:8px; height:8px; border-radius:50%; display:inline-block; }
.jb-dot.on { background:#30D158; box-shadow:0 0 0 3px rgba(48,209,88,.16); }
.jb-dot.off { background:#636366; }
.jb-warn { display:flex; flex-direction:column; gap:6px; }
.jb-warn p { margin:0 !important; padding:8px 12px; border-radius:9px; font-size:12.5px !important; line-height:1.45;
  color:#FFD60A !important; background:rgba(255,214,10,.07); border:1px solid rgba(255,214,10,.18); }
.jb-warn p.jb-okline { color:#A1A1AA !important; background:rgba(255,255,255,.03); border-color:var(--mc-bd); }
.jb-strip { display:flex; flex-wrap:wrap; gap:8px; margin:6px 0 2px; }
.jb-pill { padding:6px 12px; border-radius:999px; font-size:12.5px; color:#D1D1D6; background:rgba(255,255,255,.04);
  border:1px solid rgba(255,255,255,.09); }
.jb-pill b { margin-left:4px; color:#F5F5F7; font-weight:600; font-variant-numeric:tabular-nums; }
.jb-pill.zero { opacity:.5; }
/* jobs table: static dark HTML table, sticky header, subtle hover - no selectable canvas cells */
.jt-wrap { max-height:420px; overflow:auto; border:1px solid var(--mc-bd); border-radius:14px; background:var(--mc-bg);
  box-shadow:var(--mc-sh); }
table.jt { width:100%; border-collapse:separate; border-spacing:0; font-size:13px; color:#E5E5EA; margin:0 !important;
  font-family:-apple-system, BlinkMacSystemFont, "SF Pro Text", "Inter", "Segoe UI", Roboto, sans-serif; }
table.jt th { position:sticky; top:0; z-index:1; text-align:left; padding:10px 14px; font-size:11px; font-weight:600;
  letter-spacing:.07em; text-transform:uppercase; color:#8E8E93; background:#17171A; border-bottom:1px solid var(--mc-bd); }
table.jt td { padding:9px 14px; border-bottom:1px solid rgba(255,255,255,.05); white-space:nowrap; background:transparent; }
table.jt tbody tr:last-child td { border-bottom:none; }
table.jt tbody tr { cursor:default; }
table.jt tbody tr:hover td { background:rgba(255,255,255,.10) !important; color:#FFFFFF !important; }
table.jt tbody tr:hover td:first-child { box-shadow:inset 3px 0 0 #E5E5EA; }
table.jt td.jt-num { text-align:right; font-variant-numeric:tabular-nums; color:#AEAEB2; width:1%; }
table.jt td.jt-ad { max-width:300px; overflow:hidden; text-overflow:ellipsis; }
table.jt td.jt-err { max-width:260px; overflow:hidden; text-overflow:ellipsis; color:#FF8A80; }
table.jt td.jt-time { color:#8E8E93; font-variant-numeric:tabular-nums; }
table.jt tr.jt-failed td { background:rgba(255,99,99,.04); }
table.jt ::selection { background:#48484A; color:#F5F5F7; }
table.jt th.jt-num { text-align:right; }
table.jt td.jt-txt { max-width:360px; overflow:hidden; text-overflow:ellipsis; }
.st-key-theme_reset_js { position:absolute !important; width:0 !important; height:0 !important; overflow:hidden !important; }.jb-h { margin:0 0 2px !important; font-size:14.5px !important; font-weight:600; color:#F5F5F7 !important; letter-spacing:-0.01em; }
.jb-t { margin:0 0 10px !important; font-size:12.5px !important; color:#8E8E93 !important; }
.jb-gap { margin-top:14px !important; padding-top:12px; border-top:1px solid var(--mc-bd); }
.jb-sep { height:1px; margin:18px 0 6px; background:var(--mc-bd); }

/* slim one-row bar once an ad is chosen */
.st-key-proc_bar { padding:6px 8px 6px 16px; border-radius:10px; border:1px solid rgba(127,127,127,.22); background:rgba(127,127,127,.06); }
.st-key-proc_bar .pb-name { margin:0 !important; font-size:14px !important; font-weight:500; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.st-key-proc_bar [data-testid="stPopover"] button { min-height:0 !important; padding:5px 12px !important; border-radius:8px !important; }
.st-key-proc_bar [data-testid="stPopover"] button p { font-size:13px !important; }
/* while an ad is being analysed the picker widgets stay mounted (so their values persist) but are not shown */
.st-key-proc_input_hidden { display:none !important; }
.st-key-proc_back button { background:transparent !important; border:none !important; box-shadow:none !important; padding:2px 8px !important;
  margin-left:-8px; min-height:0 !important; }
.st-key-proc_back button p { font-size:13px !important; opacity:.75; }
.st-key-proc_back button:hover p { opacity:1; }
</style>
"""


def apply_sidebar() -> None:
    st.sidebar.markdown(READY_CARD_CSS.strip() + '\n<div class="sb-css"></div>', unsafe_allow_html=True)
    # <style> must come first: Markdown ends a <div>-led HTML block at the first blank line, which would drop the CSS
    st.sidebar.markdown(SIDEBAR_CSS.strip() + '\n<div class="sb-css"></div>', unsafe_allow_html=True)


def meta_footer(model: str, backend: str, rules: str) -> None:
    with st.sidebar.container(key="sb_meta"):
        st.caption(f"Model `{model}` ({backend})  \nValidation rules `{rules}`")
