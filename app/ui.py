"""
Presentation helpers for the Streamlit app: the stylesheet and small HTML/SVG fragments.

Style: true-black flat (dark). Elevation is lightness alone (#000 ground, #111 card,
#2a2a2a nested), no borders or shadows on cards, colour only on data marks (one lime
hue), the primary control inverted (white pill, black label). Charts are drawn here in
SVG so they follow the same rules: one hue, no legend for one series, square marks.
"""

import base64
import html
from typing import Dict, List, Optional, Sequence, Tuple

ACCENT = "#c6f24e"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
:root {
  --ground:#000000; --surface:#111111; --raised:#2a2a2a; --border:#1a1a1a; --rule:#1c1c1c;
  --control:#444444; --text:#ffffff; --muted:#b4b4b4; --faint:#949494;
  --accent:#c6f24e; --negative:#e5484d; --warning:#e5a93b;
  --pill-positive-bg:#222a10; --pill-positive-ink:#c6f24e;
  --pill-warning-bg:#2b2414; --pill-warning-ink:#e5a93b;
  --pill-neutral-bg:#222222; --pill-neutral-ink:#ffffff;
  --radius-card:20px; --gap:14px; --pad:22px;
}
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
  background: var(--ground) !important; color: var(--text);
  font-family: "Inter", system-ui, sans-serif !important; font-size: 14px;
}
.stApp p, .stApp span:not([data-testid="stIconMaterial"]), .stApp div, .stApp input, .stApp button, .stApp label,
.stApp td, .stApp th, .stApp h1, .stApp h2 { font-family: "Inter", system-ui, sans-serif; }
[data-testid="stIconMaterial"] { font-family: "Material Symbols Rounded" !important; }
[data-testid="stHeader"], [data-testid="stToolbar"], #MainMenu, footer,
[data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none !important; }
.block-container, [data-testid="stMainBlockContainer"] {
  max-width: 1152px !important; padding: 0 0 96px 0 !important; margin: 0 auto;
}
@media (max-width: 1200px) { .block-container, [data-testid="stMainBlockContainer"] { padding: 0 20px 96px !important; } }
@media (max-width: 640px)  { .block-container, [data-testid="stMainBlockContainer"] { padding: 0 16px 96px !important; } }
[data-testid="stVerticalBlock"] { gap: var(--gap) !important; }
[data-testid="stHorizontalBlock"] { gap: var(--gap) !important; }

/* cards: Streamlit's bordered container becomes a flat #111 surface */
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]) ,
div[data-testid="stVerticalBlockBorderWrapper"][class*="border"] {
  background: var(--surface); border: none !important; border-radius: var(--radius-card);
}
[data-testid="stVerticalBlockBorderWrapper"] > div { border: none !important; }
.st-key-card, [class*="st-key-card"] {
  background: var(--surface) !important; border-radius: var(--radius-card) !important;
  padding: var(--pad) !important; border: none !important;
}
[class*="st-key-nested"] { background: var(--raised) !important; border-radius: 12px !important; padding: 14px 16px !important; }

/* top bar */
.vf-bar { display:flex; align-items:center; justify-content:space-between; height:48px; margin-top:8px; }
.vf-brand { display:flex; align-items:center; gap:10px; font-weight:600; font-size:15px; color:var(--text); }
.vf-status { color: var(--muted); font-size: 13px; }

/* type */
h1.vf-title { font-size:20px !important; font-weight:600 !important; margin:0 !important; padding:0 !important;
  letter-spacing:0; line-height:28px; color: var(--text); }
.vf-sub { color: var(--muted); font-size: 14px; line-height: 20px; margin-top: 4px; }
.vf-section { font-size:16px; font-weight:600; color:var(--text); margin:0 0 2px 0; }
.vf-label { font-size:12px; color:var(--faint); line-height:16px; }
.vf-muted { color: var(--muted); }
.vf-num { font-variant-numeric: tabular-nums; }
.vf-figure { font-size:30px; font-weight:500; line-height:36px; font-variant-numeric: tabular-nums; }
.vf-track-title { font-size:20px; font-weight:600; line-height:26px; }
.vf-row-title { font-size:14px; font-weight:500; line-height:20px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.vf-row-sub { font-size:12px; color:var(--muted); line-height:18px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.vf-p { font-size:14px; line-height:21px; color: var(--text); }

/* pills */
.vf-pills { display:flex; flex-wrap:wrap; gap:6px; margin: 8px 0; }
.vf-pill { flex:none; white-space:nowrap; display:inline-flex; align-items:center; height:22px; padding:0 9px;
  border-radius:999px; font-size:12px; font-weight:500; background:var(--pill-neutral-bg); color:var(--pill-neutral-ink); }
.vf-pill.pos { background: var(--pill-positive-bg); color: var(--pill-positive-ink); }
.vf-pill.warn { background: var(--pill-warning-bg); color: var(--pill-warning-ink); }

/* covers */
.vf-cover { width:100%; aspect-ratio:1/1; object-fit:cover; border-radius:4px; display:block; background:var(--raised); }
.vf-cover-sm { width:48px; height:48px; object-fit:cover; border-radius:3px; display:block; background:var(--raised); }
.vf-cover-empty { display:flex; align-items:center; justify-content:center; color:var(--faint); font-size:12px; }

/* lists */
.vf-ctx { display:grid; grid-template-columns: 1fr auto; column-gap:12px; row-gap:2px; margin-top:10px; }
.vf-ctx .name { font-size:14px; } .vf-ctx .val { font-size:14px; text-align:right; font-variant-numeric: tabular-nums; }
.vf-ctx .bar { grid-column: 1 / -1; height:2px; background: var(--raised); margin: 4px 0 8px; }
.vf-ctx .bar > i { display:block; height:2px; background: var(--muted); }
.vf-ctx .bar.top > i { background: var(--text); }
.vf-grid3 { display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); column-gap:24px; row-gap:12px; margin-top:16px; }
.vf-grid3 .vf-row-title, .vf-grid3 .vf-row-sub { white-space: normal; }
.vf-narrow { display:none; }
@media (max-width: 640px) { .vf-wide { display:none; } .vf-narrow { display:block; }
  .vf-cover { max-width: 200px; }
  /* "Also fits" rows: cover | title | fit on one line, the player under them */
  .st-key-card_more [data-testid="stHorizontalBlock"] { display:grid !important;
    grid-template-columns: 48px minmax(0,1fr) 44px; column-gap:12px; row-gap:6px; align-items:center; }
  .st-key-card_more [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { width:auto !important; min-width:0 !important; flex:none !important; }
  .st-key-card_more [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(4) { grid-column: 1 / -1; }
}
@media (max-width: 640px) { .vf-grid3 { grid-template-columns: repeat(2, minmax(0,1fr)); } .vf-status { display:none; } }
.vf-follow { display:grid; grid-template-columns: auto 1fr; column-gap:14px; row-gap:10px; margin-top:12px; font-size:13px; }
.vf-follow .p { color: var(--text); font-weight:500; white-space:nowrap; } .vf-follow .d { color: var(--muted); }
.vf-bars { display:grid; grid-template-columns: 110px 1fr 40px; column-gap:12px; row-gap:12px; align-items:center; margin:14px 0 4px; }
.vf-bars .n { font-size:13px; } .vf-bars .v { font-size:13px; text-align:right; font-variant-numeric: tabular-nums; color: var(--muted); }
.vf-bars .t { height:8px; background: var(--raised); border-radius:4px; } .vf-bars .t > i { display:block; height:8px; border-radius:4px; background: var(--accent); }
.vf-trace { display:grid; grid-template-columns: 96px 1fr; column-gap:16px; row-gap:10px; font-size:13px; }
.vf-trace .a { color: var(--text); font-weight:500; } .vf-trace .d { color: var(--muted); }
.vf-kv { display:grid; grid-template-columns: 1fr auto; row-gap:6px; font-size:13px; margin-top:12px; }
.vf-kv .k { color: var(--faint); } .vf-kv .v { color: var(--text); text-align:right; }

/* tables */
table.vf-table { width:100%; border-collapse:collapse; font-size:14px; }
.vf-table th { font-size:12px; font-weight:400; color:var(--faint); text-align:left; padding:0 10px 10px 0; }
.vf-table td { padding:12px 10px 12px 0; border-top:1px solid var(--rule); }
.vf-table .num { text-align:right; font-variant-numeric: tabular-nums; }
.vf-table .best { color: var(--accent); }

/* controls: primary is an inverted pill; everything else quiet */
.stButton > button, .stFormSubmitButton > button, [data-testid="stPopover"] button {
  border-radius:999px !important; height:36px; min-height:36px; padding:0 16px !important;
  background: transparent !important; color: var(--text) !important; border: 1px solid var(--control) !important;
  font-weight:500; box-shadow:none !important;
}
.stButton > button:hover, [data-testid="stPopover"] button:hover { border-color:#6a6a6a !important; background:#161616 !important; }
.stButton > button[kind="primary"], .stFormSubmitButton > button {
  background:#ffffff !important; color:#000000 !important; border-color:#ffffff !important;
}
.stFormSubmitButton > button { white-space: nowrap; min-width: 112px; }
.stFormSubmitButton > button p, .stButton > button[kind="primary"] p { color:#000000 !important; font-weight:500; }
.stButton > button p { color: inherit !important; }
.st-key-card_examples .stButton > button { height:auto; min-height:36px; padding:8px 16px !important; white-space:normal;
  text-align:left; justify-content:flex-start; border-radius:18px !important; }
.stButton > button:focus-visible { outline: 2px solid #ffffff !important; outline-offset: 2px; }
[data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea {
  background: var(--surface) !important; color: var(--text) !important; border:1px solid var(--control) !important;
  border-radius: 12px !important; font-size:15px !important; min-height:44px;
}
[data-testid="stTextInput"] input::placeholder { color: var(--faint) !important; }
[data-testid="stTextInput"] > div, [data-testid="stTextInput"] div[data-baseweb] { background:transparent !important; border:none !important; }

/* tabs: white underline, never the accent */
[data-testid="stTabs"] [role="tablist"] { gap: 24px; border-bottom:1px solid var(--rule); }
[data-testid="stTabs"] button[role="tab"] { color:#bbbbbb; font-size:14px; font-weight:500; padding: 10px 0; background:none; }
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] { color:#ffffff; }
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background:#ffffff !important; height:2px; }
[data-testid="stTabs"] [data-baseweb="tab-border"] { display:none; }

[data-testid="stExpander"] details { background: transparent; border:none !important; }
[data-testid="stExpander"] summary { color: var(--muted); font-size:13px; padding-left:0; }
[data-testid="stExpander"] summary:hover { color: var(--text); }
audio { width:100%; height:36px; color-scheme: dark; }
[data-testid="stAudio"] { margin-top: 6px; }
[data-testid="stCaptionContainer"] { color: var(--faint) !important; }
[data-testid="stMarkdownContainer"] p { color: var(--text); }
[data-testid="stAlert"] { background: var(--pill-warning-bg) !important; color: var(--pill-warning-ink) !important; border:none; border-radius:12px; }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
</style>
"""

MARK = (f'<svg width="20" height="20" viewBox="0 0 20 20" aria-hidden="true">'
        f'<rect x="2" y="8" width="3.2" height="9" rx="1.6" fill="{ACCENT}"/>'
        f'<rect x="8.4" y="3" width="3.2" height="14" rx="1.6" fill="{ACCENT}"/>'
        f'<rect x="14.8" y="10" width="3.2" height="7" rx="1.6" fill="{ACCENT}"/></svg>')


def svg_img(svg: str, alt: str, style: str = "display:block;width:100%;height:auto") -> str:
    """st.html sanitises inline <svg> away, so SVG ships as a data: image (with real alt text)."""
    if 'xmlns=' not in svg:
        svg = svg.replace("<svg ", '<svg xmlns="http://www.w3.org/2000/svg" ', 1)
    data = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f'<img src="data:image/svg+xml;base64,{data}" alt="{html.escape(alt, quote=True)}" style="{style}">'


def e(text) -> str:
    return html.escape(str(text), quote=True)


def top_bar(status: str) -> str:
    mark = svg_img(MARK, "", style="display:block;width:20px;height:20px")
    return (f'<div class="vf-bar"><div class="vf-brand">{mark}<span>VibeFinder</span></div>'
            f'<div class="vf-status">{e(status)}</div></div>')


def pills(items: Sequence[Tuple[str, str]]) -> str:
    """[(text, kind)] with kind '' | 'pos' | 'warn'."""
    return '<div class="vf-pills">' + "".join(
        f'<span class="vf-pill {k}">{e(t)}</span>' for t, k in items) + "</div>"


def cover(url: Optional[str], alt: str, small: bool = False) -> str:
    cls = "vf-cover-sm" if small else "vf-cover"
    if url:
        return f'<img class="{cls}" src="{e(url)}" alt="{e(alt)}" loading="lazy">'
    return f'<div class="{cls} vf-cover-empty" role="img" aria-label="{e(alt)}">no art</div>'


def context_list(retrieved: List[Tuple[str, float]], titles: Dict[str, str]) -> str:
    if not retrieved:
        return '<div class="vf-p vf-muted" style="margin-top:8px">No listening context matched; ranked on your words alone.</div>'
    top = max(s for _, s in retrieved) or 1.0
    rows = []
    for i, (cid, score) in enumerate(retrieved):
        width = max(2.0, 100.0 * score / top)
        rows.append(f'<div class="name">{e(titles.get(cid, cid))}</div>'
                    f'<div class="val">{score:.2f}</div>'
                    f'<div class="bar{" top" if i == 0 else ""}"><i style="width:{width:.0f}%"></i></div>')
    return '<div class="vf-ctx">' + "".join(rows) + "</div>"


def context_grid(items: Sequence[Tuple[str, str]]) -> str:
    """The knowledge base as a quiet three-column list: situation over its genres."""
    return '<div class="vf-grid3">' + "".join(
        f'<div><div class="vf-row-title">{e(t)}</div><div class="vf-row-sub">{e(g)}</div></div>'
        for t, g in items) + "</div>"


def follow_ups(rows: Sequence[Tuple[str, str]]) -> str:
    return '<div class="vf-follow">' + "".join(f'<div class="p">{e(p)}</div><div class="d">{e(d)}</div>'
                                               for p, d in rows) + "</div>"


def trace(steps: List[Dict]) -> str:
    return '<div class="vf-trace">' + "".join(
        f'<div class="a">{e(s["action"])}</div><div class="d">{e(s["detail"])}</div>' for s in steps) + "</div>"


def kv(rows: Sequence[Tuple[str, str]]) -> str:
    return '<div class="vf-kv">' + "".join(f'<div class="k">{e(k)}</div><div class="v">{e(v)}</div>'
                                           for k, v in rows) + "</div>"


FEATURES = (("energy", "Energy"), ("danceability", "Danceability"), ("acousticness", "Acousticness"))


def landing_chart(targets: Dict, songs: List[Dict]) -> str:
    """Drawn twice, wide and narrow, so its 12-13px labels stay readable at every width."""
    return (f'<div class="vf-wide">{_landing_svg(targets, songs, 1100, 128)}</div>'
            f'<div class="vf-narrow">{_landing_svg(targets, songs, 360, 96)}</div>')


def _landing_svg(targets: Dict, songs: List[Dict], width: int, label_w: int) -> str:
    """The signature element: the request's target band per feature, and where each pick lands.

    One row per feature on a shared 0-1 axis: a 6px track, the target band in the pale
    tint of the data hue, each pick as a white dot and the top pick in the hue itself.
    Features the request did not constrain show their track with no band ("any")."""
    right_pad, row_h, top = 76, 44, 8
    plot = width - label_w - right_pad
    h = top + row_h * len(FEATURES) + 22
    x = lambda v: label_w + plot * max(0.0, min(1.0, v))   # noqa: E731
    out = [f'<svg viewBox="0 0 {width} {h}" width="{width}" height="{h}" '
           f'font-family="Inter, Segoe UI, system-ui, sans-serif">']
    for i, (attr, name) in enumerate(FEATURES):
        cy = top + row_h * i + row_h / 2
        out.append(f'<text x="0" y="{cy + 4:.1f}" fill="#b4b4b4" font-size="13">{name}</text>')
        out.append(f'<rect x="{label_w}" y="{cy - 3:.1f}" width="{plot}" height="6" rx="3" fill="#2a2a2a"/>')
        t = targets.get(attr)
        if t is not None:
            lo, hi = (t, t) if not isinstance(t, (tuple, list)) else t
            lo, hi = max(0.0, lo - (0.02 if lo == hi else 0)), min(1.0, hi + (0.02 if lo == hi else 0))
            out.append(f'<rect x="{x(lo):.1f}" y="{cy - 9:.1f}" width="{max(3.0, x(hi) - x(lo)):.1f}" height="18" '
                       f'rx="4" fill="#3a4a12"/>')
            span = f"{lo:.2f}-{hi:.2f}" if hi - lo > 0.05 else f"{(lo + hi) / 2:.2f}"
            out.append(f'<text x="{width}" y="{cy + 4:.1f}" fill="#949494" font-size="12" text-anchor="end" '
                       f'style="font-variant-numeric:tabular-nums">{span}</text>')
        else:
            out.append(f'<text x="{width}" y="{cy + 4:.1f}" fill="#949494" font-size="12" text-anchor="end">any</text>')
        for j, song in reversed(list(enumerate(songs))):
            fill = ACCENT if j == 0 else "#e6e6e6"
            r = 6 if j == 0 else 4.5
            out.append(f'<circle cx="{x(song[attr]):.1f}" cy="{cy:.1f}" r="{r}" fill="{fill}" stroke="#111" '
                       f'stroke-width="2"><title>{e(song["title"])}: {name.lower()} {song[attr]:.2f}</title></circle>')
    base = top + row_h * len(FEATURES) + 14
    for v in (0.0, 0.5, 1.0):
        anchor = "start" if v == 0 else "end" if v == 1 else "middle"
        out.append(f'<text x="{x(v):.1f}" y="{base:.1f}" fill="#949494" font-size="12" text-anchor="{anchor}">{v:g}</text>')
    out.append("</svg>")
    return svg_img("".join(out), "Target ranges for energy, danceability and acousticness, with each "
                   "recommended song plotted on them")


def bar_list(rows: List[Tuple[str, float]], fmt: str = "{:.2f}") -> str:
    """Horizontal bars on a 0-1 scale in HTML (crisp at any width), all in the one data hue.
    A zero has no stub: it prints 0.00 with an empty track."""
    out = ['<div class="vf-bars">']
    for name, v in rows:
        out.append(f'<div class="n">{e(name)}</div><div class="t"><i style="width:{100 * max(0.0, min(1.0, v)):.0f}%"></i></div>'
                   f'<div class="v">{fmt.format(v)}</div>')
    out.append("</div>")
    return "".join(out)


def table(headers: Sequence[str], rows: Sequence[Sequence], numeric_from: int = 1,
          best: Optional[Dict[int, int]] = None) -> str:
    """A rules-and-alignment table. `best` maps column -> row index to mark in the data hue."""
    best = best or {}
    head = "".join(f'<th class="{"num" if i >= numeric_from else ""}">{e(h)}</th>' for i, h in enumerate(headers))
    body = []
    for r, row in enumerate(rows):
        cells = []
        for i, v in enumerate(row):
            cls = ("num" if i >= numeric_from else "") + (" best" if best.get(i) == r else "")
            cells.append(f'<td class="{cls.strip()}">{e(v)}</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f'<table class="vf-table"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>'
