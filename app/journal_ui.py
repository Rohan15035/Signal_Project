"""
journal_ui.py -- the "journal figure" presentation layer for the Streamlit app.

The app's content and physics live in `streamlit_app.py`; this module only
decides how things look. The visual model is a figure page from a journal
paper: lettered image panels with a numbered caption, three-rule tables
instead of metric cards, and "Remark." paragraphs instead of coloured alert
boxes.

PROJECTOR NOTES
---------------
The palette and sizes are tuned for a projector in a lit room, which washes
out light tints and thin strokes:

* Body text is near-black and secondary text is still dark (slate-700), so
  captions stay legible from the back of the room.
* The accent is a saturated mid-blue rather than a navy, which projectors
  crush to black.
* The three chart series differ in hue, marker shape AND dash pattern, so
  they stay distinguishable on a washed-out screen and for colour-blind
  viewers.
* The root font size is raised to 18 px, which scales every Streamlit
  widget with it (they are sized in rem).

The fonts come from Google Fonts. Without a network connection the page
falls back to Georgia / the system sans, which keeps the same character.
"""

from __future__ import annotations

import base64
import html
import io
import re
from typing import Iterable, Sequence

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

PAPER = "#FFFFFF"
PANEL = "#EEF1F5"
INK = "#0F172A"
INK_2 = "#334155"
INK_3 = "#475569"
HAIRLINE = "#CBD5E1"
BLUE = "#1D5FA8"
ORANGE = "#C2571A"
GREEN = "#2E7D4F"
RED = "#A4262C"

# Series order matches `ks.ACQUISITION_MASKS`: Cartesian, radial, variable
# density. Colour, marker and dash all differ, so no series relies on hue.
SERIES_COLORS = [BLUE, ORANGE, GREEN, INK_3]
SERIES_SHAPES = ["circle", "square", "triangle-up", "diamond"]
SERIES_DASHES = [[1, 0], [7, 3], [2, 2], [5, 2, 1, 2]]

SERIF = "'Source Serif 4', Georgia, 'Times New Roman', serif"
SANS = "'Source Sans 3', 'Segoe UI', Helvetica, Arial, sans-serif"

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;0,8..60,700;1,8..60,400;1,8..60,600&display=swap');

html {{ font-size: 18px; }}
html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"],
[data-testid="stMarkdownContainer"], button, input, select, textarea {{
  font-family: {SANS};
}}
[data-testid="stAppViewContainer"] {{ background: {PAPER}; color: {INK}; }}
.block-container {{ padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1500px; }}

/* Streamlit chrome that has no place in a presentation. */
[data-testid="stDecoration"], [data-testid="stAppDeployButton"],
[data-testid="stMainMenu"] {{ display: none !important; }}
header[data-testid="stHeader"] {{ background: transparent; }}

/* Headings are serif, like section titles in a paper. */
h1, h2, h3, h4 {{ font-family: {SERIF} !important; color: {INK}; letter-spacing: -0.005em; }}
h1 {{ font-weight: 700 !important; font-size: 2.05rem !important; line-height: 1.2 !important; padding-bottom: 0 !important; }}
h2 {{ font-weight: 600 !important; font-size: 1.55rem !important; }}
h3 {{ font-weight: 600 !important; font-size: 1.3rem !important; }}
h4 {{ font-weight: 600 !important; font-size: 1.12rem !important; }}
[data-testid="stHeadingWithActionElements"] a {{ display: none; }}

[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {{
  color: {INK}; line-height: 1.6;
}}
code {{ color: {INK} !important; background: {PANEL} !important; }}

/* Sidebar: a quiet "methods" panel with small-caps field labels. */
[data-testid="stSidebar"] {{ background: {PANEL}; border-right: 1px solid {HAIRLINE};
  min-width: 330px !important; }}
[data-testid="stSidebar"] h2 {{ font-size: 1.25rem !important; }}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{
  text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.74rem;
  font-weight: 700; color: {INK_2};
}}

/* Tabs: underlined text, no pills. Newer Streamlit renders tabs as
   [data-testid="stTab"], older releases as baseweb buttons; both are styled. */
[data-baseweb="tab-list"], [role="tablist"] {{ gap: 1.15rem; }}
button[data-baseweb="tab"], [data-testid="stTab"] {{ background: transparent; }}
button[data-baseweb="tab"] p, [data-testid="stTab"] p {{ font-size: 1rem; color: {INK_3}; font-weight: 600; }}
button[data-baseweb="tab"][aria-selected="true"] p,
[data-testid="stTab"][aria-selected="true"] p {{ color: {INK}; }}
[data-baseweb="tab-highlight"] {{ background-color: {BLUE}; height: 3px; }}

/* Expanders read as an appendix, not a card. */
[data-testid="stExpander"] details {{ border: 1px solid {HAIRLINE}; border-radius: 2px; }}
[data-testid="stExpander"] summary p {{ font-family: {SERIF}; font-weight: 600; font-size: 1.02rem; }}

/* ---- Journal elements ---------------------------------------------- */
.jr-subtitle {{ font-family: {SERIF}; font-style: italic; color: {INK_2};
  font-size: 1.1rem; margin: -0.4rem 0 0.6rem; }}
.jr-lede {{ font-family: {SERIF}; font-size: 1.05rem; line-height: 1.6; color: {INK};
  max-width: 78ch; margin: 0 0 0.4rem; }}

.jr-fig {{ margin: 0.4rem 0 1.1rem; }}
.jr-grid {{ display: grid; gap: 0.7rem; }}
.jr-panel {{ margin: 0; }}
/* The frame keeps a saturated (all-white) panel visible against the page. */
.jr-img {{ position: relative; background: #000; line-height: 0; box-shadow: 0 0 0 1px {INK_3}; }}
.jr-img img {{ width: 100%; height: auto; display: block; image-rendering: auto; }}
.jr-letter {{ position: absolute; top: 0.4rem; left: 0.4rem; background: {PAPER}; color: {INK};
  font-family: {SERIF}; font-weight: 700; font-size: 0.95rem; line-height: 1;
  padding: 0.22rem 0.36rem; }}
.jr-panel figcaption {{ font-size: 0.86rem; color: {INK_2}; margin-top: 0.35rem; line-height: 1.35; }}
.jr-cap {{ font-family: {SERIF}; font-size: 0.98rem; line-height: 1.55; color: {INK};
  max-width: 100ch; margin-top: 0.7rem; }}
.jr-cap b {{ font-weight: 700; }}

.jr-table {{ border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums;
  font-size: 0.95rem; margin: 1rem 0 1rem; }}
.jr-table caption {{ caption-side: top; text-align: left; font-family: {SERIF};
  font-size: 0.95rem; color: {INK_2}; padding-bottom: 0.4rem; }}
.jr-table caption i {{ color: {INK}; }}
.jr-table th, .jr-table td {{ padding: 0.32rem 0.6rem 0.32rem 0; text-align: left; color: {INK};
  border: 0 !important; background: transparent !important; }}
.jr-table tr {{ border: 0; background: transparent !important; }}
.jr-table th:last-child, .jr-table td:last-child {{ padding-right: 0; }}
.jr-table .num {{ text-align: right; }}
.jr-table thead tr {{ border-top: 2px solid {INK}; border-bottom: 1px solid {INK}; }}
.jr-table tbody tr:last-child {{ border-bottom: 2px solid {INK}; }}
.jr-table th {{ font-weight: 700; }}
.jr-table .delta {{ color: {INK_3}; font-size: 0.85rem; margin-left: 0.35rem; }}

.jr-remark {{ font-family: {SERIF}; font-size: 1.02rem; line-height: 1.6; color: {INK};
  border-left: 3px solid {BLUE}; padding: 0.1rem 0 0.1rem 1rem; margin: 0.6rem 0 1.1rem;
  max-width: 96ch; }}
.jr-remark .lead {{ font-weight: 700; font-style: italic; }}
.jr-remark.result {{ border-left-color: {GREEN}; }}
.jr-remark.result .lead {{ color: {GREEN}; }}
.jr-remark.caution {{ border-left-color: {RED}; }}
.jr-remark.caution .lead {{ color: {RED}; }}
.jr-remark.note {{ border-left-color: {INK_3}; }}
.jr-remark code, .jr-cap code {{ font-size: 0.88em; }}

.jr-note {{ font-family: {SERIF}; font-style: italic; color: {INK_2}; font-size: 0.95rem; line-height: 1.5; }}
.jr-rule {{ border: 0; border-top: 1px solid {HAIRLINE}; margin: 1.6rem 0 1.2rem; }}
</style>
"""


def apply_style() -> None:
    """Inject the stylesheet. Call once, right after `st.set_page_config`."""
    st.markdown(_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------

_CODE = re.compile(r"`([^`]+)`")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])")


def inline(text: str) -> str:
    """
    Escape `text` for HTML and apply the inline markdown the app's prose
    uses (`code`, **bold**, *italic*).

    The journal elements are raw HTML, and Streamlit does not run markdown
    inside an HTML block, so the few inline styles are converted here.
    """
    out = html.escape(text, quote=False)
    out = _CODE.sub(r"<code>\1</code>", out)
    out = _BOLD.sub(r"<strong>\1</strong>", out)
    out = _ITALIC.sub(r"<em>\1</em>", out)
    return out.replace("\n\n", "<br><br>")


def _html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def title(text: str, subtitle: str) -> None:
    st.title(text)
    _html(f'<p class="jr-subtitle">{inline(subtitle)}</p>')


def lede(text: str) -> None:
    """An introductory paragraph, set in the serif."""
    _html(f'<p class="jr-lede">{inline(text)}</p>')


def note(text: str) -> None:
    """A small italic aside (replaces grey `st.caption` lines)."""
    _html(f'<p class="jr-note">{inline(text)}</p>')


def rule() -> None:
    _html('<hr class="jr-rule">')


_REMARK_KINDS = {
    "remark": "Remark.",
    "result": "Result.",
    "caution": "Caution.",
    "note": "Note.",
}


def remark(text: str, kind: str = "remark", lead: str | None = None) -> None:
    """
    A "Remark." paragraph with a coloured rule, in place of `st.info` and
    friends. `kind` picks the rule colour: remark (blue), result (green),
    caution (red), note (grey).
    """
    label = lead or _REMARK_KINDS[kind]
    _html(
        f'<div class="jr-remark {kind}"><span class="lead">{html.escape(label)}</span> '
        f"{inline(text)}</div>"
    )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _to_unit(array: np.ndarray, stretch: bool) -> np.ndarray:
    """
    Map an image to [0, 1] for display.

    `stretch=False` pins the range to [0, 1] so brightness is comparable
    between panels -- "the reconstruction is darker than the original" is a
    real finding, not a display artifact. `stretch=True` rescales min..max.
    """
    data = np.asarray(array, dtype=np.float64)
    if stretch:
        low, high = float(data.min()), float(data.max())
        return (data - low) / (high - low) if high > low else np.zeros_like(data)
    return np.clip(data, 0.0, 1.0)


def _png_uri(array: np.ndarray) -> str:
    """
    Encode a [0, 1] grey or RGB image as a PNG data URI.

    PNG, never JPEG: the app is about the artifacts that appear when high
    frequencies are discarded, and a lossy codec would add a fake layer of
    exactly that effect.
    """
    pixels = (np.clip(array, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def kspace_display(kspace: np.ndarray) -> np.ndarray:
    """log(1 + |K|), normalised: the only way k-space's dynamic range is visible."""
    log_magnitude = np.log1p(np.abs(kspace))
    peak = float(log_magnitude.max())
    return log_magnitude / peak if peak > 0 else log_magnitude


def panel(array: np.ndarray, label: str, stretch: bool = False) -> dict:
    """One figure panel: an image array and its short label."""
    return {"array": array, "label": label, "stretch": stretch}


def figure(
    panels: Sequence[dict],
    caption: str | None = None,
    number: str | None = None,
    columns: int | None = None,
    first_letter: str = "a",
) -> None:
    """
    Lettered image panels in one row, with an optional numbered caption.

    Each panel gets a letter chip, "(a)", "(b)", ..., in its top-left
    corner and its label underneath. `first_letter` lets a figure split
    across Streamlit columns keep one continuous lettering.
    """
    columns = columns or len(panels)
    start = ord(first_letter)
    cells = []
    for index, item in enumerate(panels):
        data = _to_unit(item["array"], item["stretch"])
        letter = chr(start + index)
        cells.append(
            '<figure class="jr-panel">'
            f'<div class="jr-img"><img src="{_png_uri(data)}" alt="{html.escape(item["label"])}">'
            f'<span class="jr-letter">({letter})</span></div>'
            f'<figcaption>{inline(item["label"])}</figcaption></figure>'
        )
    caption_html = ""
    if caption:
        prefix = f"<b>Figure {number}.</b> " if number else ""
        caption_html = f'<div class="jr-cap">{prefix}{inline(caption)}</div>'
    _html(
        f'<div class="jr-fig"><div class="jr-grid" '
        f'style="grid-template-columns:repeat({columns},minmax(0,1fr))">'
        f'{"".join(cells)}</div>{caption_html}</div>'
    )


def caption(text: str, number: str) -> None:
    """A figure caption on its own, for a figure assembled from columns."""
    _html(f'<div class="jr-cap"><b>Figure {number}.</b> {inline(text)}</div>')


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def table(
    header: Sequence[str],
    rows: Iterable[Sequence[str]],
    caption: str | None = None,
    number: str | None = None,
    numeric: Sequence[bool] | None = None,
) -> None:
    """
    A three-rule ("booktabs") table: thick rule above the header, thin rule
    below it, thick rule at the bottom, no vertical lines. Cells are
    pre-formatted strings; `numeric` right-aligns the columns it marks.
    """
    numeric = numeric or [False] + [True] * (len(header) - 1)
    cls = lambda i: ' class="num"' if numeric[i] else ""
    head = "".join(f"<th{cls(i)}>{inline(h)}</th>" for i, h in enumerate(header))
    body = "".join(
        "<tr>" + "".join(f"<td{cls(i)}>{cell}</td>" for i, cell in enumerate(row)) + "</tr>"
        for row in rows
    )
    cap = ""
    if caption:
        prefix = f"<i>Table {number}.</i> " if number else ""
        cap = f"<caption>{prefix}{inline(caption)}</caption>"
    _html(f'<table class="jr-table">{cap}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>')


def dataframe_table(
    frame: pd.DataFrame,
    formats: dict[str, str],
    caption: str | None = None,
    number: str | None = None,
    index: bool = False,
) -> None:
    """Render a DataFrame as a three-rule table, formatting each column."""
    columns = list(frame.columns)
    header = ([""] if index else []) + [c[:1].upper() + c[1:] for c in columns]
    rows = []
    for label, record in frame.iterrows():
        cells = [html.escape(str(label))] if index else []
        for column in columns:
            value = record[column]
            fmt = formats.get(column)
            cells.append(fmt.format(value) if fmt else html.escape(str(value)))
        rows.append(cells)
    numeric = ([False] if index else []) + [column in formats for column in columns]
    table(header, rows, caption=caption, number=number, numeric=numeric)


def format_psnr(value: float) -> str:
    # A reduced-FOV reconstruction inside its own box can be exact, which
    # makes PSNR infinite (log of a zero error). That is a real result, not
    # a failure, so it is labelled rather than formatted into "inf dB".
    return "exact (∞)" if not np.isfinite(value) else f"{value:.2f} dB"


def metrics_table(
    scores: dict,
    baseline: dict | None = None,
    caption: str | None = None,
    number: str | None = None,
    extra: Sequence[tuple[str, str]] = (),
) -> None:
    """
    PSNR and SSIM as a small three-rule table, optionally with the change
    against a baseline (e.g. zero-filling) in a smaller grey figure.
    """
    psnr_delta = ssim_delta = ""
    if baseline is not None:
        if np.isfinite(scores["psnr"]) and np.isfinite(baseline["psnr"]):
            psnr_delta = f'<span class="delta">({scores["psnr"] - baseline["psnr"]:+.2f})</span>'
        ssim_delta = f'<span class="delta">({scores["ssim"] - baseline["ssim"]:+.4f})</span>'
    rows = [
        ("PSNR", format_psnr(scores["psnr"]) + psnr_delta),
        ("SSIM", f'{scores["ssim"]:.4f}' + ssim_delta),
        *extra,
    ]
    table(["Metric", "Value"], rows, caption=caption, number=number)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


def _style(chart: alt.Chart, height: int) -> alt.Chart:
    """Journal-style axes: framed plot, light grid, large projector-legible labels."""
    return (
        chart.properties(width="container", height=height)
        .configure(font=SANS, background=PAPER)
        .configure_view(stroke=INK_2, strokeWidth=1)
        .configure_axis(
            labelFontSize=14, titleFontSize=15, titleFontWeight="normal",
            labelColor=INK, titleColor=INK, domain=False,
            gridColor="#E2E8F0", tickColor=INK_2, titlePadding=10,
        )
        .configure_legend(
            orient="bottom", title=None, labelFontSize=14, labelColor=INK,
            symbolSize=110, symbolStrokeWidth=2, columnPadding=24, labelLimit=320,
        )
    )


def line_chart(
    frame: pd.DataFrame,
    x: str,
    y: str,
    series: str | None = None,
    x_title: str | None = None,
    y_title: str | None = None,
    log2_x: bool = False,
    reference: tuple[float, str] | None = None,
    points: bool = True,
    height: int = 320,
    series_order: Sequence[str] | None = None,
) -> None:
    """
    A line chart in the journal style.

    `series` names a long-format column that splits the data into lines;
    each line gets its own colour, marker and dash pattern. `reference`
    draws a labelled dashed horizontal rule, e.g. a zero-fill baseline.
    """
    x_scale = alt.Scale(type="log", base=2) if log2_x else alt.Scale(zero=False, nice=False)
    # On a log2 axis the ticks sit on the nominal doublings; the measured
    # ratios (24.91% for a radial mask, say) are close to them but not equal.
    ticks = None
    if log2_x:
        low, high = float(frame[x].min()), float(frame[x].max())
        ticks = [v for v in (1.5625, 3.125, 6.25, 12.5, 25, 50, 100) if low * 0.8 <= v <= high * 1.2]
    x_enc = alt.X(f"{x}:Q", title=x_title or x, scale=x_scale,
                  axis=alt.Axis(values=ticks, format="~g") if log2_x else alt.Axis())
    y_enc = alt.Y(f"{y}:Q", title=y_title or y, scale=alt.Scale(zero=False))

    if series:
        domain = list(series_order) if series_order else list(dict.fromkeys(frame[series]))
        n = len(domain)
        color = alt.Color(f"{series}:N", scale=alt.Scale(domain=domain, range=SERIES_COLORS[:n]))
        dash = alt.StrokeDash(f"{series}:N", scale=alt.Scale(domain=domain, range=SERIES_DASHES[:n]))
        shape = alt.Shape(f"{series}:N", scale=alt.Scale(domain=domain, range=SERIES_SHAPES[:n]))
        base = alt.Chart(frame).encode(x=x_enc, y=y_enc, color=color)
        layers = [base.mark_line(strokeWidth=2.4).encode(strokeDash=dash)]
        if points:
            layers.append(base.mark_point(size=90, filled=False, fill=PAPER, fillOpacity=1, strokeWidth=2.2).encode(shape=shape))
    else:
        base = alt.Chart(frame).encode(x=x_enc, y=y_enc)
        layers = [base.mark_line(strokeWidth=2.4, color=BLUE)]
        if points:
            layers.append(base.mark_point(size=60, filled=False, fill=PAPER, fillOpacity=1, strokeWidth=2, color=BLUE))

    if reference is not None:
        value, label = reference
        ref = pd.DataFrame({"y": [value], "label": [label]})
        layers.append(alt.Chart(ref).mark_rule(color=INK_2, strokeDash=[6, 4], strokeWidth=1.6).encode(y="y:Q"))
        layers.append(
            alt.Chart(ref).mark_text(align="left", dy=-8, fontSize=14, color=INK_2, font=SANS)
            .encode(y="y:Q", x=alt.value(8), text="label:N")
        )

    st.altair_chart(_style(alt.layer(*layers), height), theme=None)
