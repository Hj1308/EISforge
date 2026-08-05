"""Shared visual tokens for EISForge charts and pages.

Single source of truth for the colour values that the Plotly figures and the
hand-written CSS ``:root`` block in ``app.py`` both depend on.

Three places describe one design system, and
``tests/test_visualization_consistency.py`` asserts they agree:

1. This module            -> charts (PLOTLY_LAYOUT, trace palette)
2. ``app.py`` :root block -> custom CSS components (the human design doc)
3. ``.streamlit/config.toml`` -> native Streamlit widgets (TOML cannot import
   Python, so it is a literal mirror locked by the same test)

Note on native chart colours: Streamlit's ``chartCategoricalColors`` (and the
Sequential/Diverging variants) only recolour charts built on the Streamlit
template. This app forces ``template="plotly_white"`` and sets explicit trace
colours, so those options would be a fourth, unreachable copy. If a native
chart (``st.line_chart`` etc.) is ever added, set
``chartCategoricalColors = CHART_PALETTE`` in ``.streamlit/config.toml`` at
the same time.
"""

# ── core tokens (mirror of app.py :root; locked by the consistency test) ──
BG = "#ffffff"       # app background / chart paper
SURFACE = "#f7f7fb"  # cards, panels, chart plot area
TEXT = "#18162a"     # body and chart text
MUTED = "#76738a"
ACCENT = "#6d28d9"   # brand purple — also the first trace colour
GREEN = "#059669"
WARN = "#b45309"
DANGER = "#dc2626"

# ── fonts ───────────────────────────────────────────────────────────────────
# Plus Jakarta Sans is loaded by every page's CSS @import and matches the UI
# body, so charts sit on the same typeface as the app (Inter was never loaded).
FONT_FAMILY = "Plus Jakarta Sans"


# ── categorical trace palette (CVD-safe, greyscale-printable) ───────────────
# Okabe-Ito hues with the brand accent first. Ordered so that no adjacent pair
# is the two closest-luminance colours and no adjacent pair collapses under
# deuteranopia/protanopia/tritanopia (Vischeck model). Both adjacency
# properties are asserted in the consistency test, so a future reorder cannot
# silently undo the greyscale-print guarantee.
CHART_PALETTE = [
    ACCENT,     # purple     L=0.098
    "#E69F00",  # orange     L=0.416
    "#0072B2",  # blue       L=0.152
    "#56B4E9",  # sky        L=0.405
    "#D55E00",  # vermilion  L=0.222
    "#CC79A7",  # rose       L=0.293
    "#009E73",  # green      L=0.257
]

# Second encoding channels for multi-series figures. Colour alone must never
# be the only difference between traces: series_style() cycles dash / symbol
# automatically, so the distinction holds no matter how many series are added
# later.
CHART_DASHES = ["solid", "dash", "dot", "dashdot"]
CHART_SYMBOLS = [
    "circle-open", "square-open", "diamond-open", "triangle-up-open",
    "cross-open", "x", "hexagon-open", "star-open",
]


def series_style(i, *, line_width=2, marker_size=8):
    """Style the i-th series so colour is never the only differentiator.

    Returns ``line`` and ``marker`` dicts; colour comes from CHART_PALETTE
    and line dash / marker symbol cycle independently, so a figure stays
    distinguishable in greyscale and for colour-vision-deficient readers
    regardless of how many series are added later.
    """
    return {
        "line": {
            "color": CHART_PALETTE[i % len(CHART_PALETTE)],
            "dash": CHART_DASHES[i % len(CHART_DASHES)],
            "width": line_width,
        },
        "marker": {
            "color": CHART_PALETTE[i % len(CHART_PALETTE)],
            "symbol": CHART_SYMBOLS[i % len(CHART_SYMBOLS)],
            "size": marker_size,
        },
    }


def rgba(color, alpha):
    """Return ``color`` (hex) as an rgba() string at the given opacity."""
    c = color.lstrip("#")
    r, g, b = (int(c[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


# ── shared Plotly layout (built from the tokens, never from literals) ───────
PLOTLY_LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor=BG,
    plot_bgcolor=SURFACE,
    font=dict(family=FONT_FAMILY, color=TEXT),
    margin=dict(l=60, r=20, t=50, b=50),
    colorway=CHART_PALETTE,
)
