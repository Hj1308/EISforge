"""Consistency of the shared visual tokens across their three mirrors.

theme.py               -> charts (PLOTLY_LAYOUT, trace palette)
app.py :root block     -> custom CSS components (the human design doc)
.streamlit/config.toml -> native Streamlit widgets (TOML cannot import Python,
                          so it is a literal mirror)

These three describe one design system. If they drift, charts stop matching
the UI — exactly the issue-3 bug (#f8f9fa in Python vs #f7f7fb in CSS). These
tests turn that drift into a loud pytest failure instead of a silent visual
regression. The palette tests also guard the properties manuscripts depend on:
greyscale-printable series order and colour-vision-deficiency-safe adjacency.
"""

import math
import re
import tomllib
from pathlib import Path

from eisforge.visualization.theme import (
    ACCENT, BG, DANGER, GREEN, PLOTLY_LAYOUT, SURFACE, TEXT, WARN,
    CHART_PALETTE,
)

ROOT = Path(__file__).resolve().parents[1]

# Tokens that must agree across all three sources.
SHARED = {
    "--bg": BG,
    "--surface": SURFACE,
    "--text": TEXT,
    "--accent": ACCENT,
    "--green": GREEN,
    "--warn": WARN,
    "--danger": DANGER,
}


def _css_root_tokens():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    m = re.search(r":root\{([^}]*)\}", src)
    assert m, "no :root{...} block found in app.py"
    return {
        k: v.lower()
        for k, v in re.findall(r"(--[\w-]+):\s*(#[0-9a-fA-F]{6})", m.group(1))
    }


def _config_theme():
    cfg = (ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    return tomllib.loads(cfg)["theme"]


def test_css_root_matches_theme_module():
    css = _css_root_tokens()
    for var, value in SHARED.items():
        assert css[var] == value.lower(), (
            f"app.py CSS {var} = {css[var]} but theme.py = {value.lower()} — "
            "edit both: the CSS block is the human design doc, theme.py the "
            "machine one, and the test is the contract between them"
        )


def test_config_toml_matches_theme_module():
    theme = _config_theme()
    assert theme["primaryColor"].lower() == ACCENT.lower()
    assert theme["backgroundColor"].lower() == BG.lower()
    assert theme["secondaryBackgroundColor"].lower() == SURFACE.lower()
    assert theme["textColor"].lower() == TEXT.lower()


def test_plotly_layout_built_from_tokens():
    assert PLOTLY_LAYOUT["paper_bgcolor"] == BG
    assert PLOTLY_LAYOUT["plot_bgcolor"] == SURFACE  # the issue-3 regression
    assert PLOTLY_LAYOUT["font"]["family"] == "Plus Jakarta Sans"
    assert PLOTLY_LAYOUT["colorway"] == CHART_PALETTE


def test_palette_first_entry_is_accent():
    assert len(CHART_PALETTE) >= 5
    assert CHART_PALETTE[0].lower() == ACCENT.lower()


def test_no_stale_chart_literals_in_active_files():
    # Guards against reverting to the pre-theme hardcoded values.
    stale = [
        "#2563eb", "#7c3aed", "#f8f9fa", "#1e293b",
        "rgba(37,99,235", "family=\"Inter\"", "qualitative",
    ]
    for path in ("app.py", "pages/batch_eis.py", "pages/band_edge.py"):
        src = (ROOT / path).read_text(encoding="utf-8")
        for token in stale:
            assert token not in src, (
                f"stale chart literal {token!r} still in {path} — "
                "migrate it to the shared tokens in eisforge/visualization/theme.py"
            )


def _rel_luminance(hexcol):
    r, g, b = (int(hexcol.lstrip("#")[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

    def _lin(c):
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)


def test_palette_adjacent_pairs_greyscale_separable():
    # If a future reorder puts the two closest-luminance colours adjacent in
    # the sequence, a 5-series plot lands on them consecutively and greyscale
    # print becomes ambiguous. Guard the gap between every adjacent pair.
    lums = [_rel_luminance(c) for c in CHART_PALETTE]
    for a, b in zip(lums, lums[1:]):
        assert abs(a - b) >= 0.03, (
            "adjacent palette entries too close in greyscale luminance "
            f"({a:.3f} vs {b:.3f}) — reorder CHART_PALETTE"
        )


# Standard Vischeck linearized dichromacy matrices.
_VISCHECK = {
    "protanopia": ((0.56667, 0.43333, 0.0),
                   (0.55833, 0.44167, 0.0),
                   (0.0, 0.24167, 0.75833)),
    "deuteranopia": ((0.625, 0.375, 0.0),
                     (0.7, 0.3, 0.0),
                     (0.0, 0.3, 0.7)),
    "tritanopia": ((0.95, 0.05, 0.0),
                   (0.0, 0.43333, 0.56667),
                   (0.0, 0.475, 0.525)),
}


def _rgb(hexcol):
    c = hexcol.lstrip("#")
    return tuple(int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def _simulate(rgb, mat):
    return tuple(sum(mat[i][j] * rgb[j] for j in range(3)) for i in range(3))


def test_palette_adjacent_pairs_cvd_separable():
    # Guard the colourblind-safe ordering (Okabe-Ito hues + accent first) so
    # adjacent series never collapse under common CVD, under the Vischeck model.
    for a, b in zip(CHART_PALETTE, CHART_PALETTE[1:]):
        for name, mat in _VISCHECK.items():
            d = math.dist(_simulate(_rgb(a), mat), _simulate(_rgb(b), mat))
            assert d >= 0.20, (
                f"adjacent palette pair {a}~{b} collapses under {name} "
                f"(d={d:.3f}) — reorder CHART_PALETTE"
            )
