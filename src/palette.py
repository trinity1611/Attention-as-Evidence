"""One fixed colour per finding, used everywhere in the app.

The same hue identifies a finding in the probability bars, its heatmap, its
contour, the report table and the legend, so a reader learns the mapping once.
Hues are drawn from tested categorical palettes (Tableau 20 / Okabe-Ito
territory) and are always paired with the finding's name -- colour never
carries meaning on its own, and it never encodes confidence or severity.
"""

from __future__ import annotations

from src.config import CLASSES

FINDING_COLOURS: dict[str, str] = {
    "Atelectasis":        "#4E79A7",   # steel blue
    "Cardiomegaly":       "#F28E2B",   # amber
    "Consolidation":      "#E15759",   # coral red
    "Edema":              "#76B7B2",   # teal
    "Effusion":           "#59A14F",   # green
    "Emphysema":          "#EDC948",   # gold
    "Fibrosis":           "#B07AA1",   # mauve
    "Hernia":             "#FF9DA7",   # pink
    "Infiltration":       "#9C755F",   # brown
    "Mass":               "#BAB0AC",   # warm grey
    "Nodule":             "#D37295",   # rose
    "Pleural_Thickening": "#86BCB6",   # sea green
    "Pneumonia":          "#F1CE63",   # sand
    "Pneumothorax":       "#8CD17D",   # lime
}
assert set(FINDING_COLOURS) == set(CLASSES)

# Report diagrams use a light editorial palette; the app uses a dark clinical
# workstation palette (navy ground, cyan accent, glowing film frames).
UI_LIGHT = {
    "bg": "#FAFAF7", "card": "#FFFFFF", "ink": "#1C2430", "muted": "#5B6673",
    "accent": "#2E4B73", "accent2": "#D1873C", "line": "#E6E3DC",
    "ok": "#1A7F37", "bad": "#B3261E",
}
UI = {
    "bg": "#0B1220", "bg2": "#0F1728", "card": "#131C2E", "card2": "#182338",
    "ink": "#E6ECF5", "muted": "#8A97AD", "faint": "#5B6880",
    "accent": "#4FD1FF", "accent2": "#F2A65A", "line": "#22304A",
    "ok": "#3DDC97", "bad": "#FF6B6B", "warn": "#F2A65A",
    "glow": "rgba(79,209,255,.35)",
}

FILM_TINTS = {
    "grayscale": None,
    "sepia": ((0.40, 0.30, 0.18), (1.00, 0.92, 0.72)),     # (shadow rgb, highlight rgb)
    "cyan":  ((0.06, 0.18, 0.32), (0.86, 0.97, 1.00)),
    "amber": ((0.36, 0.20, 0.05), (1.00, 0.85, 0.55)),
}


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def colour(finding: str) -> str:
    return FINDING_COLOURS[finding]


def pretty(finding: str) -> str:
    return finding.replace("_", " ")
