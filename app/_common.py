"""Shared setup for the Streamlit pages: paths, cached service, dark-theme CSS,
small widgets (KPI tiles, status chips, section headers, Plotly layout)."""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "app")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.config import CLASSES, load_config, read_csv  # noqa: E402
from src.palette import FINDING_COLOURS, UI, colour, pretty  # noqa: E402

CFG = load_config()
PLOTS = Path(CFG.dirs.plots)
METRICS = Path(CFG.dirs.metrics)
DIST = Path(CFG.dirs.distribution)
DISCLAIMER_SHORT = ("Research and educational use only. Not a medical device; "
                    "not for clinical diagnosis.")


@st.cache_resource(show_spinner="Loading the classifier and explanation engine…")
def service():
    from src.service import get_service
    return get_service()


def inject_css() -> None:
    st.markdown(f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600&display=swap');
      :root {{
        --bg:{UI['bg']}; --bg2:{UI['bg2']}; --card:{UI['card']}; --card2:{UI['card2']};
        --ink:{UI['ink']}; --muted:{UI['muted']}; --faint:{UI['faint']};
        --accent:{UI['accent']}; --accent2:{UI['accent2']}; --line:{UI['line']};
        --ok:{UI['ok']}; --bad:{UI['bad']}; --warn:{UI['warn']}; --glow:{UI['glow']};
      }}
      html, body, [class*="css"], .stApp {{ font-family: Inter, -apple-system, Segoe UI, Roboto, sans-serif; }}
      .stApp {{ background:
          radial-gradient(1200px 600px at 10% -10%, rgba(79,209,255,.08), transparent 60%),
          radial-gradient(900px 500px at 100% 0%, rgba(242,166,90,.06), transparent 60%),
          var(--bg); }}
      .block-container {{ padding-top: 1.2rem; max-width: 1320px; }}
      h1, h2, h3 {{ color: var(--ink); letter-spacing: -0.015em; font-weight: 650; }}
      h2 {{ font-size: 1.35rem; margin-top: 1.6rem; }}
      .eyebrow {{ color: var(--accent); font-size: .74rem; letter-spacing: .14em; text-transform: uppercase;
                  font-weight: 600; margin-bottom: .2rem; }}
      .cap {{ color: var(--muted); font-size: .86rem; line-height: 1.45; }}
      .mono {{ font-family: 'JetBrains Mono', ui-monospace, Menlo, monospace; font-variant-numeric: tabular-nums; }}

      .card {{ background: linear-gradient(180deg, var(--card2), var(--card)); border: 1px solid var(--line);
               border-radius: 16px; padding: 1.05rem 1.2rem; height: 100%;
               box-shadow: 0 1px 0 rgba(255,255,255,.03) inset, 0 10px 30px rgba(0,0,0,.25); }}
      .card h4 {{ margin: 0 0 .35rem 0; color: var(--ink); font-size: .98rem; font-weight: 600; }}
      .card .body {{ color: var(--muted); font-size: .88rem; line-height: 1.5; }}
      .kpi {{ font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: 1.95rem; font-weight: 600;
              color: var(--accent); line-height: 1.1; letter-spacing: -.02em; }}
      .kpi-sub {{ color: var(--muted); font-size: .8rem; margin-top: .15rem; }}

      .chip {{ display:inline-flex; align-items:center; gap:.4em; padding:.2em .7em; border-radius:999px;
               font-size:.82rem; font-weight:500; color:var(--ink); margin:.15em .25em .15em 0;
               background: rgba(255,255,255,.03); border:1.5px solid; }}
      .chip .dot {{ width:.62em; height:.62em; border-radius:50%; box-shadow: 0 0 10px currentColor; }}
      .status {{ display:inline-block; padding:.18em .65em; border-radius:6px; font-size:.76rem; font-weight:600;
                 letter-spacing:.04em; text-transform:uppercase; font-family:'JetBrains Mono', monospace; }}
      .status.ok {{ background: rgba(61,220,151,.12); color: var(--ok); border:1px solid rgba(61,220,151,.35); }}
      .status.bad {{ background: rgba(255,107,107,.12); color: var(--bad); border:1px solid rgba(255,107,107,.35); }}
      .status.warn {{ background: rgba(242,166,90,.12); color: var(--warn); border:1px solid rgba(242,166,90,.35); }}
      .status.info {{ background: rgba(79,209,255,.12); color: var(--accent); border:1px solid rgba(79,209,255,.35); }}

      .warn {{ background: rgba(242,166,90,.08); border:1px solid rgba(242,166,90,.35); color:#F7D7B5;
               border-radius:12px; padding:.6rem .9rem; font-size:.88rem; }}
      .note {{ background: rgba(79,209,255,.06); border:1px solid rgba(79,209,255,.25); color:#CFEFFF;
               border-radius:12px; padding:.6rem .9rem; font-size:.88rem; }}

      div[data-testid="stImage"] img {{ border-radius: 14px; border: 1px solid var(--line);
                                       box-shadow: 0 0 0 1px rgba(79,209,255,.08), 0 0 36px rgba(79,209,255,.10); }}
      div[data-testid="stDataFrame"] {{ border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }}
      .stDownloadButton button, .stButton button {{ border-radius: 999px; border:1px solid var(--line); }}
      .stButton button[kind="primary"] {{ background: linear-gradient(135deg, #4FD1FF, #2FA8DD); color:#06111F;
                                          border: none; font-weight:600; box-shadow: 0 0 24px var(--glow); }}
      section[data-testid="stSidebar"] {{ background: var(--bg2); border-right: 1px solid var(--line); }}
      hr {{ border-color: var(--line); }}
      footer {{ visibility: hidden; }}
    </style>""", unsafe_allow_html=True)


def eyebrow(text: str) -> None:
    st.markdown(f'<div class="eyebrow">{text}</div>', unsafe_allow_html=True)


def card(title: str, body_html: str = "", kpi: str | None = None, sub: str | None = None) -> None:
    kpi_html = f'<div class="kpi">{kpi}</div>' if kpi else ""
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    body = f'<div class="body">{body_html}</div>' if body_html else ""
    st.markdown(f'<div class="card"><h4>{title}</h4>{kpi_html}{sub_html}{body}</div>',
                unsafe_allow_html=True)


def chip(finding: str, prob: float | None = None) -> str:
    c = colour(finding)
    txt = pretty(finding) + (f' <span class="mono">{prob:.2f}</span>' if prob is not None else "")
    return (f'<span class="chip" style="border-color:{c}55;">'
            f'<span class="dot" style="background:{c};color:{c}"></span>{txt}</span>')


def status(text: str, kind: str = "info") -> str:
    return f'<span class="status {kind}">{text}</span>'


def plotly_layout(fig, height: int = 460, **kw):
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=UI["ink"], size=12),
        margin=dict(l=10, r=40, t=10, b=30), **kw)
    fig.update_xaxes(gridcolor=UI["line"], zerolinecolor=UI["line"], tickfont=dict(color=UI["muted"]),
                     title_font=dict(color=UI["muted"]))
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)", tickfont=dict(color=UI["ink"]))
    return fig


def b64_png(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def show_plot(name: str, caption: str = "", sub: str | None = None, width: str = "stretch") -> None:
    p = PLOTS / sub / name if sub else PLOTS / name
    if p.exists():
        st.image(str(p), caption=caption, width=width)
    else:
        st.info(f"Figure not found: {p.name}")


def footer() -> None:
    st.markdown(f'<hr style="margin:2rem 0 .6rem"><div class="cap">{DISCLAIMER_SHORT} &nbsp;·&nbsp; '
                f'Data: NIH ChestX-ray14 (Wang et al., CVPR 2017), CC0 &nbsp;·&nbsp; Classifier: DenseNet-121 · '
                f'Explanations: LayerCAM · Writer: MedGemma 4B via Ollama &nbsp;·&nbsp; Every number on these '
                f'pages is read from the repository\'s outputs.</div>', unsafe_allow_html=True)
