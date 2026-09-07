"""Render the evaluation-metric formulas as PNGs for the briefing document.

    python scripts/make_formulas.py

python-docx cannot typeset mathematics, so each formula is rendered with
matplotlib's mathtext engine and embedded as an image. Keys here are referenced
by name from the document builder.

Outputs to outputs/plots/formulas/<key>.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config  # noqa: E402

INK = "#1c2430"

# mathtext is not full LaTeX: no \dfrac, \Big, \argmax, \operatorname or \text.
# Everything below sticks to the supported subset.
FORMULAS: dict[str, tuple[str, float]] = {
    "sigmoid": (
        r"$p_c \; = \; \sigma(z_c) \; = \; \frac{1}{1 + e^{-z_c}}"
        r"\qquad c = 1 \ldots 14$", 5.2),

    "bce": (
        r"$L \; = \; -\frac{1}{N} \sum_{i=1}^{N} \;\;\; \sum_{c=1}^{14}"
        r"\left[ w_c \, y_{ic} \log p_{ic} \; + \;"
        r"(1 - y_{ic}) \log (1 - p_{ic}) \right]$", 8.6),

    "posweight": (
        r"$w_c \; = \; \min \left( \frac{N_c^{-}}{N_c^{+}} , \; 20 \right)$", 4.8),

    "precision_recall": (
        r"$Precision_c = \frac{TP_c}{TP_c + FP_c}"
        r"\qquad \qquad Recall_c = \frac{TP_c}{TP_c + FN_c}$", 8.0),

    "f1": (
        r"$F_1 \; = \; 2 \cdot \frac{Precision \cdot Recall}"
        r"{Precision + Recall} \; = \; \frac{2 \, TP}{2 \, TP + FP + FN}$", 8.4),

    "tpr_fpr": (
        r"$TPR = \frac{TP}{TP + FN} \qquad \qquad"
        r"FPR = \frac{FP}{FP + TN}$", 7.4),

    "auroc": (
        r"$AUROC \; = \; \int_{0}^{1} TPR \, d(FPR)"
        r"\; = \; P \left( s^{+} > s^{-} \right)$", 7.6),

    "auprc": (
        r"$AUPRC \; = \; \sum_{n} \left( R_n - R_{n-1} \right) P_n$", 5.8),

    "lift": (
        r"$Lift_c = \frac{AUPRC_c}{\pi_c}"
        r"\qquad \qquad \pi_c = \frac{N_c^{+}}{N}$", 6.4),

    "youden": (
        r"$t_c^{*} \; = \; \max_{t} \; \left[ TPR_c(t) - FPR_c(t) \right]$", 6.6),

    "f1thresh": (
        r"$t_c^{*} \; = \; \max_{t} \; F_{1,c}(t)$", 4.6),

    "macro_micro": (
        r"$M_{macro} = \frac{1}{14} \sum_{c=1}^{14} M_c"
        r"\qquad \qquad"
        r"M_{micro} = M \left( \sum_c TP_c , \; \sum_c FP_c , \;"
        r"\sum_c FN_c \right)$", 9.6),

    "tta": (
        r"$p_i \; = \; \frac{1}{2} \left[ \sigma ( f(x_i) ) \; + \;"
        r"\sigma ( f( flip(x_i) ) ) \right]$", 7.0),

    "iou": (
        r"$IoU \; = \; \frac{| A_{cam} \cap A_{gt} |}"
        r"{| A_{cam} \cup A_{gt} |}$", 5.0),

    # ---- report-study metrics: F flagged, P present in report, T labels, Z zones given
    "rep_unsupported": (
        r"$U \; = \; | \, P \setminus F \, |"
        r"\qquad \qquad W \; = \; | \, P \setminus T \, |$", 6.4),

    "rep_omit": (
        r"$Omit \; = \; \frac{| \, F \setminus P \, |}{| \, F \, |}$", 4.2),

    "rep_truth": (
        r"$Precision_T = \frac{| P \cap T |}{| P |}"
        r"\qquad \qquad Recall_T = \frac{| P \cap T |}{| T |}$", 7.4),

    "rep_delta": (
        r"$\Delta U = \frac{1}{N} \sum_{i=1}^{N} \left( U_C^{(i)} - U_B^{(i)} \right)"
        r"\qquad \qquad"
        r"\Delta W = \frac{1}{N} \sum_{i=1}^{N} \left( W_C^{(i)} - W_B^{(i)} \right)$", 9.4),

    "rep_locagree": (
        r"$LocAgree \; = \; \frac{ n_{side \, and \, level \, match} }{ | \, P \cap Z \, | }$", 6.0),

    "rep_sharetrue": (
        r"$Share_{true} \; = \; \frac{ \sum_i | (P_i \setminus F_i) \cap T_i | }"
        r"{ \sum_i | P_i \setminus F_i | }$", 6.6),

    "rep_clean": (
        r"$Clean \; = \; P \left( \, | P | = 0 \;\; \middle| \;\; | T | = 0 \, \right)$", 5.6),
}


def render(key: str, tex: str, width: float, out_dir: Path) -> Path:
    fig = plt.figure(figsize=(width, 0.82))
    fig.text(0.5, 0.5, tex, ha="center", va="center", fontsize=15, color=INK)
    path = out_dir / f"{key}.png"
    fig.savefig(path, dpi=210, bbox_inches="tight", pad_inches=0.09,
                facecolor="white")
    plt.close(fig)
    return path


def main() -> None:
    cfg = load_config()
    out_dir = Path(cfg.dirs.plots) / "formulas"
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, (tex, width) in FORMULAS.items():
        render(key, tex, width, out_dir)
    print(f"rendered {len(FORMULAS)} formulas -> {out_dir}")


if __name__ == "__main__":
    main()
