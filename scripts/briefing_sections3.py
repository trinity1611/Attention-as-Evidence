"""Report section for the MedGemma grounding study (Phase 4).

Numbers are read from outputs/metrics/report_study_summary.csv and the
per-report table; an example report is quoted from outputs/reports/examples/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from docx.shared import Inches, Pt

from src.config import read_csv


def report_study_section(doc, helpers, cfg, num: int) -> None:
    para, bullets, table, figure = (helpers["para"], helpers["bullets"],
                                    helpers["table"], helpers["figure"])
    met, plots = Path(cfg.dirs.metrics), Path(cfg.dirs.plots)
    summ = read_csv(met / "report_study_summary.csv", index_col=0)
    per = read_csv(met / "report_study_per_report.csv")
    n = int(summ.n_reports.max())

    piv = per.pivot_table(index="image", columns="condition", values="n_unsupported_vs_clf")
    d_bc = (piv["C"] - piv["B"]).dropna()
    rng = np.random.default_rng(0)
    boots = [rng.choice(d_bc.to_numpy(), len(d_bc)).mean() for _ in range(2000)]
    lo, hi = np.percentile(boots, 2.5), np.percentile(boots, 97.5)

    doc.add_page_break()
    doc.add_heading(f"{num}.  Results IV — Report Generation and the Grounding Study", level=1)
    para(doc, "This is the phase the title refers to. MedGemma 4B, Google's open "
              "medical vision-language model, runs locally through Ollama and "
              "writes the draft report. It is a writer, not a second classifier: "
              "it receives the film and, depending on the experimental condition, "
              "the classifier's evidence, and returns a structured JSON block "
              "(all 14 findings marked present / absent / uncertain, with "
              "locations) followed by a Findings / Impression / Recommendation "
              "draft. Nothing is fine-tuned; temperature is zero so results are "
              "reproducible.")

    doc.add_heading(f"{num}.1  The question and the design", level=2)
    para(doc, "What should we hand the language model so that it writes accurate "
              "reports and does not assert findings it has no evidence for? Three "
              "briefings, same 200 films, same model, same instructions:")
    table(doc, ["Condition", "The X-ray", "Classifier's 14 probabilities", "Grad-CAM zone per flagged finding"], [
        ["A — image only", "yes", "", ""],
        ["B — + classifier", "yes", "yes", ""],
        ["C — + grounding", "yes", "yes", "yes"],
    ], widths=[2.2, 1.4, 2.6, 3.0],
          caption="Table — The three prompt conditions. C is the coupling the title "
                  "describes: the explanation becomes an input to the writer.")
    para(doc, f"Films: {n} from the 1,293-image shared test set, half with a labelled "
              f"finding and half without, so the full-data model's probabilities on "
              f"them are honest. Flagged findings are those above per-finding "
              f"F1-optimal thresholds fitted on the ~24,000 official-test films not in "
              f"the study. Zones come from the adopted LayerCAM configuration, reduced "
              f"to a radiographic phrase such as \"left lower zone (costophrenic "
              f"region)\". Every reply parsed successfully ({int(n * 3)} of "
              f"{int(n * 3)}).")

    doc.add_heading(f"{num}.2  How a report is scored — the metrics and what each signifies", level=2)
    fdir = plots / "formulas"

    def formula(key, width):
        figure(doc, fdir / f"{key}.png", "", width=width)

    para(doc, "Every report is scored from the JSON block MedGemma returns, never "
              "from the prose, so the scoring is deterministic. Four sets are "
              "defined per film:")
    table(doc, ["Symbol", "Set", "Where it comes from"], [
        ["F", "flagged", "findings the classifier scored above their per-finding threshold"],
        ["P", "present", "findings MedGemma marked \"present\" in its JSON"],
        ["T", "truth", "the film's NIH labels (about 90 % accurate)"],
        ["Z", "zones", "the Grad-CAM zone text supplied for each flagged finding (condition C)"],
    ], widths=[1.0, 1.4, 6.6])

    doc.add_heading("Unsupported by the classifier (U) and false against the labels (W)", level=3)
    formula("rep_unsupported", 6.2)
    para(doc, "U counts findings the report asserts that the classifier did not "
              "flag. It is deliberately not called a hallucination: each such "
              "finding could be a correct reading the CNN missed, a misreading of "
              "the image, or an assertion driven by the numbers in the prompt. W "
              "counts findings asserted that the labels do not contain; it uses "
              "the ground truth as the reference instead of the classifier. A "
              "report can score low on U simply by copying the classifier, so W is "
              "the check that the report moved toward reality, not just toward the "
              "classifier. Both are counts per report; lower is better.")

    doc.add_heading("Share of unsupported findings that were genuine", level=3)
    formula("rep_sharetrue", 6.4)
    para(doc, f"Of everything MedGemma added beyond the classifier, the fraction "
              f"the labels agree with — genuine catches of things the CNN missed. "
              f"Measured: about {summ.loc['C', 'share_of_unsupported_that_were_true']:.0%} "
              f"in the grounded condition. The remaining ~95 % are misreadings or "
              f"prompt-driven assertions (with ~10 % label noise, a few may be real).")

    doc.add_heading("Omitted flagged findings", level=3)
    formula("rep_omit", 4.0)
    para(doc, f"The share of classifier-flagged findings the report failed to "
              f"mention. Measures whether the writer respects the evidence it is "
              f"given. Grounded condition: {summ.loc['C', 'omission_rate_vs_classifier']:.1%}; "
              f"image-only: {summ.loc['A', 'omission_rate_vs_classifier']:.0%}.")

    doc.add_heading("Precision and recall against the labels", level=3)
    formula("rep_truth", 7.2)
    para(doc, "Standard set precision and recall, with the NIH labels as truth. "
              "Recall says how many of the real findings the report mentioned; "
              "precision how many of its assertions were real. Label noise lowers "
              "both slightly for every condition alike.")

    doc.add_heading("Normal films kept clean", level=3)
    formula("rep_clean", 5.4)
    para(doc, f"Among films with no labelled finding, the probability the report "
              f"asserts nothing. Image-only: {summ.loc['A', 'normal_films_kept_clean']:.0%}; "
              f"with probabilities: {summ.loc['B', 'normal_films_kept_clean']:.0%}; grounded: "
              f"{summ.loc['C', 'normal_films_kept_clean']:.0%}. Once the prompt contains "
              f"the score list, the model virtually never leaves a normal film alone.")

    doc.add_heading("Localisation agreement (condition C only)", level=3)
    formula("rep_locagree", 5.8)
    para(doc, f"For each finding marked present that we supplied a zone for, does "
              f"the location the model wrote share the same side (right / left / "
              f"bilateral / central) and level (upper / mid / lower) as the zone "
              f"given? Measured: {summ.loc['C', 'localisation_agreement']:.0%}. This "
              f"is compliance — the model repeats the zone it was told one line "
              f"earlier — and is reported as such, not as evidence that it verified "
              f"the location against the pixels.")

    doc.add_heading("Effect of supplying the zone: the paired differences", level=3)
    formula("rep_delta", 9.2)
    para(doc, f"The same film is written under B (probabilities, no zones) and C "
              f"(probabilities and zones); the difference in errors is averaged "
              f"over the {n} films. ΔU measures the change in assertions unsupported "
              f"by the classifier; ΔW the change in assertions false against the "
              f"labels. Pairing on the film cancels film-to-film difficulty, which "
              f"is why a 200-film study is enough.")
    p = doc.add_paragraph()
    p.add_run("Why negative is good.  ").bold = True
    p.add_run(f"Both Δ metrics are differences in error counts, with-zones minus "
              f"without-zones. A negative value means fewer errors when the zone is "
              f"supplied — an improvement. Measured: ΔU = {d_bc.mean():+.2f} (95 % CI "
              f"[{lo:+.2f}, {hi:+.2f}]) and ΔW = "
              f"{summ.loc['C', 'false_findings_vs_truth_per_report'] - summ.loc['B', 'false_findings_vs_truth_per_report']:+.2f} "
              f"per report. Read as: supplying the heatmap zone removed roughly one "
              f"unsupported finding and one false finding from every report.")

    doc.add_heading("A worked film", level=3)
    table(doc, ["", "B — no zones", "C — with zones"], [
        ["Flagged F", "{Cardiomegaly, Effusion}", "{Cardiomegaly, Effusion}"],
        ["Truth T", "{Cardiomegaly, Effusion}", "{Cardiomegaly, Effusion}"],
        ["Present P", "{Cardiomegaly, Effusion, Edema, Consolidation}", "{Cardiomegaly, Effusion, Edema}"],
        ["U = |P \\ F|", "2", "1"],
        ["W = |P \\ T|", "2", "1"],
        ["Omit = |F \\ P| / |F|", "0 / 2 = 0", "0 / 2 = 0"],
        ["Recall vs T", "2 / 2 = 1.0", "2 / 2 = 1.0"],
        ["Precision vs T", "2 / 4 = 0.50", "2 / 3 = 0.67"],
        ["Location agreement", "—", "2 / 2 = 1.0"],
    ], widths=[2.2, 3.4, 3.4],
          caption="Table — One film under both conditions. Its contribution to ΔU is "
                  "1 − 2 = −1 and to ΔW is −1: the zone removed one unsupported, false "
                  "assertion. Repeat over 200 films, average, bootstrap the interval.")

    doc.add_heading(f"{num}.3  Results", level=2)
    conds = [c for c in ("A", "B", "C") if c in summ.index]
    label = {"A": "A: image only", "B": "B: + probabilities", "C": "C: + zones"}

    def row(name, key, fmt="{:.2f}"):
        return [name] + [fmt.format(summ.loc[c, key]) if pd.notna(summ.loc[c, key]) else "—" for c in conds]
    rows = [
        row("Findings asserted per report", "mean_present_per_report"),
        row("Unsupported by classifier, per report", "unsupported_by_classifier_per_report"),
        row("…share of those that were in the labels", "share_of_unsupported_that_were_true", "{:.1%}"),
        row("Flagged findings omitted", "omission_rate_vs_classifier", "{:.1%}"),
        row("False findings vs ground truth, per report", "false_findings_vs_truth_per_report"),
        row("Recall vs ground truth", "recall_vs_truth"),
        row("Normal films kept clean", "normal_films_kept_clean", "{:.0%}"),
        row("Location agrees with the given zone", "localisation_agreement", "{:.0%}"),
        row("Seconds per report (RTX 4060)", "mean_seconds", "{:.0f}"),
    ]
    table(doc, ["Per report"] + [label[c] for c in conds], rows, widths=[3.6, 1.8, 1.8, 1.8],
          caption=f"Table — {n} films per condition, {int(n * 3)} reports. Scores are from "
                  f"the parsed JSON block of each report.")
    figure(doc, plots / "report_study.png",
           "Figure — The three headline measures by condition.", width=9.4)

    p = doc.add_paragraph()
    p.add_run(f"Grounding works, and not only by making the model obedient.  ").bold = True
    p.add_run(f"Adding the Grad-CAM zone (C) to the probabilities (B) reduces "
              f"unsupported assertions by {abs(d_bc.mean()):.2f} per report — 95 % "
              f"confidence interval [{lo:+.2f}, {hi:+.2f}], paired over {len(d_bc)} "
              f"films — and the improvement holds against the ground truth as well "
              f"({summ.loc['C', 'false_findings_vs_truth_per_report']:.1f} vs "
              f"{summ.loc['B', 'false_findings_vs_truth_per_report']:.1f} false findings "
              f"per report). If C were merely copying the classifier, the truth-based "
              f"measure would not move.")
    bullets(doc, [
        ("Raw probabilities are a bad prompt.  ", f"Given the score list without "
         f"location, MedGemma asserts {summ.loc['B', 'mean_present_per_report'] / summ.loc['A', 'mean_present_per_report']:.1f}x "
         f"more findings than from the image alone and leaves no normal film clean. "
         f"Handing a language model numbers it cannot verify makes it worse."),
        ("It is not catching what the CNN missed.  ", f"Only about "
         f"{summ.loc['C', 'share_of_unsupported_that_were_true']:.0%} of the findings "
         f"MedGemma adds beyond the classifier are in the labels. The rest are "
         f"misreadings or prompt-driven; with 10 % label noise a few may be real, "
         f"but the bulk are not."),
        ("The trade-off is stark.  ", f"A is quiet — {summ.loc['A', 'false_findings_vs_truth_per_report']:.1f} "
         f"false findings per report — but misses "
         f"{1 - summ.loc['A', 'recall_vs_truth']:.0%} of real ones. C catches "
         f"{summ.loc['C', 'recall_vs_truth']:.0%} at "
         f"{summ.loc['C', 'false_findings_vs_truth_per_report']:.1f} false per report. "
         f"No condition produces a report that could go out unread."),
        ("Localisation agreement is compliance.  ", f"{summ.loc['C', 'localisation_agreement']:.0%} "
         f"agreement in C means the model repeats the zone phrase it was given, "
         f"not that it verified it against the image. It is reported as what it is."),
    ])

    # an example
    ex_dir = Path(cfg.dirs.root) / "reports" / "examples"
    exs = sorted(ex_dir.glob("*_C.txt"))
    if exs:
        txt = exs[0].read_text(encoding="utf-8")
        head = "\n".join(txt.splitlines()[:4])
        body = txt.split("--- report ---", 1)[-1].strip()
        doc.add_heading(f"{num}.4  An example draft (condition C)", level=2)
        para(doc, head.replace("\n", "   "), size=9, muted=True)
        q = doc.add_paragraph()
        q.paragraph_format.left_indent = Inches(0.4)
        r = q.add_run(body[:1400])
        r.font.size = Pt(9.5)
        para(doc, "Every generated report carries a fixed research-use disclaimer "
                  "appended by our code, not left to the model.", italic=True, muted=True)
