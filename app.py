"""
Assignment 2: Gradio dashboard (bare-head / helmet detection on construction sites).

Run with:  uv run python main.py app

Never trains anything. It loads:
  weights/best.pt             the committed finetuned model (only to score uploaded images)
  artifacts/results.json      metrics, per-class numbers, chosen thresholds, sensitivity
  artifacts/test_scores.csv   one score per test image (max bare-head confidence) + ground truth

One "case" = one image. Positive = the image contains at least one bare head (`head`).
Decision per case: score < t_low -> PASS, score >= t_high -> FAIL, in between -> HUMAN REVIEW.
"""

import csv
import json
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
WEIGHTS = ROOT / "weights" / "best.pt"

# Operations assumptions (assignment step 7). Editable in the UI.
CASES_PER_DAY = 480       # ~10 cameras x 1 image / 10 min x 8 h shift
MINUTES_PER_REVIEW = 1.0  # one human look at one case
PRODUCTIVE_HOURS = 6.0    # productive review hours per person per day

RES = json.loads((ARTIFACTS / "results.json").read_text())
COSTS = RES["costs_eur"]
C_TP, C_TN = COSTS["C_TP"], COSTS["C_TN"]
CLASSES = {int(k): v for k, v in RES["dataset"]["classes"].items()}
POSITIVE = RES["dataset"]["positive_class"]
POSITIVE_ID = next(k for k, v in CLASSES.items() if v == POSITIVE)
POLICY = RES["chosen_review_policy"]
SHIPPED_SINGLE = RES["chosen_single_threshold"]["threshold"]

with open(ARTIFACTS / "test_scores.csv") as f:
    _rows = list(csv.DictReader(f))
LABELS = np.array([int(r["has_bare_head"]) for r in _rows], dtype=bool)
SCORES = np.array([float(r["score_finetuned"]) for r in _rows])
POS, NEG = SCORES[LABELS], SCORES[~LABELS]
GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)

_model = None


def get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO

        _model = YOLO(str(WEIGHTS))
    return _model


def order(t_low, t_high):
    return float(t_low), float(max(t_high, t_low))


# ------------------------------------------------------------ prediction tab --
def predict(image, show_conf, t_low, t_high):
    if image is None:
        return None, "Upload an image to get a decision.", None
    t_low, t_high = order(t_low, t_high)
    bgr = np.ascontiguousarray(image[..., ::-1])  # Gradio gives RGB, Ultralytics expects BGR
    r = get_model().predict(bgr, conf=0.001, iou=0.6, imgsz=640, max_det=100, verbose=False)[0]
    conf = r.boxes.conf.cpu().numpy()
    cls = r.boxes.cls.cpu().numpy().astype(int)
    head = conf[cls == POSITIVE_ID]
    score = float(head.max()) if len(head) else 0.0

    if score < t_low:
        decision = "PASS"
    elif score >= t_high:
        decision = "FAIL"
    else:
        decision = "HUMAN REVIEW"

    keep = np.where(conf >= show_conf)[0]
    drawn = np.ascontiguousarray(r[keep.tolist()].plot()[..., ::-1]) if len(keep) else image
    names = r.names
    table = pd.DataFrame({
        "class": [names[int(c)] for c in cls[keep]],
        "confidence": np.round(conf[keep], 3),
    })
    text = (f"### Decision: **{decision}**\n"
            f"Bare-head score for this image: **{score:.2f}** "
            f"(pass < {t_low:.2f} <= review < {t_high:.2f} <= fail).")
    return drawn, text, table


# -------------------------------------------------------------- costing maths --
def split_counts(x, lo, hi):
    return np.array([(x < lo).sum(), ((x >= lo) & (x < hi)).sum(), (x >= hi).sum()])


def expected_cost(lo, hi, c_fn, c_fp, c_rev, prev):
    """Expected cost per 1,000 cases at base rate `prev`, using test-split rates."""
    p = split_counts(POS, lo, hi) / len(POS)
    n = split_counts(NEG, lo, hi) / len(NEG)
    cost = 1000 * (prev * (p[0] * c_fn + p[1] * c_rev + p[2] * C_TP)
                   + (1 - prev) * (n[0] * C_TN + n[1] * c_rev + n[2] * c_fp))
    share = prev * p[1] + (1 - prev) * n[1]
    return float(cost), float(share)


def single_curve(c_fn, c_fp, prev):
    tpr = np.array([(POS >= t).mean() for t in GRID])
    fpr = np.array([(NEG >= t).mean() for t in GRID])
    return 1000 * (prev * ((1 - tpr) * c_fn + tpr * C_TP) + (1 - prev) * (fpr * c_fp + (1 - fpr) * C_TN))


def update(t_low, t_high, c_fn, c_fp, c_rev, prev_pct, cases, minutes, hours):
    raised = float(t_high) < float(t_low)
    t_low, t_high = order(t_low, t_high)
    prev = float(prev_pct) / 100.0
    cases, minutes, hours = float(cases), float(minutes), max(float(hours), 0.1)

    # confusion matrix on the test split: actual (rows) x decision (columns)
    cm = np.vstack([split_counts(POS, t_low, t_high), split_counts(NEG, t_low, t_high)])
    fig_cm, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3), ["PASS", "REVIEW", "FAIL"])
    ax.set_yticks(range(2), ["Bare head present", "No bare head"])
    ax.set_xlabel("Decision")
    ax.set_title(f"Test split, {len(SCORES)} images")
    for i in range(2):
        for j in range(3):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    fig_cm.tight_layout()
    plt.close(fig_cm)

    cost, share = expected_cost(t_low, t_high, c_fn, c_fp, c_rev, prev)
    curve = single_curve(c_fn, c_fp, prev)
    best_i = int(np.argmin(curve))
    review_cases = share * cases
    review_hours = review_cases * minutes / 60.0
    people = review_hours / hours
    missed, false_fail = int(cm[0, 0]), int(cm[1, 2])

    summary = (
        f"**Expected cost per 1,000 cases: {cost:,.0f} EUR** "
        f"(at an assumed base rate of {prev * 100:.1f}%)\n\n"
        f"- Share sent to a person: **{share * 100:.1f}%** "
        f"-> {review_cases:.0f} cases/day, {review_hours:.1f} h/day, **{people:.2f} person(s)**\n"
        f"- On the test split: {missed} of {int(cm[0].sum())} violations passed unseen, "
        f"{false_fail} of {int(cm[1].sum())} compliant cases failed automatically\n"
        f"- Cost per day at {cases:.0f} cases/day: **{cost * cases / 1000:,.0f} EUR**"
        + ("\n\n*t_high was raised to t_low (t_low cannot exceed t_high).*" if raised else "")
    )

    strat = pd.DataFrame([
        ["Do nothing (always pass)", 1000 * prev * c_fn, 0.0],
        ["A person checks every case", 1000 * c_rev, cases],
        [f"Model alone (best single threshold = {GRID[best_i]:.2f})", float(curve[best_i]), 0.0],
        ["Model + human review (current thresholds)", cost, review_cases],
    ], columns=["Strategy", "Cost per 1,000 cases (EUR)", "Cases to a person per day"])
    strat["Cost per day (EUR)"] = strat["Cost per 1,000 cases (EUR)"] * cases / 1000
    strat = strat[["Strategy", "Cost per 1,000 cases (EUR)", "Cost per day (EUR)",
                   "Cases to a person per day"]].round(1)

    fig_c, ax = plt.subplots(figsize=(10, 4))
    ax.plot(GRID, curve, color="#1f4e9c", label="Model alone: cost vs. threshold")
    ax.scatter([GRID[best_i]], [curve[best_i]], color="#1f4e9c", zorder=5,
               label=f"Minimum under current costs: t = {GRID[best_i]:.2f}")
    shipped = float(curve[int(np.argmin(np.abs(GRID - SHIPPED_SINGLE)))])
    ax.scatter([SHIPPED_SINGLE], [shipped], marker="*", s=200, color="#d4a017", zorder=6,
               label=f"Chosen threshold (shipped): t = {SHIPPED_SINGLE:.2f}")
    ax.axhline(1000 * c_rev, color="gray", linestyle=":", label="A person checks everything")
    ax.axvline(t_low, color="#2e8b57", linestyle="--", label=f"t_low = {t_low:.2f}")
    ax.axvline(t_high, color="#c0392b", linestyle="--", label=f"t_high = {t_high:.2f}")
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Expected cost per 1,000 cases (EUR)")
    ax.set_ylim(0, min(float(curve.max()) * 1.05, 3 * max(1000 * c_rev, float(curve.min()), 1)))
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig_c.tight_layout()
    plt.close(fig_c)
    return fig_cm, summary, strat, fig_c


# ------------------------------------------------------------ static tables --
def per_class_table():
    rows = []
    for name, m in RES["box_level"]["per_class"].items():
        tag = f"{name} (positive)" if name == POSITIVE else name
        rows.append([tag, m["precision"], m["recall"], m["ap50"], m["map50_95"]])
    ov = RES["box_level"]["overall"]
    rows.append(["ALL CLASSES", None, None, ov["map50"], ov["map50_95"]])
    return pd.DataFrame(rows, columns=["Class", "Precision", "Recall", "AP50", "mAP50-95"]).round(3)


def dataset_table():
    st = RES["dataset"]["stats"]
    names = list(CLASSES.values())
    out = []
    for name in names:
        out.append([name] + [st[s]["instances"][name] for s in ("train", "valid", "test")]
                   + [st["test"]["median_box_area_share"][name]])
    out.append(["IMAGES"] + [st[s]["images"] for s in ("train", "valid", "test")] + [None])
    df = pd.DataFrame(out, columns=["Class", "Instances train", "Instances valid",
                                    "Instances test", "Median box area (share of image, test)"])
    return df


def sensitivity_table():
    df = pd.DataFrame(RES["sensitivity"])
    df = df.rename(columns={
        "c_fn": "C_FN (EUR)", "base_rate": "Base rate", "single_threshold": "Best single t",
        "single_cost_per_1000": "Model alone (EUR/1000)", "review_t_low": "t_low",
        "review_t_high": "t_high", "review_cost_per_1000": "Model + review (EUR/1000)",
        "review_share": "Share to person", "do_nothing_cost_per_1000": "Do nothing (EUR/1000)",
        "human_all_cost_per_1000": "Person checks all (EUR/1000)"})
    return df.round(3)


# ----------------------------------------------------------------------- UI --
with gr.Blocks(title="Bare-head detection: costed thresholds") as demo:
    gr.Markdown(
        "# Hard-hat compliance: costed pass / review / fail\n"
        f"Positive class: **{POSITIVE}** (a bare head = a missing helmet). "
        "One case = one image. The score of a case is the highest bare-head confidence in it. "
        "Move the two thresholds and everything below updates."
    )
    with gr.Row():
        t_low = gr.Slider(0, 1, value=POLICY["t_low"], step=0.01, label="t_low: PASS below this score")
        t_high = gr.Slider(0, 1, value=POLICY["t_high"], step=0.01, label="t_high: FAIL from this score")
    with gr.Accordion("Cost and operations assumptions (see REPORT.md)", open=False):
        with gr.Row():
            c_fn = gr.Number(COSTS["C_FN"], label="C_FN: missed violation (EUR)")
            c_fp = gr.Number(COSTS["C_FP"], label="C_FP: compliant case failed (EUR)")
            c_rev = gr.Number(COSTS["C_REVIEW"], label="C_REVIEW: one human review (EUR)")
            prev = gr.Number(COSTS["base_rate"] * 100, label="Base rate in production (%)")
        with gr.Row():
            cases = gr.Number(CASES_PER_DAY, label="Cases per day")
            minutes = gr.Number(MINUTES_PER_REVIEW, label="Minutes per review")
            hours = gr.Number(PRODUCTIVE_HOURS, label="Productive review hours per person per day")

    with gr.Tabs():
        with gr.Tab("Try an image"):
            with gr.Row():
                with gr.Column():
                    img_in = gr.Image(type="numpy", label="Upload an image")
                    show_conf = gr.Slider(0.05, 0.95, value=0.25, step=0.05,
                                          label="Draw boxes above this confidence")
                    go = gr.Button("Predict", variant="primary")
                with gr.Column():
                    img_out = gr.Image(label="Predictions")
                    decision = gr.Markdown()
                    det_table = gr.Dataframe(label="Detections drawn")
            pred_inputs = [img_in, show_conf, t_low, t_high]
            go.click(predict, pred_inputs, [img_out, decision, det_table])
            img_in.change(predict, pred_inputs, [img_out, decision, det_table])

        with gr.Tab("Confusion matrix and cost"):
            with gr.Row():
                cm_plot = gr.Plot(label="Confusion matrix (test split)")
                summary = gr.Markdown()
            strategies = gr.Dataframe(label="Strategy comparison")

        with gr.Tab("Cost vs threshold"):
            curve_plot = gr.Plot(label="Cost vs threshold")

        with gr.Tab("Per-class results"):
            gr.Markdown(
                "Box-level results on the **test split** (AP50 at IoU 0.5; precision and recall "
                "as reported by Ultralytics at its best-F1 operating point). "
                f"The headline number is the **{POSITIVE}** row, not the average."
            )
            gr.Dataframe(per_class_table(), label="Per-class results")
            cm_png = ARTIFACTS / "confusion_matrix.png"
            if cm_png.exists():
                gr.Image(str(cm_png), label="Box-level confusion matrix (Ultralytics)")
            gr.Dataframe(dataset_table(), label="Dataset: instances per class and split")

        with gr.Tab("Sensitivity"):
            gr.Markdown("How the thresholds, the cost and the review load move when C_FN "
                        "and the production base rate change (other costs as in the assumptions).")
            gr.Dataframe(sensitivity_table(), label="Sensitivity")

    inputs = [t_low, t_high, c_fn, c_fp, c_rev, prev, cases, minutes, hours]
    outputs = [cm_plot, summary, strategies, curve_plot]
    for comp in inputs:
        comp.change(update, inputs, outputs)
    demo.load(update, inputs, outputs)

demo.launch()
