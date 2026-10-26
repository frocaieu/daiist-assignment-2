"""
Assignment 2: training pipeline (helmet / bare-head detection).

Run with:  uv run python main.py train      (or `python main.py train` on Colab)

Stages, with no manual steps:
  1. Download the pinned Roboflow Universe version (no fork).
  2. Describe the dataset (images per split, instances per class, box size).
  3. Finetune yolo11n from the COCO checkpoint; copy best weights to weights/.
  4. Per-class box-level results on the test split (P, R, AP50, mAP50-95,
     Ultralytics confusion matrix).
  5. Image-level scores on the test split for the finetuned model and for two
     baselines (COCO zero-shot, trivial). One "case" = one image; positive =
     the image contains at least one bare head (class `head`).
  6. Cost sweep over the confidence threshold, choose the cheapest, then the
     cheapest pair of thresholds (pass / human review / fail).
  7. Save everything app.py needs into artifacts/.

Cost figures live in one block below (see REPORT.md for the justification).
To recompute only the costs from saved test scores, without retraining:
    COSTS_ONLY=1 uv run python main.py train
"""

import csv
import json
import os
import shutil
from pathlib import Path

import numpy as np

from api_key import load_api_key

ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = ROOT / "weights"
ARTIFACTS = ROOT / "artifacts"
DATA_DIR = ROOT / "data"
RUNS_DIR = Path(os.environ.get("RUNS_DIR", ROOT / "runs"))  # point to Google Drive on Colab
CACHE_DIR = ROOT / "cache"  # Ultralytics downloads pretrained .pt files here

# ---------------------------------------------------------------- dataset --
WORKSPACE = "ppe-kit-detection"
PROJECT = "hardhat-safetyvest"
VERSION = 1
EXPORT_FORMAT = "yolov11"
POSITIVE_CLASS = "head"  # bare head = missing helmet = the violation

# --------------------------------------------------------------- training --
MODEL = "yolo11n.pt"
SEED = 0
IMGSZ = 640
EPOCHS = int(os.environ.get("EPOCHS", 25))
BATCH = int(os.environ.get("BATCH", 32))
PATIENCE = 8
SMOKE = os.environ.get("SMOKE") == "1"  # 1 epoch on 5% of the data, to test the pipeline

# ------------------------------------------------------------ costs (EUR) --
# Justified in REPORT.md. Every figure is an assumption to be defended; the
# sensitivity table (SENS_*) shows how the answer moves when they change.
C_FN = 100.0     # bare head not flagged: ~2,451 EUR minimum "grave" PRL fine (LISOS art. 40.2)
                 # x an assumed 4% chance a missed violation ends in a sanction (accident costs excluded)
C_FP = 5.0       # helmeted head wrongly FAILED (automatic action: foreman sent / work interrupted):
                 # ~10 min of a foreman or technician at ~30 EUR/h (assumed). Must be > C_REVIEW,
                 # otherwise failing is never worse than sending a case to review.
C_TP = 0.0       # violation correctly flagged (cost of acting on it is ignored)
C_TN = 0.0       # correct pass
C_REVIEW = 0.5   # one human review of one case (~1 min at ~30 EUR/h), assumed always correct
BASE_RATE = 0.05  # assumed share of production cases containing a violation (test split: ~29%)
SENS_C_FN = [5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0]
SENS_BASE_RATE = [0.01, 0.05, 0.10, 0.29]

THRESHOLDS = np.round(np.arange(0.01, 1.0, 0.01), 2)
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ================================================================ dataset ==
def download_dataset() -> Path:
    dest = DATA_DIR / f"{PROJECT}-v{VERSION}"
    if (dest / "data.yaml").exists():
        return dest
    from roboflow import Roboflow

    rf = Roboflow(api_key=load_api_key())
    version = rf.workspace(WORKSPACE).project(PROJECT).version(VERSION)
    version.download(EXPORT_FORMAT, location=str(dest))
    return dest


def write_resolved_yaml(dataset_dir: Path):
    """Own data yaml with absolute paths, so it never depends on Roboflow's '../'."""
    import yaml

    raw = yaml.safe_load((dataset_dir / "data.yaml").read_text())
    names = raw["names"]
    if isinstance(names, list):
        names = dict(enumerate(names))
    names = {int(k): v for k, v in names.items()}
    resolved = {
        "path": str(dataset_dir),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": names,
    }
    out = dataset_dir / "data_resolved.yaml"
    out.write_text(yaml.safe_dump(resolved, sort_keys=False))
    return out, names


def read_labels(label_file: Path):
    rows = []
    if label_file.exists():
        for line in label_file.read_text().splitlines():
            p = line.split()
            if len(p) >= 5:
                rows.append((int(p[0]), float(p[3]), float(p[4])))  # cls, w, h (normalised)
    return rows


def dataset_stats(dataset_dir: Path, names: dict) -> dict:
    stats = {}
    for split in ("train", "valid", "test"):
        imgs = [p for p in (dataset_dir / split / "images").iterdir() if p.suffix.lower() in IMG_EXT]
        inst = {n: 0 for n in names.values()}
        areas = {n: [] for n in names.values()}
        imgs_with = {n: 0 for n in names.values()}
        for p in imgs:
            rows = read_labels(dataset_dir / split / "labels" / f"{p.stem}.txt")
            seen = set()
            for c, w, h in rows:
                inst[names[c]] += 1
                areas[names[c]].append(w * h)
                seen.add(names[c])
            for n in seen:
                imgs_with[n] += 1
        stats[split] = {
            "images": len(imgs),
            "instances": inst,
            "images_with_class": imgs_with,
            "median_box_area_share": {
                n: (float(np.median(a)) if a else None) for n, a in areas.items()
            },
        }
    return stats


# ================================================================ training ==
def train(data_yaml: Path):
    from ultralytics import YOLO

    last = RUNS_DIR / "train" / "weights" / "last.pt"
    if os.environ.get("RESUME") == "1" and last.exists() and not SMOKE:
        model = YOLO(str(last))  # continue an interrupted run from its last epoch
        model.train(resume=True)
    else:
        model = YOLO(MODEL)
        model.train(
            data=str(data_yaml),
            epochs=1 if SMOKE else EPOCHS,
            fraction=0.05 if SMOKE else 1.0,
            imgsz=IMGSZ,
            batch=BATCH,
            patience=PATIENCE,
            seed=SEED,
            deterministic=True,
            project=str(RUNS_DIR),
            name="train",
            exist_ok=True,
            plots=True,
        )
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    WEIGHTS_DIR.mkdir(exist_ok=True)
    for old in WEIGHTS_DIR.glob("*.pt"):
        old.unlink()
    target = WEIGHTS_DIR / "best.pt"
    shutil.copy(best, target)
    save_dir = Path(model.trainer.save_dir)
    return target, save_dir


def free_memory():
    """Call between stages, after the previous stage's objects have gone out of scope."""
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()


# ============================================================== evaluation ==
def box_level_results(weights: Path, data_yaml: Path, names: dict) -> dict:
    """Per-class P, R, AP50, mAP50-95 on the test split (+ Ultralytics confusion matrix)."""
    from ultralytics import YOLO

    m = YOLO(str(weights)).val(
        data=str(data_yaml), split="test", imgsz=IMGSZ, conf=0.001, iou=0.6,
        batch=BATCH, plots=True, project=str(RUNS_DIR), name="test_eval", exist_ok=True,
    )
    per_class = {}
    for i, c in enumerate(m.box.ap_class_index):
        p, r, ap50, ap = m.box.class_result(i)
        per_class[names[int(c)]] = {
            "precision": float(p), "recall": float(r), "ap50": float(ap50), "map50_95": float(ap),
        }
    ARTIFACTS.mkdir(exist_ok=True)
    for f in Path(m.save_dir).glob("confusion_matrix*.png"):
        shutil.copy(f, ARTIFACTS / f.name)
    out = {
        "overall": {"map50": float(m.box.map50), "map50_95": float(m.box.map)},
        "per_class": per_class,
        "confusion_matrix_labels": [names[i] for i in sorted(names)] + ["background"],
        "confusion_matrix": np.asarray(m.confusion_matrix.matrix).tolist(),
        "note_confusion_matrix": "Ultralytics default matching (conf 0.25, IoU 0.45), raw counts; "
                                 "rows/cols = classes + background.",
    }
    return out


def image_scores(weights: str, dataset_dir: Path, class_ids: list, split="test", chunk=64):
    """One score per image = max confidence of the given classes (0 if none).
    Works in small chunks and frees memory between them (Colab RAM is limited)."""
    import gc

    import torch
    from ultralytics import YOLO

    model = YOLO(weights)
    paths = sorted(p for p in (dataset_dir / split / "images").iterdir() if p.suffix.lower() in IMG_EXT)
    scores = {}
    for i in range(0, len(paths), chunk):
        batch = [str(p) for p in paths[i:i + chunk]]
        for r in model.predict(
            source=batch, conf=0.001, iou=0.6, imgsz=IMGSZ, classes=class_ids,
            max_det=100, batch=BATCH, verbose=False,
        ):
            scores[Path(r.path).stem] = float(r.boxes.conf.max()) if len(r.boxes) else 0.0
        del batch
        gc.collect()
        if (i // chunk) % 10 == 0:
            print(f"  scored {min(i + chunk, len(paths))}/{len(paths)} images", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return scores


def ground_truth(dataset_dir: Path, positive_id: int, split="test"):
    out = {}
    for p in (dataset_dir / split / "images").iterdir():
        if p.suffix.lower() in IMG_EXT:
            rows = read_labels(dataset_dir / split / "labels" / f"{p.stem}.txt")
            out[p.stem] = any(c == positive_id for c, _, _ in rows)
    return out


# ================================================================= costing ==
def default_costs() -> dict:
    return {"fn": C_FN, "fp": C_FP, "tp": C_TP, "tn": C_TN, "review": C_REVIEW}


def rates_at(scores: np.ndarray, labels: np.ndarray, t: float):
    pred = scores >= t
    tp = int((pred & labels).sum())
    fp = int((pred & ~labels).sum())
    fn = int((~pred & labels).sum())
    tn = int((~pred & ~labels).sum())
    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "tpr": tpr, "fpr": fpr}


def cost_per_1000(tpr: float, fpr: float, prev: float = BASE_RATE, c: dict = None) -> float:
    """Expected cost per 1,000 cases at base rate `prev`, using test TPR/FPR."""
    c = c or default_costs()
    pos = prev * (tpr * c["tp"] + (1 - tpr) * c["fn"])
    neg = (1 - prev) * (fpr * c["fp"] + (1 - fpr) * c["tn"])
    return 1000.0 * (pos + neg)


def sweep(scores: np.ndarray, labels: np.ndarray, prev: float = BASE_RATE, c: dict = None) -> dict:
    rows = []
    for t in THRESHOLDS:
        r = rates_at(scores, labels, float(t))
        r["threshold"] = float(t)
        r["cost_per_1000"] = cost_per_1000(r["tpr"], r["fpr"], prev, c)
        rows.append(r)
    best = min(rows, key=lambda r: r["cost_per_1000"])
    return {"rows": rows, "best": best}


def review_policy(scores: np.ndarray, labels: np.ndarray, prev: float = BASE_RATE,
                  c: dict = None) -> dict:
    """Best (t_low, t_high): score < t_low -> pass, >= t_high -> fail, between -> human review.
    Searched on the same 0.01 grid as the single-threshold sweep (plus 0 and 1), so the
    result can never be worse than the best single threshold (t_low == t_high = no review).
    Assumes the reviewer decides correctly and costs c['review'] per case."""
    c = c or default_costs()
    grid = np.round(np.arange(0.0, 1.0 + 1e-9, 0.01), 2)
    pos, neg = np.sort(scores[labels]), np.sort(scores[~labels])
    n_pos, n_neg = len(pos), len(neg)
    lt_pos = np.searchsorted(pos, grid, side="left")  # cases with score < grid value
    lt_neg = np.searchsorted(neg, grid, side="left")
    lo, hi = np.meshgrid(np.arange(len(grid)), np.arange(len(grid)), indexing="ij")
    valid = hi >= lo
    pos_pass = lt_pos[lo] / n_pos
    pos_fail = (n_pos - lt_pos[hi]) / n_pos
    pos_rev = (lt_pos[hi] - lt_pos[lo]) / n_pos
    neg_pass = lt_neg[lo] / n_neg
    neg_fail = (n_neg - lt_neg[hi]) / n_neg
    neg_rev = (lt_neg[hi] - lt_neg[lo]) / n_neg
    cost = 1000 * (
        prev * (pos_pass * c["fn"] + pos_rev * c["review"] + pos_fail * c["tp"])
        + (1 - prev) * (neg_pass * c["tn"] + neg_rev * c["review"] + neg_fail * c["fp"])
    )
    cost = np.where(valid, cost, np.inf)
    i, j = np.unravel_index(int(np.argmin(cost)), cost.shape)
    return {"t_low": float(grid[i]), "t_high": float(grid[j]),
            "cost_per_1000": float(cost[i, j]),
            "review_share": float(prev * pos_rev[i, j] + (1 - prev) * neg_rev[i, j])}


def sensitivity(scores: np.ndarray, labels: np.ndarray) -> list:
    """How the chosen thresholds and costs move when C_FN and the base rate change."""
    rows = []
    for fn in SENS_C_FN:
        for prev in SENS_BASE_RATE:
            c = {**default_costs(), "fn": fn}
            single = sweep(scores, labels, prev, c)["best"]
            pol = review_policy(scores, labels, prev, c)
            rows.append({
                "c_fn": fn, "base_rate": prev,
                "single_threshold": single["threshold"],
                "single_cost_per_1000": single["cost_per_1000"],
                "review_t_low": pol["t_low"], "review_t_high": pol["t_high"],
                "review_cost_per_1000": pol["cost_per_1000"],
                "review_share": pol["review_share"],
                "do_nothing_cost_per_1000": cost_per_1000(0.0, 0.0, prev, c),
                "human_all_cost_per_1000": 1000.0 * c["review"],
            })
    return rows


def cost_sections(s_ft: np.ndarray, s_coco: np.ndarray, labels: np.ndarray) -> dict:
    """Everything that depends on the cost assumptions (so it can be recomputed alone)."""
    sw_ft, sw_coco = sweep(s_ft, labels), sweep(s_coco, labels)
    policy = review_policy(s_ft, labels)
    return {
        "costs_eur": {"C_FN": C_FN, "C_FP": C_FP, "C_TP": C_TP, "C_TN": C_TN,
                      "C_REVIEW": C_REVIEW, "base_rate": BASE_RATE},
        "test_set": {"cases": int(len(labels)), "positives": int(labels.sum()),
                     "base_rate_in_test": float(labels.mean())},
        "image_level": {"finetuned": sw_ft, "coco_person_proxy": sw_coco},
        "chosen_single_threshold": sw_ft["best"],
        "chosen_review_policy": policy,
        "strategies_cost_per_1000": {
            "predict_nothing_always_pass": cost_per_1000(0.0, 0.0),
            "predict_all_always_fail": cost_per_1000(1.0, 1.0),
            "human_checks_everything": 1000.0 * C_REVIEW,
            "model_alone_best_threshold": sw_ft["best"]["cost_per_1000"],
            "model_plus_human_review": policy["cost_per_1000"],
            "coco_person_proxy_best_threshold": sw_coco["best"]["cost_per_1000"],
        },
        "sensitivity": sensitivity(s_ft, labels),
    }


def load_saved_scores():
    rows = list(csv.DictReader(open(ARTIFACTS / "test_scores.csv")))
    labels = np.array([int(r["has_bare_head"]) for r in rows], dtype=bool)
    s_ft = np.array([float(r["score_finetuned"]) for r in rows])
    s_coco = np.array([float(r["score_coco_person"]) for r in rows])
    return s_ft, s_coco, labels


def costs_only():
    """Recompute only the cost sections from artifacts/test_scores.csv (no GPU, seconds)."""
    s_ft, s_coco, labels = load_saved_scores()
    results = json.loads((ARTIFACTS / "results.json").read_text())
    results.update(cost_sections(s_ft, s_coco, labels))
    results.pop("note_costs", None)
    (ARTIFACTS / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: results[k] for k in ("costs_eur", "test_set", "chosen_single_threshold",
                                               "chosen_review_policy", "strategies_cost_per_1000")},
                     indent=2))


# ===================================================================== main ==
def main():
    if os.environ.get("COSTS_ONLY") == "1":
        costs_only()
        return
    CACHE_DIR.mkdir(exist_ok=True)
    os.chdir(CACHE_DIR)  # pretrained checkpoints are downloaded here, not into the repo root
    ARTIFACTS.mkdir(exist_ok=True)

    dataset_dir = download_dataset()
    data_yaml, names = write_resolved_yaml(dataset_dir)
    positive_id = next((i for i, n in names.items() if n == POSITIVE_CLASS), None)
    if positive_id is None:
        raise SystemExit(f"Class {POSITIVE_CLASS!r} not in dataset classes: {names}")
    print("[1/6] dataset ready", flush=True)
    stats = dataset_stats(dataset_dir, names)

    print("[2/6] training", flush=True)
    weights, train_dir = train(data_yaml)
    free_memory()
    print("[3/6] box-level test evaluation", flush=True)
    box = box_level_results(weights, data_yaml, names)
    free_memory()
    print("[4/6] image-level scores (finetuned, then COCO)", flush=True)

    # --- image-level scores on the test split -------------------------------
    gt = ground_truth(dataset_dir, positive_id)
    ft = image_scores(str(weights), dataset_dir, [positive_id])
    coco = image_scores("yolo11n.pt", dataset_dir, [0])  # COCO class 0 = person
    stems = sorted(gt)
    labels = np.array([gt[s] for s in stems], dtype=bool)
    s_ft = np.array([ft.get(s, 0.0) for s in stems])
    s_coco = np.array([coco.get(s, 0.0) for s in stems])

    with open(ARTIFACTS / "test_scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "has_bare_head", "score_finetuned", "score_coco_person"])
        for s, y, a, b in zip(stems, labels, s_ft, s_coco):
            w.writerow([s, int(y), f"{a:.5f}", f"{b:.5f}"])

    print("[5/6] cost sweep and review policy", flush=True)
    # --- costs ----------------------------------------------------------------
    results = {
        "dataset": {"workspace": WORKSPACE, "project": PROJECT, "version": VERSION,
                    "positive_class": POSITIVE_CLASS, "classes": names, "stats": stats},
        "training": {"model": MODEL, "epochs_requested": 1 if SMOKE else EPOCHS, "imgsz": IMGSZ,
                     "batch": BATCH, "seed": SEED, "smoke_test": SMOKE,
                     "weights": str(weights.relative_to(ROOT))},
        "box_level": box,
        "note_coco": "COCO has no head/helmet/vest classes, so box-level AP is 0 by construction; "
                     "the image-level proxy flags an image when COCO sees a person.",
    }
    results.update(cost_sections(s_ft, s_coco, labels))
    (ARTIFACTS / "results.json").write_text(json.dumps(results, indent=2))
    for name in ("results.png", "PR_curve.png", "F1_curve.png"):
        if (train_dir / name).exists():
            shutil.copy(train_dir / name, ARTIFACTS / name)

    print(json.dumps({k: results[k] for k in ("chosen_single_threshold", "chosen_review_policy",
                                               "strategies_cost_per_1000")}, indent=2))
    print(f"Weights: {weights}   Artifacts: {ARTIFACTS}")


if __name__ == "__main__":
    main()
