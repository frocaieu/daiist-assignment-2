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

All euro figures below are PLACEHOLDERS until REPORT.md justifies them
(assignment step 6). Change them here; nothing else needs editing.
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

# ------------------------------------------------- costs (PLACEHOLDERS, EUR)
C_FN = 250.0    # bare head not flagged (missed violation)
C_FP = 2.0      # helmeted head flagged (needless check)
C_TP = 0.0      # violation correctly flagged (cost of acting on it is ignored)
C_TN = 0.0      # correct pass
C_REVIEW = 1.0  # one human review of one case
BASE_RATE = 0.05  # assumed share of production cases that contain a violation

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
def rates_at(scores: np.ndarray, labels: np.ndarray, t: float):
    pred = scores >= t
    tp = int((pred & labels).sum())
    fp = int((pred & ~labels).sum())
    fn = int((~pred & labels).sum())
    tn = int((~pred & ~labels).sum())
    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "tpr": tpr, "fpr": fpr}


def cost_per_1000(tpr: float, fpr: float, prev: float = BASE_RATE) -> float:
    """Expected cost per 1,000 cases at base rate `prev`, using test TPR/FPR."""
    pos = prev * (tpr * C_TP + (1 - tpr) * C_FN)
    neg = (1 - prev) * (fpr * C_FP + (1 - fpr) * C_TN)
    return 1000.0 * (pos + neg)


def sweep(scores: np.ndarray, labels: np.ndarray) -> dict:
    rows = []
    for t in THRESHOLDS:
        r = rates_at(scores, labels, float(t))
        r["threshold"] = float(t)
        r["cost_per_1000"] = cost_per_1000(r["tpr"], r["fpr"])
        rows.append(r)
    best = min(rows, key=lambda r: r["cost_per_1000"])
    return {"rows": rows, "best": best}


def review_policy(scores: np.ndarray, labels: np.ndarray, step=0.02) -> dict:
    """Best (t_low, t_high): score < t_low -> pass, >= t_high -> fail, between -> human review.
    Assumes the reviewer decides correctly and costs C_REVIEW per case."""
    pos, neg = scores[labels], scores[~labels]
    grid = np.round(np.arange(0.0, 1.0 + 1e-9, step), 2)
    best = None
    for lo in grid:
        for hi in grid[grid >= lo]:
            pos_pass, pos_fail = (pos < lo).mean(), (pos >= hi).mean()
            pos_rev = 1 - pos_pass - pos_fail
            neg_pass, neg_fail = (neg < lo).mean(), (neg >= hi).mean()
            neg_rev = 1 - neg_pass - neg_fail
            cost = 1000 * (
                BASE_RATE * (pos_pass * C_FN + pos_rev * C_REVIEW + pos_fail * C_TP)
                + (1 - BASE_RATE) * (neg_pass * C_TN + neg_rev * C_REVIEW + neg_fail * C_FP)
            )
            review_share = BASE_RATE * pos_rev + (1 - BASE_RATE) * neg_rev
            if best is None or cost < best["cost_per_1000"]:
                best = {"t_low": float(lo), "t_high": float(hi),
                        "cost_per_1000": float(cost), "review_share": float(review_share)}
    return best


# ===================================================================== main ==
def main():
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
    sw_ft, sw_coco = sweep(s_ft, labels), sweep(s_coco, labels)
    policy = review_policy(s_ft, labels)
    baselines = {
        "predict_nothing_always_pass": cost_per_1000(0.0, 0.0),
        "predict_all_always_fail": cost_per_1000(1.0, 1.0),
        "human_checks_everything": 1000.0 * C_REVIEW,
        "model_alone_best_threshold": sw_ft["best"]["cost_per_1000"],
        "model_plus_human_review": policy["cost_per_1000"],
        "coco_person_proxy_best_threshold": sw_coco["best"]["cost_per_1000"],
    }

    results = {
        "dataset": {"workspace": WORKSPACE, "project": PROJECT, "version": VERSION,
                    "positive_class": POSITIVE_CLASS, "classes": names, "stats": stats},
        "training": {"model": MODEL, "epochs_requested": 1 if SMOKE else EPOCHS, "imgsz": IMGSZ,
                     "batch": BATCH, "seed": SEED, "smoke_test": SMOKE,
                     "weights": str(weights.relative_to(ROOT))},
        "costs_eur": {"C_FN": C_FN, "C_FP": C_FP, "C_TP": C_TP, "C_TN": C_TN,
                      "C_REVIEW": C_REVIEW, "base_rate": BASE_RATE, "placeholder": True},
        "test_set": {"cases": len(stems), "positives": int(labels.sum()),
                     "base_rate_in_test": float(labels.mean())},
        "box_level": box,
        "image_level": {"finetuned": sw_ft, "coco_person_proxy": sw_coco},
        "chosen_single_threshold": sw_ft["best"],
        "chosen_review_policy": policy,
        "strategies_cost_per_1000": baselines,
        "note_coco": "COCO has no head/helmet/vest classes, so box-level AP is 0 by construction; "
                     "the image-level proxy flags an image when COCO sees a person.",
    }
    (ARTIFACTS / "results.json").write_text(json.dumps(results, indent=2))
    for name in ("results.png", "PR_curve.png", "F1_curve.png"):
        if (train_dir / name).exists():
            shutil.copy(train_dir / name, ARTIFACTS / name)

    print(json.dumps({k: results[k] for k in ("chosen_single_threshold", "chosen_review_policy",
                                               "strategies_cost_per_1000")}, indent=2))
    print(f"Weights: {weights}   Artifacts: {ARTIFACTS}")


if __name__ == "__main__":
    main()
