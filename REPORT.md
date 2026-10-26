# Assignment 2 Report

- **Name**: Francisco Javier Roca González
- **Student ID**: 15701
- **Email**: froca.ieu2021@student.ie.edu
- **Group**: BBADBA 5A 


## Business problem, and the cost of being wrong

**Decision.** For each image from a construction-site camera: is a worker visible
with a bare head, i.e. without a hard hat? One *case* is one image. The *positive
class* is `head` (a bare head); a case is positive if the image contains at least
one.

**Who acts.** The site's safety coordinator (prevention technician). The model
gives each image a score (its highest bare-head confidence) and two thresholds
split the cases into three outcomes:

- **Pass**: nothing happens.
- **Human review**: a person looks at the image and decides.
- **Fail**: an automatic alert; the foreman is sent to correct the situation.

**The two mistakes are not symmetric.**

- *False negative* (a bare head is passed without anyone seeing it): the worker keeps
  working unprotected. This is the expensive error: it exposes the worker to an
  injury and the company to a fine.
- *False positive* (a helmeted worker is failed): someone is sent to check
  for nothing. This is cheap but not free, and if frequent it leads to alert fatigue.

| Event | Cost | Where the figure comes from |
|---|---|---|
| False negative (C_FN) | **100 €** | Minimum fine for a *grave* occupational-risk infringement, 2,451 € (Spanish LISOS, Real Decreto Legislativo 5/2000, art. 40.2; range 2,451–9,830 € at minimum degree), multiplied by an **assumed 4 %** chance that a missed violation ends in a sanction. It excludes the cost of an injury, a work stoppage and liability, so it is a floor. |
| False positive: compliant case failed (C_FP) | **5 €** | About 10 minutes of a foreman or technician at an **assumed** 30 €/h. |
| Human review (C_REVIEW) | **0.5 €** | About 1 minute at the same 30 €/h; the reviewer is **assumed to decide correctly**. |
| True positive / true negative | 0 € | Acting on a real violation is not counted as a cost of the model. |


So a missed violation is assumed to cost **20× a false alarm** and **200× a review**.
Only the fine range is taken from a legal source; the 4 %, the 30 €/h, the
10 minutes and the 1 minute are my assumptions, and the section on the operating
point shows which of them matter.

**Why C_FP > C_REVIEW.** A *fail* triggers a real action (sending someone), a
*review* is only a look. If a wrong fail cost no more than a review, failing would
never be worse than reviewing and the review band would be empty. This
turned out to be a mistake in my first set of costs (I had set both to 0.5 €), which
I corrected.

## Dataset

- **Workspace / project / version**: `ppe-kit-detection` / `hardhat-safetyvest` / **v1**
- **Link**: https://universe.roboflow.com/ppe-kit-detection/hardhat-safetyvest
- **Licence**: CC BY 4.0. Used as published, **not forked**. Version v1 was generated
  on 2024-05-01; preprocessing is auto-orient and a stretch resize to 640×640, with
  **no augmentation**. 22,068 images in total.
- **Classes**: `head`, `helmet`, `vest`. One image is one camera frame; each object is a
  bounding box.

**Why it serves the problem.** It has an explicit class for the violation (`head`)
alongside `helmet`, so a false negative can actually be measured; most helmet
datasets only label helmets. Object detection also gives me the location of each
worker, which a whole-image classifier would not.

**What I assume.** The dataset has no description, so I **read `head` as a head
without a helmet** from the class names and from the way `helmet` and `head` are
used as alternatives. The confusion matrix shows few helmet/head swaps, which is
consistent with this but does not prove it.

**Where it only resembles the problem.** The images I looked at on the dataset page
include many crowded indoor scenes (classrooms, meeting rooms) as well as site
frames, and some carry a CCTV timestamp, i.e. they look like frames from surveillance
videos. It is not a dataset of construction sites with a typical worker density.

## What the dataset's composition implies

| | Train | Valid | Test |
|---|---|---|---|
| Images | 17,226 | 2,413 | 2,429 |
| `head` instances | 98,174 | 13,519 | 12,478 |
| `helmet` instances | 43,892 | 6,494 | 6,698 |
| `vest` instances | 6,322 | 861 | 928 |
| Images containing `head` | 5,801 (33.7 %) | 708 | 707 (29.1 %) |

Median box area as a share of the image (test): `head` **0.11 %** (about 21 px on a
640 px image), `helmet` 0.66 % (about 52 px), `vest` 4.1 % (about 130 px).

What this implies:

1. **No class is too thin to compute a number, but `vest` is the weakest.** 928 test
   instances in 437 images give a usable but noisier estimate than `head`. I would not
   read small differences in `vest` AP as real.
2. **`head` is the majority class by instances and the objects are tiny.** There are
   about 17.6 bare heads per image where one appears, which confirms the crowded
   scenes. Tiny boxes are hard to localise precisely, which should hurt the strict
   mAP50-95 more than AP50 (it does, see below).
3. **The base rate in the test split (29 %) is far above the 5 % I assume in
   production**, so precision measured on the test split is not the precision a site
   would see. I therefore compute costs from the test-split *rates* (true- and
   false-positive rates) and re-weight them to the assumed base rate.
4. **The test split probably looks like the training split, not like a real site.** I did
   not verify how the splits were made, but they probably are random draws from the same
   pool and likely contain near-identical frames of the same videos. I did not check
   for duplicates. Results are therefore
   likely optimistic, and I **cannot conclude** how the model performs on new cameras.

## Finetuning and baselines

**Model.** `yolo11n` for **object detection**, finetuned from the COCO checkpoint.
Detection rather than classification because I need to find each worker, and the
per-image decision is built from the detections. **Trained on Google Colab (T4 GPU)**
with `python main.py train`: 15 epochs (1.04 h), image size 640, batch 32, seed 0,
patience 8, Ultralytics 8.4.189, Ultralytics' automatic optimiser choice and default
augmentations. The best weights are committed in `weights/best.pt`.

**Results on the same test split (2,429 images), image level at the 5 % base rate.**
Image level means: a case is flagged if its bare-head score reaches the threshold.

| Approach | Accuracy | Precision* | Recall | Cost per 1,000 cases |
|---|---|---|---|---|
| Finetuned yolo11n (threshold 0.30) | 0.937 | 0.842 | 0.965 | **530 €** |
| COCO yolo11n zero-shot (flag if a *person* is seen) | 0.310 | 0.295 | 0.983 | 4,672 € |
| Trivial: predict nothing (always pass) | 0.709 | n/a | 0.000 | 5,000 € |
| Trivial: flag everything | 0.291 | 0.291 | 1.000 | 4,750 € |

*Precision and accuracy are on the test split (29 % positives), not at the 5 % base rate.

**COCO baseline.** COCO has no `head`, `helmet` or `vest` class, so its box-level AP
is zero by construction and was not computed. The only zero-shot proxy I could build
is "a person is visible, so flag it". Its best threshold (0.01) flags 96.6 % of the
compliant images, i.e. it behaves like "flag everything".

**What changed.** Fine-tuning moves the cost from about 4,700–5,000 € to 530 € per
1,000 cases. Note that the "predict nothing" baseline has **71 % accuracy while
catching no violation at all**, which is why accuracy is not the metric I use. At
box level the finetuned model reaches mAP50 0.886 and mAP50-95 0.535.

## Per-class evaluation and failure modes

Box-level results on the test split. Precision and recall are as reported by
Ultralytics at its best-F1 operating point; AP50 uses IoU 0.5.

| Class | Precision | Recall | AP50 | mAP50-95 |
|---|---|---|---|---|
| **head (positive)** | 0.897 | 0.871 | 0.892 | 0.464 |
| helmet | 0.931 | 0.845 | 0.909 | 0.598 |
| vest | 0.843 | 0.760 | 0.856 | 0.542 |
| all | n/a | n/a | 0.886 | 0.535 |

**Headline class.** `head`, because it is the violation. Its AP50 (0.892) is healthy,
but its **mAP50-95 (0.464) is the lowest**: the model finds the heads but boxes them
less precisely, as expected for objects about 21 px wide. `vest` has the **lowest
recall (0.760)** and the least data.

**What the model confuses.** In the Ultralytics confusion matrix (rows = predicted,
columns = true; built from every detection above confidence 0.001):

- True helmets predicted as `head`: **388** (5.8 % of 6,698 helmets). This is the
  *cheap* direction: a false alarm.
- True heads predicted as `helmet`: **51** (0.4 % of 12,478 heads). This is the
  *dangerous* direction, and it is rare.
- Objects missed outright: `head` 225 (1.8 %), `helmet` 114 (1.7 %), `vest` 17 (1.8 %).
- `vest` is almost never confused with another class (only 4 of 928 vests predicted as `head` or `helmet`).

Caveats: because the matrix includes very low-confidence detections, the
"background" column is huge (46,046 unmatched `head` predictions) and is **not**
meaningful on its own. The class-versus-class cells include low-confidence
predictions, so at a real threshold the confusions would be fewer.

**Where the system fails.** At the chosen policy, **14 of 707 violating images (2.0 %)
pass without being seen**. I have not yet inspected those 14 images, so I cannot say
what they have in common (for example very small or occluded heads).

## Cost matrix and operating point

**Base rate.** I assume **5 %** of cases contain a violation. This is an assumption
about a site where most workers comply; the test split has 29 %. The expected cost
at base rate *p* is computed from the test-split rates:
`1000 × [ p·(1−TPR)·C_FN + (1−p)·FPR·C_FP ]`.

**Sweep (model alone, one threshold, cost per 1,000 cases):**

| Threshold | 0.05 | 0.10 | 0.20 | **0.30** | 0.40 | 0.50 | 0.70 | 0.90 |
|---|---|---|---|---|---|---|---|---|
| Cost (€) | 696 | 625 | 568 | **530** | 539 | 535 | 908 | 5,000 |

**Chosen operating point for the model alone: 0.30** (530 € per 1,000 cases). The
curve is **flat between 0.30 and 0.50** (within about 2 %), so the exact value is not
fragile. Below 0.20 false positives grow quickly, and above 0.70 missed violations
dominate. At 0.90 no image reaches the threshold and the cost equals doing nothing.
At 0.30 the cost splits into about 177 € of missed violations and 353 € of false alarms.

**The assumption that would most change my answer.** Two matter most:

- **C_FN (i.e. the assumed 4 % chance).** The model alone beats a person checking
  everything (500 €) only if C_FN is below **about 87 €**, that is, if the chance a
  missed violation is sanctioned is below roughly 3.5 %. At my 100 € it does *not* win.
- **The base rate.** With C_FN = 100 €, the model alone beats the 500 € alternative
  only if violations are rarer than **about 4.3 %** of cases.

**Sensitivity (5 % base rate, other costs unchanged; cost per 1,000 cases):**

| C_FN | 5 € | 10 € | 25 € | 50 € | **100 €** | 250 € | 1,000 € |
|---|---|---|---|---|---|---|---|
| Model alone | 197 | 250 | 317 | 404 | **530** | 776 | 1,633 |
| Model + human review | 63 | 76 | 101 | 136 | **187** | 283 | 499 |

Two further observations:

- At C_FN ≈ 1,000 € the best policy becomes "review everything" (499 € against 500 €).
- **C_FP only matters up to a point.** For C_FP between 5 € and 20 € the review policy
  does not change. If C_FP falls to C_REVIEW (0.5 €) the review band disappears.

## What a person still has to check

**Thresholds.** Score below **0.04 → pass**; from **0.87 → fail**; in between → **human
review**. They were found by minimising the expected cost over all pairs on a 0.01 grid.

**Outcomes on the test split (707 violating, 1,722 compliant images):**

| | Pass | Review | Fail |
|---|---|---|---|
| Violation present | 14 (2.0 %) | 665 (94.1 %) | 28 (4.0 %) |
| No violation | 1,489 (86.5 %) | 233 (13.5 %) | 0 |

**What this means.** The model works as a **filter**, not as an automatic judge.
It clears 86.5 % of the compliant cases without a person and misses 2 % of the
violations. It almost never fails a case automatically: bare-head confidences rarely
reach 0.87, and a wrong *fail* (5 €) costs ten times a review (0.5 €), so sending a
case to a person is nearly always cheaper. In practice most violations are
confirmed by a person.

**Workload.** I assume a mid-sized site: **10 cameras sending one image every 10
minutes over an 8-hour shift = 480 cases per day.** At the 5 % base rate:

| Per day | Pass | Review | Fail |
|---|---|---|---|
| Cases | ≈ 395 | **≈ 84** (17.6 %) | ≈ 1 |

At **1 minute per review** that is about **84 minutes a day**, which is **about 0.23 of
a person** at 6 productive hours per person, i.e. it can be done by the existing
coordinator in short sessions and needs no new hire. It is realistic provided someone
answers reviews quickly during the shift. About 24 violating cases per day are
expected, of which about 0.5 (2 %) would pass unseen.

**The design against the three alternatives (cost per 1,000 cases, per day at 480
cases, and people needed):**

| Strategy | € per 1,000 | € per day | Human work |
|---|---|---|---|
| Do nothing | 5,000 | 2,400 | none |
| A person checks everything | 500 | 240 | 8 h/day, **1.33 people** |
| Model alone (threshold 0.30) | 530 | 254 | none, but about 57 alerts/day, **about 34 of them false (59 %)** |
| **Model + human review** | **187** | **90** | 84 min/day, **0.23 people** |

The best design saves about **150 € a day** against a person checking everything, and
cuts the human workload by about 83 %. The model **alone** does not pay off against a
person checking everything under these assumptions: at a 5 % base rate more than half
of its alerts are false. These comparisons assume that a human reviewer is always right
and never tires, which is optimistic.

## Limitations & next steps

1. **Domain mismatch.** The data are crowded indoor scenes with about 17.6 bare heads
   per image, and a 29 % base rate against the 5 % I assume. *Next:* label a few
   hundred frames from the real site cameras and evaluate on them.
2. **Probable leakage between splits.** Likely video frames of the same scenes appear
   in train and test, so the metrics are optimistic. *Next:* split by video or camera,
   and detect near-duplicates between splits (for example with image hashes).
3. **Thresholds chosen on the test split.** The cost estimates are slightly
   optimistic. *Next:* run the sweep on the validation split and report the test split
   only once.
4. **A case is an image scored by its highest bare-head confidence.** This ignores how
   many heads there are and where. A single bare head among twenty people gets one
   score. *Next:* score per person, or per tracked worker over time.
5. **Small objects.** Heads are about 21 px wide, and mAP50-95 (0.464) is low. The model
   was trained for only 15 epochs and mAP50-95 was still rising at the last one
   (0.520 to 0.527). *Next:* train longer, use a higher resolution (960 or tiling) or
   `yolo11s`.
6. **The costs are assumptions.** The 4 % chance, 30 €/h, the review times and a
   perfect reviewer are not measured. *Next:* use site incident data and time real
   reviews, including the reviewer's own error rate.
7. **I have not inspected the 14 missed violations**, so I do not know their common
   cause. *Next:* look at them and, if they share a pattern, add targeted data.
8. **`vest` is not part of the decision**, and it is the weakest class. *Next:* add a
   no-vest class and a second cost line.

## Generative AI use disclosure

I used Claude (Anthropic) as an AI assistant throughout this project. It provided initial project ideas, helped troubleshoot environment errors, and proposed the theoretical structure and assumed values for the cost model (which I reviewed, verified against sources such as the BOE, and accepted). I chose the topic and training setup, wrote the complete training pipeline (`train.py`), developed the interactive dashboard (`app.py`), ran the training on Google Colab, tested the system, and authored this report (`REPORT.md`). 