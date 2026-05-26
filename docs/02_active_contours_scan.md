# Literature scan — Active contours / splines / temporal segmentation for tracking

Quick check before moving to Phase 2: is there a classical contour /
spline / level-set approach to *temporal* tracking that would
meaningfully outperform our motion-persistence baseline on aerial
thermal footage of a small (~10–30 px) walker?

## Body of work surveyed

| Family | Key references | Idea |
|---|---|---|
| **Parametric snakes** | Kass, Witkin, Terzopoulos, *Snakes: Active Contour Models*, IJCV 1988 | Energy-minimising contour: image forces (gradient) + internal smoothness/tension. Tracking = re-evolve from previous frame contour. |
| **B-spline contour tracking** | Blake & Isard, *Active Contours*, Springer 1998; Isard & Blake, *CONDENSATION*, IJCV 1998 | B-spline-parameterised contour evolved with a particle filter; temporal prior keeps shape coherent. |
| **Geodesic active contours (level sets)** | Caselles, Kimmel, Sapiro, IJCV 1997 | Same idea in a level-set framework — handles topology changes (splits/merges) naturally. |
| **Region-based active contours** | Chan & Vese, *Active Contours Without Edges*, IEEE TIP 2001 | Drops the gradient requirement — segments by region statistics. **Most relevant family for thermal** (edges are soft). |
| **Contour-saliency for thermal** | Davis & Sharma, *Background-Subtraction in Thermal Imagery Using Contour Saliency*, CVPR-W 2005 | Pedestrian segmentation in static-camera thermal. Combines contour + intensity statistics. |
| **Level-sets for IR small-target** | Various 2010s IR small-target detection surveys | Mostly *detection* / segmentation; tracking is bolted on with Kalman afterwards (same architecture we already have). |

## Where these methods are strong (and where they aren't)

Active contours / splines pay off when **all** of the following hold:
1. Target occupies enough pixels that its **shape** carries information (typically ≥ 60–100 px on a side).
2. There is a discriminative **region/edge cue** the contour can latch onto — strong gradients (RGB faces, vehicles) or distinct region statistics (medical organs).
3. The target is **roughly continuous** between frames; the contour from frame *t* is a good initialisation at frame *t+1*.

Our scenario violates 1 and 2:
- The target is 10–30 px across — a 4-pixel contour change is huge in relative terms; the contour adds essentially no information beyond the bbox.
- Thermal edges are soft (sensor blur, no albedo contrast). Region statistics inside a 10-px blob are dominated by sensor noise.
- The drone is moving fast and the person partially hides under canopy → frequent contour topology changes (split / disappear / reappear), which is what level-sets are good at *handling*, but also what makes the contour signal unreliable in the first place.

## What about *temporal retrieval* specifically?

The temporal-segmentation classics (CONDENSATION particle filters on
B-spline contours, dynamic-shape priors) buy you robust recovery after
occlusion **when the shape prior is strong** — e.g. tracking a hand or a
fish where the contour shape is highly constrained. A running person's
silhouette at 20-px scale is essentially shape-less and rotation-invariant
in thermal; the prior carries no information our Kalman+persistence
already has.

## Verdict

Not worth integrating. The conditions under which active contours add
value to a tracker (resolved target, edge or region contrast, informative
shape prior) are exactly the conditions our clip does **not** satisfy.
The one piece of this literature that *would* be relevant — Davis &
Sharma's contour-saliency for thermal pedestrians — assumes a static
camera, which we also do not have.

→ Skip. Move to Phase 2 (deep-learning block).

## What we *will* carry forward into Phase 2 from this scan

- Davis & Sharma's HOG-on-thermal trick for re-ID features.
- The region-based (Chan-Vese-style) idea that thermal segmentation
  should rely on **intensity statistics**, not gradients. Useful sanity
  check when picking pre-processing for the DL detector.
- The "shape prior is weak at this scale" observation — a useful prior
  when comparing DL architectures: shape-aware models (Mask R-CNN,
  segmentation-based) will overspend capacity on shape detail that does
  not exist at this resolution. A simple bbox detector (YOLO-style) is
  more proportionate to the signal.
