# Literature Review — Classical Detection & Tracking on Thermal/IR Aerial Imagery

Working notes for the non-DL baseline. Scope: a *person-scale* target in
aerial thermal-style footage, partially occluded by foliage, moving
camera, low texture.

---

## 1. Why thermal ≠ RGB (and why classical RGB recipes break)

| Aspect | RGB | Thermal / LWIR |
|---|---|---|
| Channels | 3 (color + brightness) | 1 (apparent radiance) |
| Polarity | Fixed | White-hot vs black-hot, can invert per AGC mode |
| Dynamic range | 8-bit display, scene-bound | 14–16-bit sensor, AGC re-stretches each frame → **global intensity is unstable** |
| Texture | Rich (albedo, color edges) | Sparse (only thermal gradients), edges often soft |
| Discriminative cues | Color, texture, shape | Mostly intensity + crude shape; **no color histogram tricks** |
| Confusers | Color-matched objects | Sun-warmed rocks, asphalt, fires, fauna, NUC artefacts, "halo" around hot bodies |
| Subject signature | ~constant | Depends on clothing, ambient T, sweat, wet clothes hide a person almost completely |

**Practical consequences for our pipeline:**
- No color histograms → mean-shift / CAMShift / Staple-color channels are out.
- AGC drift means *absolute* threshold values are unreliable across the clip; we need **adaptive / per-frame** normalization (CLAHE) or **relative** features (top-hat, local contrast).
- A person can look identical to a hot rock for tens of frames → identity is fragile; we lean harder on **motion continuity** (Kalman) than on appearance.
- Polarity should be detected, not assumed.

---

## 2. Classical detection of warm targets in IR

Two regimes in the IR literature: **small/dim target** (target is a few-pixel
blob, typical for long-range surveillance) and **extended target** (target is
resolved, e.g. a person at moderate range from a drone). Our clip is closer
to the *extended-but-small* regime — the person occupies tens to a few
hundred pixels.

**Family A — Morphological / contrast filters** (workhorse baselines)
- **Top-hat transform** (Tom et al., 1993; Deshpande et al., 1999): `I − open(I, B)` extracts bright structures smaller than the structuring element `B`. Robust to slow background variation, cheap. Sensitive to `B` size.
- **Local Contrast Measure (LCM)** (Chen et al., TGRS 2014) and **Multi-scale Patch Contrast Measure (MPCM)** (Wei et al., 2016): compute the ratio of a centre patch to its neighbours, multi-scale. Suppresses cluttered backgrounds better than top-hat.
- **Max-Median / Max-Mean filters** (Deshpande): cheap clutter rejection.

**Family B — Statistical / decomposition models**
- **IPI — Infrared Patch-Image model** (Gao et al., TIP 2013): low-rank background + sparse target via RPCA. Strong but slow; not realtime out-of-the-box.
- **NRAM / RIPT / PSTNN** (2017–2019): tensor extensions of IPI. Same trade-off.

**Family C — Blob & region descriptors**
- **MSER** (Matas et al., 2002): extremal regions stable across thresholds — naturally matches "hot blob on cooler background." Good fit for thermal.
- **Adaptive thresholding** (Otsu per ROI, Niblack, Sauvola): pairs well with top-hat as a final binarisation step.

**Family D — Motion-based (background subtraction)**
- **MOG2 / KNN / ViBe**: assume static or near-static camera. **Our camera moves** → these fail unless we ego-motion-compensate first.
- **Frame differencing + homography warp** (estimate global motion with ORB+RANSAC, warp previous frame, then diff): a usable middle ground, but expensive and fails on parallax-heavy aerial scenes.

**Recommendation for the baseline detector:** top-hat → adaptive threshold
→ connected-component analysis → shape/area filter. Add LCM as a confidence
score for ranking candidates. Skip RPCA/IPI for the baseline (latency).

---

## 3. Classical single-object tracking

Once detected, we need to keep identity through occlusion. Established
non-DL trackers (most available in OpenCV's `cv2.legacy` / `cv2`):

| Tracker | Idea | Strengths | Weaknesses (thermal-relevant) |
|---|---|---|---|
| **MOSSE** (Bolme et al., CVPR 2010) | Minimum-output-sum-of-squared-error correlation filter on grayscale | Very fast (~600 fps), grayscale-native | No scale adaptation; drifts under heavy occlusion |
| **KCF** (Henriques et al., TPAMI 2014) | Kernelised correlation filter | Fast, robust to small appearance change | No scale adaptation, single channel only in baseline form |
| **DSST** (Danelljan et al., BMVC 2014) | KCF + separate scale filter | Handles scale, still fast | Drifts on long occlusions |
| **CSRT** (Lukežič et al., CVPR 2017) | Channel & spatial reliability DCF | Most accurate classical tracker in OpenCV, handles non-rectangular targets | ~25 fps, can lose target if occluded long |
| **MedianFlow** (Kalal et al., ICPR 2010) | Forward-backward Lucas-Kanade consistency | Detects its own failure cleanly | Needs textured target; thermal targets often too smooth |
| **Mean-Shift / CAMShift** | Colour-histogram mode-seeking | — | **Useless on monochannel thermal** |

**Occlusion handling — the real problem.** A pure DCF tracker has no memory.
Standard classical recipe:
1. **Kalman filter** with constant-velocity (or constant-acceleration) state on the bounding-box centre + size. Provides a *prediction* the tracker can fall back on.
2. **Track-quality gate** (PSR — Peak-to-Sidelobe Ratio for DCF trackers; Bolme 2010 reports PSR < 7 ≈ lost). When the tracker reports low confidence, freeze the appearance model and let Kalman coast.
3. **Re-detection / re-acquisition**: re-run the detector inside a Kalman-predicted gate; greedily match the best candidate by IoU + appearance similarity (intensity histogram, HOG cosine).

This Kalman-gated, re-detect-on-lost loop is essentially the classical
single-target analogue of SORT (Bewley et al., ICIP 2016).

---

## 4. Thermal-specific tracking tweaks worth lifting

- **Polarity normalisation** at clip start: estimate target polarity (warmer or cooler than median background) on the first detection and freeze it.
- **CLAHE** (Zuiderveld, 1994) per frame to fight AGC drift before any feature extraction.
- **Halo suppression** with a small morphological closing before top-hat.
- **HOG descriptors** still work on thermal (gradients survive); used by Davis & Keck (WACV 2005) for person detection in LWIR, and by FLIR's own benchmarks. We can use HOG as the *appearance* signature for re-ID inside the Kalman gate, instead of a colour histogram.

---

## 5. What we will (and won't) build for the baseline

**In scope (baseline):**
- Frame I/O + visualiser helpers.
- Preprocessing: polarity detection, CLAHE, light denoise.
- Detector: top-hat + adaptive threshold + CC filtering, ranked by LCM-style local contrast.
- Tracker: CSRT (OpenCV) for short-term + Kalman filter for motion + re-detection on PSR/score drop.
- End-to-end runner on the 40 s clip with annotated output video.

**Out of scope (left for the DL system):**
- IPI / tensor-RPCA detectors (too slow for realtime).
- Learning-based re-ID embeddings.
- Multi-object tracking (assignment is single-target).

---

## 6. Honest expectations

On this specific clip the classical baseline will likely:
- Work when the person is a clear bright blob against a darker canopy.
- Lose identity when the person sits under dense canopy for more than ~1–2 s (no signature, Kalman coast will drift).
- Confuse the target with sun-warmed rocks or fauna of similar size.

That gap — exactly where a learned appearance/temporal model helps — is
the motivation we'll use to justify the deep-learning block in step 2.

---

## References (foundational; for the eventual write-up)

- Tom, V. T., et al. "Morphology-based algorithm for point target detection in IR imagery." SPIE, 1993.
- Deshpande, S. D., et al. "Max-mean and max-median filters for detection of small targets." SPIE, 1999.
- Chen, C. L. P., et al. "A local contrast method for small infrared target detection." IEEE TGRS, 2014.
- Wei, Y., et al. "Multiscale patch-based contrast measure for small infrared target detection." Pattern Recognition, 2016.
- Gao, C., et al. "Infrared patch-image model for small target detection in a single image." IEEE TIP, 2013.
- Matas, J., et al. "Robust wide baseline stereo from maximally stable extremal regions." BMVC, 2002.
- Bolme, D. S., et al. "Visual object tracking using adaptive correlation filters." CVPR, 2010.
- Henriques, J. F., et al. "High-speed tracking with kernelized correlation filters." IEEE TPAMI, 2014.
- Danelljan, M., et al. "Accurate scale estimation for robust visual tracking." BMVC, 2014.
- Lukežič, A., et al. "Discriminative correlation filter with channel and spatial reliability." CVPR, 2017.
- Kalal, Z., et al. "Forward-backward error: Automatic detection of tracking failures." ICPR, 2010.
- Bewley, A., et al. "Simple online and realtime tracking." ICIP, 2016.
- Davis, J. W., & Keck, M. A. "A two-stage template approach to person detection in thermal imagery." WACV, 2005.
- Zuiderveld, K. "Contrast limited adaptive histogram equalization." Graphics Gems IV, 1994.
