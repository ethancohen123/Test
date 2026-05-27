# Hybrid SOT — what best-in-class designs actually do, and what we should adopt

Working notes for redesigning the hybrid pipeline. The current naïve
union-and-AND-gate is not enough. This doc surveys what modern systems
do that we are not yet doing, and picks the highest-impact, lowest-cost
imports for our exact scenario (aerial thermal, single moving person,
CPU inference).

---

## 1. What state-of-the-art hybrids actually do

Methods examined: **ByteTrack**, **OC-SORT / Deep OC-SORT**,
**DeepSORT / StrongSORT**, **LTMU** (Long-Term Meta-Updater),
**KeepTrack**, **STMTrack** (space-time memory), **HSSNet** (Siamese
TIR), **PDAT-CAR** (progressive domain adaptation TIR).

| Idea | Where it comes from | What we're missing |
|---|---|---|
| **Use *all* detections, not just high-conf** — two-stage matching: high-conf first (assigns tracks), low-conf fills in for unmatched tracks (keeps them alive through occlusion) | ByteTrack, ICCV 2022 | We naively union DL + motion and pick top-scored. No staging. |
| **Camera-motion compensation inside the Kalman state** — apply the inter-frame homography to the prior before Kalman predict, so position prediction accounts for ego-motion. | StrongSORT (ECC), Deep OC-SORT | Our Kalman lives in pixel coordinates → drifts during coast on a moving drone. We **already estimate H** for the motion detector; we just never feed it to the Kalman. |
| **CNN re-ID embedding instead of patch-NCC** — extract a feature vector per bbox from a small CNN, match by cosine distance, maintain a feature bank | DeepSORT, StrongSORT, Deep OC-SORT | Our appearance is a 32×32 mean-subtracted patch with NCC. Brittle to small appearance changes; useless across the modality switch. |
| **Feature/template bank with sliding window of past-K embeddings** | DeepSORT (100-frame bank), STMTrack (space-time memory) | We keep one EMA patch. One stale appearance drifts; multiple recent ones don't. |
| **Validation / meta-update step** — a small classifier decides each frame whether the current track is still on target; triggers global re-detection when not | LTMU, CVPR 2020 | We use the AND-gate (appearance ∧ motion) for this, but it's binary and brittle. |
| **Distractor modelling** — explicit candidates of "things that look similar but aren't the target" near the predicted location, so the tracker chooses between target and distractor instead of just "best score" | KeepTrack, ICCV 2021 | We have no notion of distractors. Hot rocks and the actual person both get fused candidates indistinguishably. |
| **Online discriminative target model** (DiMP/PrDiMP) — learn a small classifier online that says target-vs-background for this clip | DiMP, ICCV 2019; PrDiMP, CVPR 2020 | Beyond our CPU-inference budget; needs GPU. We approximate this much more crudely via the persistence map. |
| **Gaussian-smoothed interpolation during gaps** — once a track is matched after a coast period, retroactively interpolate the bbox through the gap | StrongSORT (GSI) | We do nothing — the box just sits at the last Kalman prediction. Cosmetic but matters for the demo video and for any downstream metrics. |
| **Adaptive gate width by track quality** — strict when track is recently confirmed, lenient when track is uncertain | LTMU, KeepTrack | Our Mahalanobis gate already widens with covariance; this is the *same* idea expressed differently. We have this. |

## 2. The single most-important gap for our scenario

The drone moves; the target moves; our Kalman state lives in pixel
coordinates and assumes the *camera* is stationary. So during any coast
period, the Kalman prediction stays in absolute image coordinates while
the camera pans the scene under it. Whichever way the camera moves, our
"prediction" is wrong by exactly the camera's translation.

This is **the** standard root cause of mid-air-target tracker failures
and the explicit motivation for StrongSORT's ECC and Deep OC-SORT's
camera-compensated Kalman. We already estimate the inter-frame
homography for the persistence map — we just never apply it to the
Kalman state. Cost: ~10 lines. Expected impact: large, particularly on
camera-pan segments.

## 3. What we should adopt (ranked by impact / cost)

| # | Adoption | Effort | Expected impact |
|---|---|---|---|
| **A** | **Camera-motion-warped Kalman** — apply the per-frame homography to `statePost` before `predict()`. Inherited from StrongSORT / Deep OC-SORT. | ~30 LOC | ★★★★★ |
| **B** | **CNN appearance embedding** — replace 32×32 patch NCC with a small ImageNet-pretrained backbone (torchvision MobileNetV3-Small / ResNet-18). Cosine distance for matching. | ~80 LOC | ★★★★ |
| **C** | **Template bank** with sliding window of the last K embeddings; match against best-of-K, not against an EMA. Inherited from DeepSORT. | ~30 LOC | ★★★ |
| **D** | **Two-stage ByteTrack association** — separate DL candidates into high-conf (≥ τ_h) and low-conf (τ_l ≤ c < τ_h). Match high-conf first; use low-conf + motion candidates only for unmatched tracks. | ~50 LOC | ★★★ |
| **E** | **NMS-based fusion** — IoU-dedupe between DL and motion candidates so we don't double-count the same blob with two scores. | ~20 LOC | ★★ |
| **F** | **Gaussian-smoothed interpolation across COAST gaps** in post — once the track is re-acquired, smooth the bbox through the gap for the rendered video. | ~30 LOC | ★ (cosmetic + metric) |

## 4. What we are explicitly *not* doing (and why)

- **DiMP / PrDiMP / KeepTrack / STMTrack online learners.** All require
  per-clip gradient steps; we are CPU-only and inference-only.
- **Transformer trackers (OSTrack, MixFormer, AiATrack).** Literature
  (CST Anti-UAV, ICCV-W 2025) reports they collapse on tiny TIR targets
  with camera motion — exactly our regime.
- **Multi-hypothesis tracking.** Single-target task; overkill.
- **Re-training / fine-tuning on this clip.** Out of scope per the
  assignment.

## 5. Recommended sequence

The four-letter combination **A + B + C + D** is what would close most
of the gap between our current hybrid and a defensible best-in-class
single-target tracker for this regime, while staying inside the CPU /
inference-only budget. **A** alone will already move the needle a lot
on the moving-camera segments; **B + C** address the "tracker glued to
the wrong thing" pattern; **D** improves which candidate we even
consider.

## References

- Zhang, Y., et al. *ByteTrack: Multi-Object Tracking by Associating
  Every Detection Box.* ECCV 2022.
- Du, Y., et al. *StrongSORT: Make DeepSORT Great Again.* IEEE TMM 2023.
- Maggiolino, G., et al. *Deep OC-SORT: Multi-Pedestrian Tracking by
  Adaptive Re-Identification.* 2023.
- Wojke, N., et al. *Simple Online and Realtime Tracking with a Deep
  Association Metric (DeepSORT).* ICIP 2017.
- Dai, K., et al. *High-Performance Long-Term Tracking with
  Meta-Updater (LTMU).* CVPR 2020.
- Mayer, C., et al. *Learning Target Candidate Association to Keep
  Track of What Not to Track (KeepTrack).* ICCV 2021.
- Fu, Z., et al. *STMTrack: Template-free Visual Tracking with
  Space-time Memory Networks.* CVPR 2021.
- Bhat, G., et al. *Learning Discriminative Model Prediction for
  Tracking (DiMP).* ICCV 2019.
- Liu, Q., et al. *Hierarchical Spatial-aware Siamese Network for
  Thermal Infrared Object Tracking (HSSNet).* Knowledge-Based Systems
  2019.
- Cao, J., et al. *Observation-Centric SORT (OC-SORT): Rethinking SORT
  for Robust Multi-Object Tracking.* CVPR 2023.
