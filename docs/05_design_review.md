# Design Review — Five Pipelines, Their Algorithms, the Math,
# and Where to Push Each One Further

This document is the definitive technical reference for the five
pipelines built during the assignment. For each pipeline it lays out:

1. **Block diagram** — the actual functional decomposition.
2. **Per-frame algorithm** — the exact step-by-step procedure.
3. **The math** — the equations, with the role each quantity plays.
4. **What it does well** — the specific failure modes it cleanly
   handles.
5. **What it does poorly** — and the structural reason for it.
6. **Concrete next-step improvements** — actionable changes, ranked,
   that go beyond the current code.

Common building blocks (Kalman filter, HOG embedding, ego-motion,
persistence map, AND-gate, Mahalanobis gating, modality monitor) are
defined once in §0 and reused throughout.

The runnable code for everything below lives under `src/baseline/` and
each pipeline is selected via:

```bash
python scripts/run_on_video.py --pipeline {intensity, motion, dl, hybrid, follow}
```

---

## 0. Building blocks (used by ≥ 1 pipeline)

### 0.1  Pre-processing

Polarity detection on the first frame (per regime):
- Compute median `m` of the raw grayscale, plus two tails:
  $$ u = \mathbb{P}(I > m + 25), \qquad l = \mathbb{P}(I < m - 25) $$
- Scene is **white-hot** if $u \ge l$, else **black-hot**.
- If black-hot, replace `I` by `255 - I` so the target is always
  bright downstream.

CLAHE (Zuiderveld 1994) on the polarity-corrected grayscale:
$$ I_{\text{clahe}} = \mathrm{CLAHE}(I; \text{clip}, \text{grid}) $$
clips per-tile histograms to limit noise amplification; tile grid =
8×8, clip limit = 2.5.

Light Gaussian denoise (3×3) optional.

### 0.2  Top-hat morphology (detector for intensity pipeline)

For a structuring element $B$ (8-px ellipse here):
$$ T(I) = I - \mathrm{open}(I, B), \qquad
   \mathrm{open}(I, B) = (I \ominus B) \oplus B $$
extracts bright structures *smaller than* `B`. Robust to slow
background variation; cheap.

The detection threshold combines an absolute floor with a per-frame
Otsu cut on $T(I)$:
$$ \tau = \max(\tau_{\text{abs}}, \tau_{\text{Otsu}}(T)) $$
Binary mask = $\{ T \ge \tau \}$, morphologically opened, connected
components extracted; per component we compute a local contrast score
(LCM, §0.4) for ranking.

### 0.3  Ego-motion homography (motion / hybrid / follow)

Between consecutive raw-gray frames $I_{t-1}$ and $I_t$:

1. Detect ORB keypoints in both, descriptors $D_{t-1}, D_t$ (binary).
2. Brute-force match with Hamming + cross-check.
3. Keep the top ½ matches by distance (capped at 80 pairs).
4. Estimate the **3×3 homography** $H$ that maps points in $I_{t-1}$
   to $I_t$ via RANSAC (reprojection tolerance 3 px, ≥ 25 inliers
   required):
   $$ \tilde{p}_t \simeq H \, \tilde{p}_{t-1} $$
   in homogeneous coordinates.

If RANSAC fails or inliers < 25, we set $H = \emptyset$ and skip
motion compensation for that frame.

### 0.4  Motion-compensated persistence map

Warp the previous raw frame into the current frame's coordinates:
$$ \hat{I}_{t-1} = \mathrm{warp}(I_{t-1}, H) $$
Compensated diff (Gaussian-blurred, $k = 5$):
$$ D_t = G_k * \bigl|I_t - \hat{I}_{t-1}\bigr| $$
Bright in $D_t$ ⇔ motion that survives the camera warp.

Temporal persistence is an EMA in the moving frame:
$$ P_t = \alpha \cdot \mathrm{warp}(P_{t-1}, H) + (1-\alpha) \cdot D_t,
   \qquad \alpha = 0.6 $$
This way a *persistently* moving target accumulates response while
one-shot warp residuals decay (half-life ≈ 1.7 frames).

We detect on $P_t$ with a **data-driven anomaly threshold**:
$$ \tau_P = \max\bigl(\tau_{\text{floor}},\;
   \mathrm{med}(P_t) + k_{\text{MAD}} \cdot \sigma_P\bigr), $$
$$ \sigma_P \;=\; 1.4826 \cdot \mathrm{med}\bigl(|P_t -
   \mathrm{med}(P_t)|\bigr), $$
i.e. a $k$-MAD cut against the persistence map's own distribution
(MAD = Median Absolute Deviation; the 1.4826 scaling reproduces the
standard deviation for a Gaussian). $k_{\text{MAD}} = 4$. No
hand-set absolute level.

For each connected component (after open/close), the candidate score is
$$ s_{\text{motion}} = \overline{P_t}\bigr|_{\text{bbox}} \cdot
                       \sqrt{\text{area}} $$
$\sqrt{\text{area}}$ moderates the contribution of large warp-residual
sheets without giving away to one-pixel flashes.

### 0.5  Modality monitor (thermal ↔ IR ↔ colormap)

Per frame we compute a 32-bin normalised intensity histogram $h_t$ on
the raw gray channel. The dissimilarity to the previous accepted
histogram is the symmetric chi-square distance:
$$ \chi^2(h_t, h_{t-1}) = \tfrac{1}{2} \sum_i
   \frac{(h_t^{(i)} - h_{t-1}^{(i)})^2}{h_t^{(i)} + h_{t-1}^{(i)} + \varepsilon}$$
A modality switch is also flagged on a step-change of the
mean per-pixel $|R - B|$ (colour vs grayscale).

If $\chi^2 > 1.5$ **or** the colour/gray flag flips, **everything**
temporal — persistence map, polarity, tracker — is reset. A
10-frame cooldown prevents repeated triggers across one transition.

### 0.6  Bad-diff guard (motion / hybrid / follow)

A scene cut (or catastrophic homography failure) produces an unusually
high $D_t$ globally. We detect it as an anomaly against the rolling
history of recent per-frame diff medians:
$$ \text{bad}(t) \;\equiv\; \mathrm{med}(D_t) > \mu_h + k \cdot \sigma_h $$
with $\mu_h, \sigma_h$ the median / MAD-σ over the last 20 frames'
diff medians, and $k = 3$. On a bad frame the persistence map is
wiped but the tracker is *not* reset (its identity is preserved across
a single bad frame).

### 0.7  Kalman filter (all tracking pipelines)

State $x = [c_x, c_y, w, h, v_x, v_y]^\top \in \mathbb{R}^6$,
measurement $z = [c_x, c_y, w, h]^\top \in \mathbb{R}^4$.

Constant-velocity transition:
$$ F = \begin{pmatrix}
1 & 0 & 0 & 0 & 1 & 0 \\
0 & 1 & 0 & 0 & 0 & 1 \\
0 & 0 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 1
\end{pmatrix}, \quad
H = \begin{pmatrix} I_4 & 0_{4\times 2} \end{pmatrix}. $$

Process and measurement noise:
$$ Q = \mathrm{diag}(4,\,4,\,1,\,1,\,9,\,9), \quad
   R = \mathrm{diag}(1,\,1,\,4,\,4). $$
Larger velocity-process noise reflects the fact that, without
measurements, we should be highly uncertain about velocity (after
~10 coast frames the predicted velocity is essentially
non-informative). On every (re-)acquisition the **velocity is reset
to zero** and the covariance is restored to its initial value:
$P_0 = \mathrm{diag}(10,10,10,10,100,100)$. This prevents stale
velocity carrying through a re-init.

Predict:
$$ \hat{x}_{t|t-1} = F\,x_{t-1}, \quad
   P_{t|t-1} = F\,P_{t-1}\,F^\top + Q $$

Update on measurement $z_t$:
$$ S_t = H P_{t|t-1} H^\top + R, \quad
   K_t = P_{t|t-1} H^\top S_t^{-1}, $$
$$ x_t = \hat{x}_{t|t-1} + K_t(z_t - H \hat{x}_{t|t-1}), \quad
   P_t = (I - K_t H) P_{t|t-1}. $$

#### 0.7.1  Camera-motion-aware Kalman (StrongSORT / Deep-OC-SORT style)

Before each predict, the Kalman state is warped by the inter-frame
homography $H$. Position transforms by point projection; velocity by
finite-difference of two projected points:
$$ \begin{pmatrix} c_x' \\ c_y' \\ 1 \end{pmatrix} =
   H \begin{pmatrix} c_x \\ c_y \\ 1 \end{pmatrix}, \qquad
   \begin{pmatrix} t_x' \\ t_y' \\ 1 \end{pmatrix} =
   H \begin{pmatrix} c_x + v_x \\ c_y + v_y \\ 1 \end{pmatrix}, $$
$$ v_x' = t_x' - c_x', \quad v_y' = t_y' - c_y'. $$
Result: the Kalman prior lives in the **current** frame's pixel
coordinates, so during coast the predicted bbox doesn't drift
"backwards" relative to the moving scene.

#### 0.7.2  Mahalanobis validation gate

A candidate at center $(c, \cdot)$ is accepted by the gate iff
$$ d^2(c) = (c - \hat{c})^\top \Sigma_{xy}^{-1} (c - \hat{c}) \le \chi^2_{\text{gate}}, $$
where $\Sigma_{xy} = P_{t|t-1}[0\!:\!2, 0\!:\!2]$ is the 2×2 position
sub-covariance after predict, and $\chi^2_{\text{gate}} = 3^2 = 9$ (a
3-σ ellipse). A small pixel floor `gate_min_radius_px = 12` is added
so the gate never shrinks below a sane minimum when the filter thinks
it is very certain.

### 0.8  Appearance / re-identification

Two backends, selected automatically:

- **HOG descriptor** (used by default, no external download). 64×64
  grayscale crop, 8×8 cells, 2×2 blocks, 9 orientation bins → 1764-D
  vector, L2-normalised. Similarity is the cosine:
  $$ \mathrm{sim}(a, b) = \frac{\langle a, b\rangle}{\|a\| \|b\|} $$
  (which is just $a^\top b$ since both are unit-norm).
- **MobileNetV3-Small CNN** embedding. Same interface; only available
  when `download.pytorch.org` is reachable (it is not in our sandbox).

A **template bank** of the K = 8 most-recent embeddings is maintained
DeepSORT-style; a candidate's appearance score is the **best-of-K**
cosine similarity against the bank, with a near-duplicate guard that
prevents the bank from filling with identical templates.

### 0.9  AND-gate cross-validation

A CSRT measurement (§1.3) is accepted iff
$$ \mathrm{sim}_{\text{bank}}(\text{csrt patch}) \ge \tau_{\text{app}}
   \;\wedge\; z_P\bigr|_{\text{csrt bbox}} \ge \tau_z, $$
with $\tau_{\text{app}} = 0.30$ and $\tau_z = 2$. Two **independent**
evidence sources (visual gradient template, temporal motion) must
agree. This is the principled reason the tracker can refuse to
silently glue itself to a static hot rock.

When persistence is unavailable (the `dl` pipeline has no $P_t$), only
appearance is required.

### 0.10  ByteTrack-style two-stage association

In the hybrid pipeline, DL candidates with raw confidence above a
"strong" threshold (0.20) are processed **first** and only need an
appearance match — the spatial Mahalanobis gate is bypassed. Rationale:
a confident person-class detection is independent identity evidence, so
spatial inconsistency with a stale Kalman prediction shouldn't
disqualify it. Only if no priority candidate passes do we fall through
to the standard Mahalanobis search on all candidates.

### 0.11  DL detector

Pretrained YOLOv8n thermal-human model (`pitangent-ds/YOLOv8-human-detection-thermal`,
≈ 3 M params, single `HUMAN` class). 640-px inference, NMS at IoU = 0.45,
returns up to 30 detections; we keep the top 10. **Polarity is
corrected before inference**: the model was trained on white-hot;
black-hot frames are bit-inverted before being passed in.

---

## 1. Pipeline `intensity` (v1)

### 1.1  Diagram

```
frame ─► preprocess ─► top-hat T(I) ─► Otsu+floor thresh ─► CC + shape gate
                                                                  │
                                          ┌──────────── ranked ───┘
                                          ▼
                                     ┌─────────────────────────┐
                                     │ CSRT + Kalman + LCM rank│
                                     └─────────────────────────┘
                                          │
                                          ▼
                                       annotated
```

### 1.2  Per-frame algorithm

1. Polarity → CLAHE → Gaussian blur.
2. Top-hat $T = I - \mathrm{open}(I, B)$.
3. Binary: $\{ T \ge \max(\tau_{\text{abs}}, \tau_{\text{Otsu}}) \}$.
4. Morphological open → connected components → area/aspect filter.
5. Score each component by **local contrast** (LCM, Chen 2014):
   $$ s_{\text{LCM}} = \overline{I}|_{\text{inner}} -
                       \overline{I}|_{\text{ring}}, $$
   inner = bbox, ring = a surrounding pad of the same scale with the
   inner mass subtracted.
6. Track: CSRT for short-term, Kalman for motion.

### 1.3  CSRT in one paragraph

CSRT (Lukežič 2017) is a discriminative correlation filter learnt
online on the target patch. The filter $w$ is updated by minimising
$$ \|w \star x - y\|_2^2 + \lambda \|w\|_2^2 $$
in the frequency domain (via FFT), where $y$ is a Gaussian-shaped
ideal response. Channel and spatial reliability terms reweight the
contributions of unreliable features. On thermal grayscale it runs at
about 30 fps and survives small appearance changes.

### 1.4  Strengths

- Cheap (~ 20 fps CPU).
- Works when the target is a clear bright blob against a darker
  background, exactly the textbook IR small-target regime.
- All blocks are interpretable; debugging is easy.

### 1.5  Weaknesses

- **No motion cue** — picks any bright blob, including hot rocks.
- **Auto-init takes the top blob at frame 0**, which is rarely the
  target on our clip.
- **Frozen polarity from frame 0** invalidates the threshold after a
  modality switch.
- **CSRT drifts onto a similar texture during occlusion** and
  reports "TRACKING" while the box sits on a rock.

### 1.6  Improvement directions

- Initialise on a *motion-aware* candidate (already done in `motion`).
- Replace CSRT with a learned discriminative tracker (DiMP family) —
  out of CPU budget.
- Treat the LCM score as one signal among many in a soft scoring rule
  rather than as the ranker.

---

## 2. Pipeline `motion` (final classical baseline)

### 2.1  Diagram

```
                  ┌─────────────────┐
[I_t] ─►│preproc│  │ modality monitor│ ─►(switch ⇒ hard reset)
                  └────────┬────────┘
                           ▼
[I_t]──► ORB+RANSAC ── H ─► compensated diff D_t ─► EMA persistence P_t
                                                       │
                                                       ▼
                                          k-MAD threshold + CC + shape
                                                       │
                                                       ▼
                                          ranked motion candidates
                                                       │
                                                       ▼
            ┌──────────────────────────────────────────┴───┐
            │ CSRT  +  Kalman (state-warp by H)  +  AND-gate │
            │ (appearance HOG bank ∧ persistence z-score)  │
            │ Mahalanobis re-acquisition gate              │
            └────────────────┬─────────────────────────────┘
                             ▼
                         annotated
```

### 2.2  Per-frame algorithm

1. **Modality monitor**: if switch, hard-reset persistence and tracker.
2. **Polarity** + CLAHE (re-evaluated after each modality switch).
3. **Ego-motion**: ORB + RANSAC homography $H$ on raw gray.
4. **Compensated diff** $D_t$ (§0.4); rolling **bad-diff guard**.
5. **Persistence EMA** $P_t$, warped each frame to track the camera.
6. **Detection** on $P_t$: $k$-MAD anomaly threshold → CC →
   shape/area filter → score $\bar{P} \cdot \sqrt{\text{area}}$.
7. **Init**: top candidate's persistence z-score must clear 4 for 3
   consecutive frames within $0.12 \cdot \mathrm{diag}$ of its
   previous location.
8. **Track**:
   - Warp Kalman state by $H$ → predict.
   - CSRT update → AND-gate (HOG-bank cosine $\ge 0.30$ AND
     persistence z $\ge 2$).
   - If rejected: Mahalanobis search; if found, **velocity-reseed**
     Kalman and re-init CSRT.
   - Else coast (TRACKING → COASTING → LOST after 30 frames).

### 2.3  Strengths

- Fully **unsupervised** init (no `--init-bbox`).
- **Camera-motion-compensated** detection and tracking from
  end to end.
- **AND-gate** prevents silent drift: the tracker can't claim
  TRACKING while glued to a static patch with no motion evidence.
- **Data-driven thresholds** (median + MAD) instead of fixed pixel
  constants; portable across clips without retuning.
- **Modality monitor** handles the thermal↔IR switch automatically.

### 2.4  Weaknesses

- No "person" prior. Any consistently moving blob can be picked
  (e.g. a moving shadow on a dirt path).
- Persistence decays in ~ 2 frames; a momentarily stationary
  walker becomes invisible and the tracker enters COASTING.
- ORB+RANSAC fails on near-textureless segments (uniform sky or
  monochromatic canopy), forcing a $H = \emptyset$ frame and losing
  motion-compensation.
- Honest reporting causes visible **TRACKING ↔ COASTING flicker**
  that reads to a human viewer as instability.

### 2.5  Improvement directions

- **Longer-life persistence** ($\alpha = 0.85$) so a briefly-stopped
  target stays visible — at the price of accumulating warp residuals.
- **Hysteresis on the AND-gate**: stricter at acquisition, looser
  while already TRACKING (LTMU-style two-thresholds).
- **Switch to dense Lucas-Kanade** when ORB has too few inliers (low
  texture).
- **Adaptive $k_{\text{MAD}}$** based on persistence map entropy:
  more lenient when the map is sparse, stricter when dense.

---

## 3. Pipeline `dl` (pure deep-learning detection + tracker)

### 3.1  Diagram

```
[I_t] ─► YOLOv8n-thermal (polarity-corrected) ─► person candidates
                                                       │
                                                       ▼
                              CSRT + Kalman + HOG appearance-only
                              (no persistence ⇒ AND-gate degrades
                               to appearance-only acceptance)
                                                       │
                                                       ▼
                                                   annotated
```

### 3.2  Per-frame algorithm

1. Polarity → invert frame if black-hot.
2. YOLOv8n inference; keep top-K = 10 person detections with
   conf ≥ 0.10.
3. Init: first detection with conf ≥ 0.25 that persists in the
   same neighbourhood for 3 frames.
4. Track: CSRT + Kalman (no camera-motion compensation — no
   homography is computed in this pipeline), HOG appearance with the
   sliding bank; no persistence map means the AND-gate degrades to
   appearance-only.

### 3.3  Strengths

- **Person-class prior** is the strongest signal we have when the
  detector fires.
- Highest TRACKING% on this clip (83 %).
- No hand-tuned thresholds inside the detector — the network learnt
  them.

### 3.4  Weaknesses

- The pretrained model misses a large fraction of frames in the
  colormapped IR segment and during heavy canopy occlusion. The
  tracker then has no measurement and either coasts on stale
  velocity or accepts a CSRT measurement of dubious quality.
- **No camera-motion compensation**: during coast in this pipeline,
  the Kalman prediction drifts opposite to the drone pan because
  velocity is in absolute pixel coordinates.
- **No "is this a target?" secondary signal**: when CSRT drifts onto
  a similar-looking patch, only HOG appearance pushes back, which
  isn't enough to prevent slow drift.
- **Dependent on the specific checkpoint's training distribution**:
  swapping the YOLO model changes everything downstream.

### 3.5  Improvement directions

- **Add camera-motion warp to the Kalman state** even in DL mode
  (we already estimate it elsewhere — just need to plumb it).
- **Use detector confidence in the appearance EMA**: only update
  appearance from high-conf detections, never from low-conf or
  CSRT-only measurements (prevents bank contamination).
- **A larger thermal-trained detector** (e.g. YOLOv8s/m fine-tuned
  on HIT-UAV) — would directly raise recall on the missing frames.

---

## 4. Pipeline `hybrid` (DL ∪ motion, identity-preserving)

### 4.1  Diagram

```
                            ┌──────────────────────┐
[I_t] ─►│ modality monitor │ ─► reset on switch
        └─────────┬────────┘
                  ▼
        ┌──────────────────────────────────────┐
        │ preprocess (gray, polarity, CLAHE)   │
        └──┬──────────────────────┬────────────┘
           ▼                      ▼
   ┌──────────────┐     ┌──────────────────────┐
   │ DL detector  │     │ ego-motion + persist │
   │ (polarity-c.)│     │ + motion candidates  │
   └──────┬───────┘     └────────────┬─────────┘
          │ DL cands + conf          │ motion cands + z
          └─────────┬────────────────┘
                    ▼
             ┌──────────────────────────────┐
             │ Fuse: each candidate gets a  │
             │ joint score combining DL conf│
             │ and persistence z            │
             └──────────────┬───────────────┘
                            ▼
            ┌──────────────────────────────┐
            │ Tracker w/ camera-motion-    │
            │ warp + ByteTrack 2-stage +   │
            │ AND-gate + Mahalanobis +     │
            │ HOG template bank            │
            └──────────────┬───────────────┘
                           ▼
                       annotated
```

### 4.2  Joint scoring

For a DL candidate with raw confidence $c$ and bbox $b$, joint score:
$$ s(b) = c + \lambda \cdot \max(z_P(b), 0), \qquad \lambda = 0.05. $$

For a motion candidate scored on $P_t$, the joint score is comparable:
$$ s(b) = \tanh\bigl(z_P(b)/4\bigr) + \lambda \cdot \max(z_P(b), 0). $$

The $\tanh$ maps an unbounded z-score to $[0, 1)$ so it can compete
with the DL conf on the same scale (in practice motion candidates with
high z still outrank low-conf DL — one of the imbalances flagged in
§4.5).

### 4.3  Per-frame algorithm (in addition to §2.2 and §3.2)

1. Run DL detector with polarity-correction.
2. Run motion pipeline → persistence map and motion candidates.
3. **Fuse** the two candidate sets (no IoU dedup).
4. **Two-stage matching**:
   - Priority pool = DL detections with raw conf $\ge 0.20$.
   - Try CSRT, AND-gate the result.
   - If CSRT rejected: try priority pool by **appearance match
     alone** (no spatial gate).
   - Else: Mahalanobis search over the full fused pool.
5. Camera-motion-warp Kalman state with $H$.
6. Update HOG bank only on successful, AND-gated measurements.

### 4.4  Strengths

- All the benefits of `motion` (camera compensation, modality reset,
  unsupervised init) **plus** a person-class prior.
- 0 % LOST on this clip — the tracker always has *some* evidence
  source firing.
- Two-stage matching means a high-conf DL detection can re-acquire
  identity even when the Kalman gate is stale.

### 4.5  Weaknesses

- **Scoring imbalance**: motion candidates with high $z$ can outrank
  legitimate low-conf DL detections, biasing init toward strong
  moving non-target blobs.
- **Template bank contamination**: when the tracker accepts a
  near-target patch, that patch enters the bank; subsequent matches
  become permissive in the wrong direction.
- **More moving parts**: the pipeline is the union of two systems,
  each with its own failure modes. Diagnosing a single bad frame
  requires looking at both streams.
- Compute cost ~ 3 fps CPU (YOLO + ORB + persistence per frame).

### 4.6  Improvement directions

- **IoU-based fusion**: when a DL bbox and a motion bbox have
  IoU $\ge 0.3$ they almost certainly describe the same target;
  merge them into a single candidate with both signals.
- **Conditional bank update**: only push to the HOG bank when *both*
  DL conf and persistence z are strong (joint update rule). Prevents
  contamination.
- **Mahalanobis-gated priority**: priority candidates that are
  spatially impossible (mahal² > 25) should still be rejected — not
  every "confident DL" detection is the right target.
- **CNN re-ID** when sandbox restrictions are lifted (already wired
  in `reid.py`).

---

## 5. Pipeline `follow` (detector-following, philosophy change)

### 5.1  Diagram

```
[I_t] ─► YOLOv8n (polarity-c.) ──┐
                                 │
[I_t] ─► persistence + motion ──┐│
                                ▼▼
                  ┌─────────────────────────────┐
                  │ Selection rule:             │
                  │ 1) closest DL to predicted  │
                  │    position (conf ≥ 0.08)   │
                  │ 2) strongest DL anywhere    │
                  │    (conf ≥ 0.15)            │
                  │ 3) closest motion candidate │
                  │    with z ≥ 4 to prediction │
                  │    (only when already on    │
                  │     a track)                │
                  └──────────────┬──────────────┘
                                 ▼
                  ┌─────────────────────────────┐
                  │ Kalman (state-warp by H)    │
                  │ — smoothing + gap fill only │
                  └──────────────┬──────────────┘
                                 ▼
                             annotated
```

### 5.2  Per-frame algorithm

1. Modality + preprocess + polarity, as before.
2. Compute homography $H$ and persistence $P_t$ (for fallback only).
3. Get DL candidates (polarity-corrected); get motion candidates.
4. Camera-motion-warp the Kalman state (if any track).
5. Run **selection rule** (§5.1, three rules in order).
   - The motion fallback (rule 3) is **disabled when there is no
     prior track** — motion alone can't decide "is this a person?",
     so we refuse to seed identity from it.
6. If a winner is found: Kalman.predict + correct on the winner's
   bbox.
7. Else: Kalman.predict only; advance coast counter; LOST after 30.

### 5.3  Strengths

- The simplest, most directly auditable design.
- No template-clinging → when the tracker claims TRACKING, the box
  is genuinely on a current detection.
- The motion fallback covers the YOLO-misses-the-person gap *without*
  letting motion contaminate identity at init time.

### 5.4  Weaknesses

- Lowest TRACKING% — when YOLO misses, the tracker honestly coasts.
- No appearance ReID at all; two different people in the frame would
  be indistinguishable (not relevant for single-target but a
  weakness if the scene grew).
- The proximity radius is a fraction of the frame diagonal; for a
  very small target this is loose enough that a wrong detection
  nearby can be picked.

### 5.5  Improvement directions

- **Soft proximity scoring** instead of a hard radius: score
  $\propto e^{-d^2/(2\sigma^2)}$ multiplied by DL conf. Closer is
  better but a strong DL detection a bit farther can still win.
- **Detector confidence smoothing**: track an EMA of recent DL conf;
  when it drops, raise the bar for accepting any new candidate to
  prevent a low-conf detection in a noisy frame from hijacking the
  track.
- **Re-introduce light appearance check (HOG cosine ≥ 0.3)** in
  the proximity test so two simultaneously-detected people are
  disambiguated by appearance, not only by distance.
- **A better thermal detector** is the single biggest improvement
  path for *this* pipeline, because its TRACKING% is bounded above
  by the detector's per-frame recall.

---

## 6. Cross-pipeline comparison

| Property                       | intensity | motion | dl | hybrid | follow |
|---|---|---|---|---|---|
| Person prior                   | ✗ | ✗ | ✓ | ✓ | ✓ |
| Motion / temporal evidence     | ✗ | ✓ | ✗ | ✓ | (fallback) |
| Camera-motion-warped Kalman    | ✗ | ✓ | ✗ | ✓ | ✓ |
| Modality switch handling       | ✗ | ✓ | ✗ | ✓ | ✓ |
| AND-gate cross-validation      | ✗ | ✓ | partial | ✓ | (implicit) |
| Identity preservation          | ✓ | ✓ | ✓ | ✓ | ✗ |
| Unsupervised init              | ✓ | ✓ | ✓ | ✓ | ✓ |
| TRACKING %                     | 66 | 65 | 83 | 73 | 42 |
| COASTING %                     | 13 | 21 | 12 | 26 | 34 |
| LOST %                         | 21 | 13 | 5  | 0  | 24 |
| CPU fps (CPU-only sandbox)     | 20 | 10 | 5  | 3  | 6 |

The TRACKING% column is **not** an accuracy ranking — it is partly a
function of how honestly each pipeline reports uncertainty. The pure
`dl` pipeline reports TRACKING 83 % partly because it has no
mechanism to refuse a stale CSRT measurement; the `follow` pipeline
reports TRACKING 42 % because it only ever claims TRACKING when the
detector actually fires near the predicted spot.

---

## 7. The end-to-end improvement directions worth pursuing

Ranked by expected impact for *this* assignment's deliverable:

1. **Get a better thermal-aerial detector.** Every downstream design
   inherits the detector's per-frame recall as an upper bound on
   honest TRACKING%. A larger or HIT-UAV-fine-tuned YOLO would lift
   the ceiling for `dl`, `hybrid`, and `follow` simultaneously.
2. **IoU-based fusion + conditional bank update.** Fixes the two
   most-cited weaknesses of the current `hybrid`: motion outranking
   DL at init, and the appearance bank slowly absorbing wrong
   templates.
3. **Hysteresis on the AND-gate.** Maintain strict thresholds at
   acquisition; loosen them while already TRACKING (LTMU-style).
   Should eliminate the visible flicker the `motion` and `hybrid`
   pipelines have.
4. **CNN re-ID** (already coded — `reid.py` falls back to HOG when
   `download.pytorch.org` is blocked). Replace HOG by MobileNetV3
   embeddings in an offline build environment to gain robustness
   to small appearance changes.
5. **Soft proximity scoring** for `follow` (Gaussian × DL conf
   instead of hard radius) — should raise its TRACKING% materially
   without sacrificing identity robustness.
6. **Online discriminative model (DiMP-family) on a GPU machine.**
   Out of CPU/inference scope here but the single biggest jump in
   the literature, and where a real production system would go next.

These six interventions are mutually compatible, each is independently
testable in the current architecture (every block is swap-out-able by
construction), and each addresses a specific weakness identified in
§§ 1–5 rather than a generic "make it better."
