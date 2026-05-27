# Multi-object tracking with persistent IDs and re-identification
# — what the literature does, and what we'll build

## 0. The feature in one sentence

For every frame: YOLO produces *N* detections; each detection gets
an **integer ID**; the same physical person keeps the same ID over
time, **including across short disappearances** (occlusion, leaving
and re-entering the frame).

This is the textbook multi-object tracking (MOT) problem with
**re-identification (ReID)** on top of track death/birth.

---

## 1. What the literature does

### 1.1  The canonical pipeline (used by everyone since 2017)

```
[detections this frame] ──► [cost matrix: detection × active track]
                                          │
                                          ▼
                          ┌──────────────────────────────────┐
                          │ Linear assignment (Hungarian /   │
                          │ greedy) gated by:                │
                          │   - motion (IoU or Mahalanobis)  │
                          │   - appearance (ReID embedding)  │
                          └──────────────┬───────────────────┘
                                         ▼
        ┌────────────────────────────────┼────────────────────────────┐
        ▼                                ▼                            ▼
  matched (det, track)         unmatched detections          unmatched tracks
        │                                │                            │
        ▼                                ▼                            ▼
  update Kalman,              try matching against        coast Kalman;
  refresh appearance,         "lost" tracks by             after K frames
  push to bank                appearance only;             without measurement,
                              if no match,                 move to "lost" pool
                              create new track ID
```

Track lifecycle:

```
NEW ──► TENTATIVE (k consecutive hits) ──► CONFIRMED
                                              │
                                              ▼
                                       COASTING (k coast frames)
                                              │
                                              ▼
                                            LOST  ─────► re-acquired
                                              │           (back to CONFIRMED
                                              ▼            with same ID)
                                          expired
                                          (forgotten)
```

### 1.2  The four trackers worth knowing

| Tracker | Year | Key idea | Re-ID? | What we already have |
|---|---|---|---|---|
| **SORT** (Bewley et al., ICIP 2016) | 2016 | Kalman + Hungarian on IoU. Fast, simple, no appearance. | ✗ | most of it |
| **DeepSORT** (Wojke et al., ICIP 2017) | 2017 | SORT + CNN appearance embedding for ReID. Lost tracks live K frames in a pool; can be re-associated by cosine. | ✓ | template bank in single-target form |
| **ByteTrack** (Zhang et al., ECCV 2022) | 2022 | Two-stage association: high-conf detections first, low-conf detections used only to keep unmatched tracks alive through occlusion. No appearance needed. | ✗ | our hybrid pipeline already does this for one target |
| **BoT-SORT** (Aharon et al., 2022) | 2022 | ByteTrack + camera-motion compensation + appearance ReID. Today's de-facto default for surveillance MOT. | ✓ | we already have camera-motion warp + HOG ReID |
| **StrongSORT** (Du et al., TMM 2023) | 2023 | DeepSORT polished: better detector, GSI gap interpolation, ECC camera motion, AFLink trajectory association. | ✓ | partial |
| **OC-SORT** (Cao et al., CVPR 2023) | 2023 | Observation-centric updates: when a track recovers from occlusion, re-fit its motion model from the observation rather than the stale prediction. | ✗ | not yet |

### 1.3  What this means for our build

**BoT-SORT is the right reference design** for our case because we
*already* have all of its parts in the single-target form:

| BoT-SORT ingredient | We already have (single-target) |
|---|---|
| Constant-velocity Kalman per track | `tracker._make_kalman` |
| Camera-motion compensation (ECC homography) | `motion.estimate_homography` + `tracker._warp_state_by_homography` |
| ByteTrack two-stage association | `tracker.update` priority pool |
| Appearance embedding (CNN / HOG) | `reid.HOGReIDExtractor` |
| Template bank of recent embeddings | `tracker._push_to_bank` / `_bank_similarity` |
| Modality reset | `modality.ModalityMonitor` |

What's missing is the **multi-track plumbing**:

1. Hold a *list* of tracks instead of one.
2. **Data association** — per frame, decide which detection belongs
   to which track via Hungarian (or greedy) on a cost matrix.
3. **Birth** — unmatched detections become new tracks with new IDs.
4. **Death** — tracks that don't get matched for K frames move from
   `active` to `lost`.
5. **Re-ID** — unmatched detections are checked against the `lost`
   pool by appearance before being given a new ID.
6. **Expiry** — `lost` tracks that haven't been recovered after M
   frames are dropped.

---

## 2. Proposed design

### 2.1  Architecture

```
       YOLO (thermal) ──► N detections
                              │
[modality monitor] ──► reset all tracks on switch
                              │
[ego-motion H] ──► warp Kalman state of every active track
                              │
                              ▼
            ┌─────────────────────────────────────────┐
            │ IdentityTracker.update(detections, H,   │
            │                       frame_bgr)        │
            │                                         │
            │   Stage 1: ByteTrack high-conf          │
            │     - cost = motion (IoU) only          │
            │     - greedy on (active tracks ×        │
            │       high-conf detections)             │
            │                                         │
            │   Stage 2: appearance for the rest      │
            │     - cost = (1 - cosine HOG)           │
            │     - greedy on (unmatched active ×     │
            │       remaining detections)             │
            │                                         │
            │   Stage 3: re-ID from lost pool         │
            │     - cost = (1 - cosine HOG) only      │
            │     - tighter threshold (~0.65)         │
            │                                         │
            │   Stage 4: spawn new tracks for         │
            │   anything still unmatched              │
            └────────────────┬────────────────────────┘
                             ▼
                   list[IdentityTrack]
                   (each with persistent .id)
```

### 2.2  Per-track state

```python
@dataclass
class IdentityTrack:
    id: int                         # persistent across frames
    bbox: tuple[int, int, int, int]
    state: TrackState               # CONFIRMED / COASTING / LOST
    coast_frames: int
    age: int                        # frames since birth
    hits: int                       # frames matched to a detection
    appearance: np.ndarray | None   # latest HOG embedding (L2-normed)
    bank: list[np.ndarray]          # last K embeddings, newest-first
    kf: cv2.KalmanFilter
    last_seen_frame: int
    last_conf: float
```

### 2.3  Per-frame algorithm (in plain English)

1. Run the modality monitor; on switch, drop all tracks (start over).
2. Compute the ego-motion homography between previous and current frame.
3. For each **active** track, apply the camera-motion warp to its
   Kalman state, then `kf.predict()`.
4. **Stage 1 — motion (IoU) matching on high-conf detections.**
   Cost matrix = `1 − IoU(det, predicted_bbox)`. Greedy assignment
   with a gate IoU ≥ 0.30. Update matched tracks.
5. **Stage 2 — appearance matching on remaining tracks/detections.**
   For each unmatched active track and each remaining detection,
   compute `1 − cosine(HOG(det_crop), bank_best)`. Greedy with a
   gate `cosine ≥ 0.40`. Update matched.
6. **Stage 3 — re-ID from the lost pool.** For each still-unmatched
   detection, compare against every lost track's bank. If a
   `cosine ≥ 0.65` match exists, **re-activate** that track with
   its original ID. (Tighter threshold because we have no spatial
   prior at re-ID.)
7. **Stage 4 — spawn new tracks** for any detection that still
   didn't match anything. Assign the next available ID.
8. **Coast unmatched active tracks.** If a track's coast counter
   exceeds `max_coast_frames`, move it from `active` → `lost`.
9. **Expire** lost tracks older than `max_lost_age` frames.

### 2.4  Implementation choices

- **Greedy assignment** over `scipy`'s `linear_sum_assignment`. With
  ≤ 30 candidates and ≤ 30 tracks the optimality gap is tiny and
  we don't pull in a new dependency.
- **HOG ReID** (already there). The template bank carries up to 8
  recent embeddings per track; matching uses best-of-K (DeepSORT style).
- **Tentative state** is optional. For the assignment clip we'll
  just confirm-on-first-hit because the detector is fairly
  conservative already; can add a `hits >= 3` gate later if
  spurious YOLO false positives become an issue.
- **Visualisation.** A deterministic hash maps each ID to a colour
  (HSV → BGR). The ID number is drawn above the box.
- **What this is and isn't.** This is the multi-object version of
  what we already have for single-object. It's "BoT-SORT with HOG
  ReID and our existing camera-motion compensation." Not a research
  contribution — a textbook pipeline applied with care.

### 2.5  What this fixes / what it leaves on the table

**Fixes**
- The user sees a stable integer ID per person across the clip,
  including across the colormap segment if YOLO recovers.
- The single-target glue-onto-wrong-thing failure mode goes away by
  design — if YOLO finds the person, they keep their ID; if YOLO
  finds something else, that "something else" gets its own ID and
  doesn't contaminate the person's ID.
- "When the person hides" works correctly: their track first goes
  COASTING (Kalman predicts for a few frames), then LOST (sits in
  the lost pool with their appearance bank), then **re-acquires the
  same ID** when YOLO sees them again, as long as appearance is
  close enough.

**Leaves on the table**
- ID switches between two visually-similar people (e.g. two soldiers
  in matching uniforms) are not solved by HOG alone — would need
  a stronger CNN embedding (`reid.CNNReIDExtractor`, blocked by
  the sandbox download wall).
- Trajectories aren't smoothed across long gaps; if you want the
  rendered bbox to interpolate smoothly across a 60-frame coast,
  add StrongSORT-style Gaussian-smoothed interpolation in a post
  pass.

### 2.6  CLI

```bash
python scripts/run_on_video.py --pipeline ids \
    --video assets/How_to_hide_from_a_thermal_drone_Ukraine.mp4 \
    --out outputs/ids.mp4 --side-by-side \
    --dl-weights weights/yolov8_thermal.pt --dl-conf 0.10
```

Same flags as the other DL pipelines; new `--pipeline ids`.

---

## 3. References

- Bewley, A., et al. *Simple Online and Realtime Tracking (SORT).*
  ICIP 2016.
- Wojke, N., et al. *Simple Online and Realtime Tracking with a Deep
  Association Metric (DeepSORT).* ICIP 2017.
- Zhang, Y., et al. *ByteTrack: Multi-Object Tracking by Associating
  Every Detection Box.* ECCV 2022.
- Aharon, N., et al. *BoT-SORT: Robust Associations Multi-Pedestrian
  Tracking.* 2022.
- Du, Y., et al. *StrongSORT: Make DeepSORT Great Again.* IEEE TMM 2023.
- Cao, J., et al. *Observation-Centric SORT (OC-SORT).* CVPR 2023.
