"""v2 motion baseline + DL person-class verifier (every K frames).

v2's MotionPipeline runs unchanged as the primary detector / tracker.
A thermal YOLO is invoked every K frames as a *verifier only*:
  - if DL fires and at least one detection overlaps v2's current bbox
    (IoU >= 0.2) -> v2 is "DL-VERIFIED", no change
  - if DL is silent -> "DL-SILENT", trust v2 (its motion path is what
    we built for exactly this case)
  - if DL fires somewhere else with no overlap with v2's bbox ->
    "DL-CONTRADICTED". After N consecutive CONTRADICTED checks we
    reseed v2's tracker from the strongest DL detection.

The DL detector is never used to initialize v2 from scratch -- v2's
own motion-persistence init logic decides identity. DL only intervenes
once v2 already has a track that DL disagrees with.
"""
