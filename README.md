# VIGLANCE — Clean Foundation (Raksha Netra hackathon)

This directory is the **reproducible foundation** for the fight-architecture
redesign. It ports the `fight-main` system with foundational defects fixed.
The new fight architecture is **not** implemented here, by design.

## Architecture map (audited from `fight-main`, code-read, not README)

| Concern | File (here) | Legacy source | Finding |
|---|---|---|---|
| Entrypoint / viz | `main.py` | `fight-main/main.py` | Thin CLI now; business logic moved out of `main` |
| Detector | `core/detector.py` | `core/detector.py` | Default was `yolo26x-pose.pt` (missing from repo); fixed to `yolo26m-pose.pt`. Tracker/imgsz/conf/device/max_people configurable |
| Tracker | ultralytics BoT-SORT via `Detector(tracker=...)` | hard-coded `"botsort.yaml"` | Now `config.tracker` (default `botsort.yaml`) |
| Fight logic (legacy) | `scenarios/fight_logic.py` | `scenarios/fight_logic.py` (BiLSTM-128x2 + attention, 30-frame, 102-D) | Module preserved but OFF by default (`enable_legacy_fight`); ST-GCN++ pair/graph is the fight path |
| Fall logic | `scenarios/fall_logic.py` | `scenarios/fall_logic.py` (torso angle + aspect + stillness) | Timestamp explicit; `reset()` added |
| Bag logic | `scenarios/bag_logic.py` | `scenarios/bag_logic.py` (person–bag distance, COCO class 26) | Timestamp explicit; `reset()` added |
| Utilities | `utils/keypoint_utils.py` | `utils/keypoint_utils.py` | Dead 4-person assumption generalized (`flatten_persons`); `flatten_four_person` kept as compat wrapper |
| Canonical person | `core/tracked_person.py` | ad-hoc dicts in `pipeline.py` | NEW: `TrackedPerson(track_id, bbox, keypoints(17,3), detection_conf, timestamp, present)` |
| Orchestration | `core/pipeline.py` | `core/pipeline.py` | `print(track_ids)` removed; broad `except: print` → logged traceback + `strict` flag; pairing via single atomic loop |
| Config | `config.py` + `config/default.json` | none (hard-coded; `Pipeline()` took no args, `main.py` only parsed `--source`) | NEW |
| Replay | `core/replay.py` | none | NEW: `frame_index/fps` timestamps, seeded, JSONL evidence, `compare_runs` |
| Logging | `core/logging_utils.py` | `print()` everywhere | NEW: `vigilance.*` loggers |
| Tests | `tests/test_smoke.py` | none | NEW: 11 smoke tests, numpy+pytest only |
| Checkpoints | `models/` (git-ignored, resolved at runtime) | `yolo26m-pose.pt`, `yolo26m.pt`, `models/fight_classifier.pt` vendored | Weights NOT vendored here; see `models/README.md` |
| Requirements | `requirements.txt` | README snippet (`opencv-python ultralytics torch numpy`, no versions) | Version-floored; tests need only numpy+pytest |
| Training/eval | — | **none** (no dataset code, no harness; `fight_classifier.pt` has no provenance) | Still missing — blocker for the redesign phase, out of scope here |

## Quickstart

```bash
pip install -r requirements.txt          # full ML stack
# smoke tests need only: pip install pytest numpy
pytest -q
python main.py --config config/default.json --source 0
python main.py --config config/default.json --source ./video.mp4
python main.py --config config/default.json --no-bag
```

Weights are not vendored in this archive (they are ~127 MB):
`yolo26m-pose.pt`, `yolo26m.pt`, and `models/stgcnpp/best_fight_stgcnpp_v1.pth`
must be placed at the paths in `models/README.md`. Tests that need them skip
themselves; the repo runs without them.

Config precedence: file → `VIGILANCE_*` env vars → CLI flags.
Weights resolution: explicit flag → env var → repo root → `models/` → cwd.

## Multi-person perception (this phase)

Detections flow through `core/track_registry.py` before any scenario sees
them. No truncation: `max_people` (default 32) is a safety cap only.

- `TrackedPerson` carries `track_id, bbox, keypoints(17,3),
  keypoint_conf(17), detection_conf, timestamp, first_seen, last_seen,
  age_frames (+`track_age_sec`), `generation`, `state`, `present`.
- `TrackRegistry`: `ACTIVE -> LOST -> (REAPPEARED | REMOVED)` with explicit
  `TrackEvent`s. Query API: `all_people / active_people / recently_lost /
  get(id)` (also via `Pipeline.active_people()` etc.).
- `PerTrackStore`: per-identity scenario state keyed by
  `(track_id, generation)` — a returning ID gets a fresh key, so event
  history is never silently inherited. Residual risk: if the tracker itself
  recycles a numeric ID after removal, it enters as NEW; true
  re-identification across recycling needs appearance embeddings (out of
  scope, flagged for the redesign).
- Tracker: BoT-SORT primary, ByteTrack fallback (`config.tracker` /
  `tracker_fallback`; one-shot failover, logged). Lifecycle timeouts:
  `lost_timeout_sec` (default 1.0), `lost_grace_sec` (default 5.0).
- Viz: `core/visualization.py` draws bbox + track ID (color-coded) + COCO
  skeleton; `main.py` uses it, logs a diagnostics summary at stream end.
- Diagnostics: `TrackerDiagnostics.summary()` → frames, avg/max
  people-per-frame, empty-frame dropout rate, unique IDs, churn
  (new/lost/reappeared/removed).
- No fight / fall / bag classification was added or changed in this phase;
  existing scenario modules are untouched and keep working on active persons.

## Pairwise violence inference + interaction-graph reasoning (this phase)

VIGLANCE performs pairwise violence inference and derives multi-person
assault candidates through interaction-graph reasoning. There is no
"mob lynching" classifier: ST-GCN++ scores individual person pairs, and a
deterministic graph layer aggregates violent edges into scene candidates
(`NO_FIGHT / FIGHT / MULTI_PERSON_FIGHT / GROUP_ASSAULT`).

- Pairs: `core/pairs.py` — every ACTIVE person eligible (no first-two
  truncation); spatial filter on body-scale-normalized distance
  (`fight.pair_max_distance`), visibility, top-K cap. Identity is the
  canonical `(id_a, gen_a, id_b, gen_b)` key; each `PairTrack` owns a
  timestamped buffer with a LOST grace period, then closes and clears.
- Skeleton: `core/skeleton.py` — one `COCO_KEYPOINT_NAMES` definition
  (identity-mapped from YOLO order, documented); per-person root/scale
  normalization (hip→shoulder→bbox fallback; never NaN/Inf);
  `CanonicalSkeleton`; joint/bone/joint-motion/bone-motion preprocessing.
- Adapter: `core/stgcn_adapter.py` — `preprocess_pair` → `(2, T, 17, 3)`
  tensor, batch `(N, 2, T, 17, C)`; timestamp resampling to
  `fight.temporal_length` over `fight.window_seconds`; `predict` refuses
  without a loaded model (a pretrained ST-GCN++ checkpoint is not a
  fight classifier; head fine-tuning is a later stage).
- PYSKL ground truth (verified against cloned repos + parsed checkpoints):
  2D configs are NTU60-HRNet with `layout='coco'` (V=17); bone topology and
  `JointToBone` pairs match `pyskl/utils/graph.py` exactly; model input uses
  frame-size normalization (`PreNormalize2D`), which the adapter applies —
  body-scale coordinates stay internal only. Bone/motion streams follow
  `JointToBone`/`ToMotion` conventions (score averaging, last-frame-zero).
  PYSKL releases bare state dicts, MMAction2 wraps `{meta, state_dict}`;
  only the 2 head keys differ (`fc_cls` vs `fc`) — handled by
  `normalize_state_dict`. No Kinetics-Skeleton ST-GCN++ exists in either
  framework, so fine-tuning starts from NTU60 weights.
- Graph: `core/graph.py` — nodes are people, edges are candidate pairs +
  violence probabilities; hub-with-quiet-spokes → `GROUP_ASSAULT`
  candidate, connected violent component (3+) → `MULTI_PERSON_FIGHT`,
  single/disjoint violent edges → `FIGHT`; all thresholds under
  `fight.graph` in config. Neutral "violent interaction" terminology;
  no attacker/victim claims.
- Pipeline wires it per frame (skeletons computed once, batched pair
  inference, insufficient buffers skipped); legacy fall/bag modules
  untouched, and the legacy BiLSTM fight module is OFF by default
  (`enable_legacy_fight`, opt in with `--legacy-fight` for A/B comparison
  only) — ST-GCN++ pair/graph is the only active fight path. Debug viz:
  pair lines, scores, components, scene panel.
- CLI/config precedence: unset flags never override the config file
  (`--source`/`--log-level` default to `None`; explicit flags and
  `VIGILANCE_*` env vars still win). CLI flag order: explicit flag >
  env var > config file > built-in default.
- Replay: `python replay.py --source video.mp4 [--viz]`,
  `python replay.py --synthetic [--demo-scores "1-2:0.94,1-3:0.91"]`
  (no auto-download; cv2 only for real video/viz).

## ST-GCN++ Fight Model Integration

Current status: **fight-fine-tuned ST-GCN++ checkpoint available**. The
repository also retains the upstream representation-only checkpoint; scores
from that checkpoint remain labeled `UNTRAINED_HEAD` and are not fight
predictions.

- Fight checkpoint: `models/stgcnpp/best_fight_stgcnpp_v1.pth` (Kaggle
  fine-tuned head and backbone). The representation-only checkpoint is
  `models/stgcnpp/j.pth`. Override via `fight.model.checkpoint` /
  `VIGILANCE_FIGHT_MODEL_CHECKPOINT`. The repo stays usable with weights
  absent (explicit `ModelNotLoadedError`, never fake scores).
- Model architecture: vendored PYSKL-compatible ST-GCN++
  (`core/stgcn_model.py`; upstream: PYSKL @ `f2bf3a6`, `configs/stgcn++`,
  `models/gcns/{stgcn,utils/{gcn,tcn}}`, `models/heads/simple_head.py`,
  `utils/graph.py`; MMAction2 `configs/skeleton/stgcnpp` secondary ref):
  STGCN backbone (`gcn_adaptive='init'`, residual, MSTCN, COCO graph,
  in=3, 10 stages, 256 output channels) + `Linear(256, 2)` VIGLANCE head.
- Input shape: `(N, M=2, T=100, V=17, C=3)` with C=(x_norm, y_norm, score);
  COCO-17 mapping identity from YOLO order; `T` from
  `fight.temporal_length` over `fight.window_seconds`; frame-size
  normalization per PYSKL `PreNormalize2D` (score ≤ 0.01 zeroes x/y).
- Output classes: `0=NO_FIGHT`, `1=FIGHT`; logits → softmax. The imported
  fight checkpoint restores both `fight_head.*` tensors and reports
  `TRAINED_HEAD`; the upstream checkpoint remains representation-only.
- Device: `fight.model.device` (`auto` → CUDA iff available); CPU works,
  lazy load, never at import time. Autoload in `Pipeline` only when
  `fight.model.autoload=true` and the file exists.
- Provenance/license: PYSKL is Apache-2.0; checkpoint redistribution terms
  were NOT verified in this prompt — check before any production or
  commercial use.
- Model smoke test (real forward pass when torch + checkpoint present):

```bash
python model_smoke_test.py [--device auto]
python model_smoke_test.py --checkpoint models/stgcnpp/j.pth --device auto
```

## Measuring fight accuracy (deployed, end-to-end)

`eval_fight.py` runs labeled clips through the real pipeline (YOLO + BoT-SORT
+ PairTracks + trained ST-GCN++ head) and reports accuracy, precision,
recall/F1, the confusion matrix, and an F1-optimal threshold. Yes — this is
exactly the "feed it fight and non-fight videos" check; it is the number that
matters because it includes detector and tracker failures, not just the
model on clean tensors.

```bash
# one command: builds the manifest from two labeled folders, then evaluates
python eval_fight.py --fight-dir Fight-videos --no-fight-dir No-FIGHT

# explicit manifest instead (CSV: video_path,label ; 1=fight, 0=no-fight)
python eval_fight.py --manifest evidence/val_manifest.csv --threshold 0.6
```

Clip score = max pair fight probability over the clip; prediction = score >=
threshold (default `fight.graph.violence_edge_threshold` = 0.6). Per-clip
evidence (score, scored pair-frames, unique track IDs, average people, max
continuous pair span) lands in `evidence/eval_fight.csv`.

Measured on the bundled 10 clips (5 fight / 5 no-fight, 5 s each, 30 fps,
320x240 to 1280x720):

| threshold | accuracy | precision | recall | scored clips |
|---|---|---|---|---|
| 0.6 (default) | 0.70 | 1.00 | 0.40 | 4/10 |

- Precision 1.00 = zero false alarms: every scored no-fight clip peaked at
  0.15-0.49, well under 0.6.
- The misses are pipeline coverage, not model error: a pair is only scored
  after a continuous history of `fight.window_seconds` (4.0 s deployed
  default). On 5 s clips with BoT-SORT ID churn, pair buffers never reach
  that span, so 6/10 clips produced zero scored pair-frames (clips with 0 or
  1 visible people cannot form a pair at all).
- Where evidence IS collected the model separates cleanly: scored fight clips
  peaked 0.90-0.97, scored no-fight clips 0.15-0.49.

Practical guidance: use 10-20 s clips for a meaningful recall number, or
`--window-seconds 1.0` to trade temporal context for coverage on short clips
(fight peaks then drop to 0.50-0.93 and no-fight stays <= 0.37). The
checkpoint itself reports Kaggle tensor-level best val loss 0.350 over 25
epochs (see `models/stgcnpp/training_summary.json`), which is the ceiling for
the deployed number above.

## Deterministic replay

```python
from config import VigilanceConfig
from core.replay import ReplayRunner
records = ReplayRunner(VigilanceConfig(fps=30.0)).run_frames(frames, pipeline)
```

Same frames + same config + same seed → comparable record streams
(verified by `test_deterministic_replay_comparable`).

## Web UI (no web dependencies)

```bash
.venv/Scripts/python.exe ui.py --source 0                     # webcam + real YOLO
.venv/Scripts/python.exe ui.py --source video.mp4             # video file
.venv/Scripts/python.exe ui.py --synthetic                    # scripted demo, no camera/weights
.venv/Scripts/python.exe ui.py --config config/default.json
.venv/Scripts/python.exe ui.py --fight-checkpoint models/stgcnpp/best_fight_stgcnpp_v1.pth
```

Open http://localhost:8000 — live MJPEG (`/stream.mjpg`), status JSON
(`/api/status`: people, pairs + scores, scene verdict, model state, threat
detections, alerts, diagnostics), single frame (`/snapshot.jpg`). The
dashboard has a webcam button and a browser upload control for MP4, AVI, MOV,
MKV, MPEG, WebM, and other formats supported by the local OpenCV build.
It also has an incident timeline and browser fight notifications. Click
`Enable fight notifications` once to grant browser notification permission.
Served from stdlib `http.server`, so it works everywhere including Kaggle.

The real UI downloads this auxiliary YOLO model from Hugging Face on first
use unless `--no-threat-download` is supplied:

- `xDevine/yolo11s-multiclass-surveillance-detection` for weapons, fire,
  smoke, and bag candidates (21 classes incl. person/car context).

`Subh775/Threat-Detection-YOLOv8n` is **disabled**: it is never resolved or
downloaded, and its cross-model "agreement" rule is gone. The single
surveillance model covers every category VIGLANCE reasons about.

Weapon alerts are disabled by default for the hackathon demo because weapon
scores were producing unwanted warnings. The auxiliary detections remain
visible. Enable the alert path explicitly with `--enable-weapon-alerts`; it
requires a high-confidence detection from the surveillance model persisting
across several frames. Use `--surveillance-weights` to provide a local file
instead.

The on-frame banner and `/api/status` model block always show the fight-model
state honestly: `ABSENT` (graph structure only), `UNTRAINED_HEAD` (backbone
loaded, scores not meaningful), or `TRAINED`. Post-training integration is
one flag: `--fight-checkpoint <.pth>` (same flag exists on `main.py`);
`fight.model.autoload`/`checkpoint` in config or `VIGILANCE_FIGHT_CHECKPOINT`
do the same without flags.

Env note: torch and torchvision builds must match (`+cu130` with `+cu130`).
A CPU torchvision beside CUDA torch breaks YOLO NMS on GPU with
`torchvision::nms ... CUDA backend` — reinstall torchvision from the same
CUDA index as torch.

## What changed vs legacy (foundational only)

1. Broken pose default fixed; all model paths configurable.
2. `max_people` replaces the hard-coded 2 (comment said 4, code did 2).
3. Track/box/keypoint pairing is one atomic index-aligned loop.
4. No import-time model init in any scenario module.
5. Timestamps are explicit arguments end-to-end; `time.time()` only as a
   logged backward-compat fallback inside `check()`.
6. Structured logging; per-frame prints deleted.
7. Scenario errors are logged with tracebacks (isolated, loud); `strict`
   mode re-raises.
8. `reset()` on pipeline + every scenario enables replay.
