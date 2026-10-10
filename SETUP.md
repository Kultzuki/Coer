# VIGLANCE — setup & run guide

This is a local AI surveillance prototype: it watches a webcam or video file,
tracks people (YOLO26m-pose + BoT-SORT), scores every pair of people with a
trained ST-GCN++ fight classifier, and aggregates pairs into a scene verdict
(no fight / fight / group fight / group assault). It also detects weapons,
fire, smoke and unattended bags with one auxiliary YOLO model. Everything runs
on this computer — no cloud, no API keys.

Time needed: ~15 minutes setup (most of it is the Python install).

---

## 1. What you need

| Thing | Notes |
|---|---|
| Windows 10/11 (macOS/Linux also works) | guide shows Windows commands |
| Python 3.10–3.12 | check with `python --version`; **3.11 recommended** |
| ~7 GB free disk | torch + CUDA libraries are the bulk |
| NVIDIA GPU (optional) | the code auto-uses CUDA when available, else CPU |
| Webcam (optional) | or just use video files |

---

## 2. Get the code

Unzip `Re-Vigilance.zip` somewhere without spaces in the path if you can
(e.g. `C:\Users\<you>\Coding\Re-Vigilance`), then open a terminal in that
folder:

```powershell
cd C:\Users\<you>\Coding\Re-Vigilance
```

---

## 3. Create the virtual environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks the activate script:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

(cmd instead of PowerShell: `.\.venv\Scripts\activate.bat` — Git Bash:
`source .venv/Scripts/activate`.)

You should see `(.venv)` at the start of your prompt. From now on every
command assumes that. (`pip install pytest numpy` alone is enough if you only
want to run the test suite.)

**GPU note:** a plain `pip install torch` already brings a CUDA build. If you
ever see errors like `torchvision::nms ... CUDA backend`, your torch and
torchvision were built for different backends — reinstall torchvision from the
same source as torch. Everything also works CPU-only, just slower.

---

## 4. Put the model weights in place

The weights are **not** in the zip (128 MB). You need three files:

```
Re-Vigilance\
  yolo26m-pose.pt                        <- person pose detector
  yolo26m.pt                             <- object detector
  models\stgcnpp\best_fight_stgcnpp_v1.pth   <- the FIGHT classifier
```

Easiest: copy that `models\stgcnpp\best_fight_stgcnpp_v1.pth` from whoever gave
you this repo — that one is a fine-tuned model and cannot be re-downloaded.

The two YOLO weights can be fetched automatically (needs internet, ~90 MB):

```powershell
python -c "from ultralytics import YOLO; YOLO('yolo26m-pose.pt'); YOLO('yolo26m.pt')"
```

That drops both files in the current folder, which is exactly where they need
to be.

**No weights?** The app still starts, but it is honest about it: the dashboard
banner says `Fight detection: not installed - movement only`, and 14 tests skip
themselves instead of faking results.

---

## 5. Verify the install

```powershell
pytest -q
```

Expected: `112 passed` (or `98 passed, 14 skipped` if you skipped the fight
checkpoint). Then check the fight model does a real forward pass:

```powershell
python model_smoke_test.py
```

---

## 6. Run it

### A. Demo with no camera and no weights (30 seconds)

```powershell
python ui.py --synthetic
```

Open <http://localhost:8000> — scripted people, pipeline and dashboard fully
live, nothing downloaded.

### B. Web dashboard on your webcam

```powershell
python ui.py --source 0 --fight-checkpoint models\stgcnpp\best_fight_stgcnpp_v1.pth
```

Open <http://localhost:8000>. You get a live MJPEG stream, people/pair scores,
the scene verdict, an incident timeline, and browser fight notifications
(click **Turn on fight alerts on this device** once to allow them). Upload a
video file with the **Play a video file** button, or switch back to the camera.

### C. Dashboard on a video file

```powershell
python ui.py --source "C:\path\to\clip.mp4"
```

### D. Plain OpenCV window, no browser

```powershell
python main.py --config config\default.json --source "C:\path\to\clip.mp4"
```

(`--source 0` for the webcam, or point it at an image path to test one frame.
Press `q` to quit.)

### E. Measure accuracy on labeled clips

Put fight clips in one folder and non-fight clips in another, then:

```powershell
python eval_fight.py --fight-dir Fight-videos --no-fight-dir No-FIGHT
```

It builds a manifest, pushes every clip through the real detector + tracker +
fight model, and prints accuracy / precision / recall / F1, the confusion
matrix, an F1-optimal threshold, and per-clip evidence in
`evidence\eval_fight.csv`. Expect the deployed number to be lower than the
checkpoint's own validation score: short 5-second clips often never accumulate
the 4 seconds of pair history the model needs (the per-clip `max_span` column
shows this). Try 10–20 second clips for a meaningful recall figure, or
`--window-seconds 1.0` to trade context for coverage.

First run with real video optionally downloads the weapons/fire/smoke detector
(`xDevine/yolo11s-multiclass-surveillance-detection`, 18 MB) into your Hugging
Face cache. Skip it with `--no-threats`.

---

## 7. Flags worth knowing

| Flag | What it does |
|---|---|
| `--fight-checkpoint <path>` | load a specific ST-GCN++ fight checkpoint |
| `--no-threats` | disable the weapons/fire/smoke/bag YOLO model |
| `--no-threat-download` | never download that model; local files only |
| `--surveillance-weights <path>` | use your own copy of that model |
| `--enable-weapon-alerts` | weapon alerts (off by default, noisy for demos) |
| `--no-fight` / `--no-fall` / `--no-bag` | switch one scenario off |
| `--legacy-fight` | old BiLSTM fight module, for A/B comparison only |
| `--device cpu` | force CPU (default `auto` = CUDA if present) |
| `--config config\default.json` | load `config/default.json` (enables fight autoload) |
| `--max-people N` | safety cap on tracked people (default 32) |
| `--window-seconds S` | pair history required before scoring (default 4.0) |
| `--threshold T` | decision threshold on pair fight probability (default 0.6) |

Config precedence: CLI flag > `VIGILANCE_*` environment variable > config file >
built-in default. Any field in `config/default.json` can also be set with an
env var, e.g. `$env:VIGILANCE_FIGHT_MODEL_CHECKPOINT="models\stgcnpp\best_fight_stgcnpp_v1.pth"`.

---

## 8. What the dashboard shows

- **Hero banner**: the scene verdict (from the interaction graph over scored
  pairs) — `All clear` while nothing is detected, `Possible fight detected`
  when it is.
- **On the video itself**: only alarm states get a banner (`POSSIBLE FIGHT -
  CHECK NOW`, `GROUP FIGHT - CHECK NOW`, `GROUP ASSAULT - CHECK NOW`). A calm
  scene deliberately draws nothing on the frame — a "everyone looks fine"
  banner would be a false claim, because a quiet pair score can also mean the
  pair history never filled (short clips, tracker ID churn). Track boxes,
  person labels, risk chips and threat boxes stay visible as usual.
- **At a glance**: people in view, alerts right now, incident count, source.
- **Recent activity**: incident timeline, clickable timestamps of fight scenes.
- **Threat chips**: weapon / fire / smoke / bag candidates seen in frame.
- **Technical details** (expand at the bottom): fight-model state, threat-model
  state, per-person tracks, per-pair scores and buffers, tracker diagnostics.
- Model state is always honest: `not installed` (no checkpoint),
  `not trained` (backbone only, scores meaningless), or `active`.

---

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `Detector failed: Model weights 'yolo26m-pose.pt' not found` | weights missing — see step 4 |
| `Fight detection: not installed` banner | checkpoint missing/wrong path — pass `--fight-checkpoint` |
| `torchvision::nms ... CUDA backend` | torch and torchvision mismatch — reinstall from the same index |
| Scores stay 0.000 on short clips | pair history (4 s) never reached — use longer clips or `--window-seconds 1.0` |
| Dashboard says `Video connection lost` | camera busy or unreadable file — close other apps, or try a different `--source` |
| Webcam doesn't open on index `0` | try `--source 1` |
| `ModuleNotFoundError` | virtualenv not activated — `.\.venv\Scripts\Activate.ps1` |
| Weapon alerts never fire | they are opt-in: add `--enable-weapon-alerts` |
| 14 tests skipped | expected without the fight checkpoint; 112 pass with it |

Logs are structured and go to stderr (`event=...` lines); raise verbosity with
`--log-level DEBUG`. Every scenario error is logged with a traceback rather
than silently swallowed.

---

## 10. Layout

```
main.py              CLI: window + overlays, no business logic
ui.py                web dashboard (stdlib http.server, no web deps)
eval_fight.py        accuracy harness on labeled clips
model_smoke_test.py  real forward pass through the fight model
replay.py            deterministic replay / demo scoring
config.py            all settings (dataclasses, env + file + CLI)
config/default.json  the default config file
core/                detector, tracker registry, pairs, skeleton, graph,
                     ST-GCN++ adapter + model, threat detector, visualization
scenarios/           fall, abandoned-bag, legacy BiLSTM fight logic
tests/               pytest suite (numpy + pytest only)
models/              where weights go (not vendored)
evidence/            eval output (CSV), created automatically
```

Kill a web dashboard with `Ctrl+C`.
