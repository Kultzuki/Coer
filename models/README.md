# Model weights are NOT vendored in this repository.

The foundation build resolves weights at runtime (see `config.py`
`resolve_model_path`): explicit `--pose-weights` / `--det-weights` flags,
`VIGILANCE_POSE_WEIGHTS` / `VIGILANCE_DET_WEIGHTS` env vars, repo root,
`models/`, then cwd. Defaults (`yolo26m-pose.pt`, `yolo26m.pt`) match the
files the project actually ships; the legacy default `yolo26x-pose.pt` did
not exist in the repo and is fixed.

Place `yolo26m-pose.pt`, `yolo26m.pt`, and the fight checkpoint
`models/stgcnpp/best_fight_stgcnpp_v1.pth` here or point the config at them.
The legacy `models/fight_classifier.pt` is a separate BiLSTM path and is not
the active ST-GCN++ classifier. `*.pt` is git-ignored by design so the
repo stays small and Kaggle/Linux/Windows setups can each fetch or mount
their own copies.

The web demo can also download one optional object detector on first use:

- `xDevine/yolo11s-multiclass-surveillance-detection` (`weights/best.pt`)

That model supplies the surveillance classes VIGLANCE reasons about: weapons
(short-gun, long-gun, knife), fire, smoke, phone, person, car, and bags.
`Subh775/Threat-Detection-YOLOv8n` is disabled: it is never downloaded, and the
UI requires a high-confidence multi-frame streak from the single surveillance
model before raising a weapon alert.
