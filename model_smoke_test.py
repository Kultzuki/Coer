"""ST-GCN++ model smoke test: python model_smoke_test.py [--checkpoint PATH] [--device auto|cpu|cuda].

Performs a REAL forward pass through the vendored PYSKL-compatible
ST-GCN++ backbone + VIGLANCE 2-class head when torch is available, loading
official NTU60 backbone weights when the checkpoint is present. Reports
environment, load result, output shapes, timings, and batch benchmarks.
The output reports whether the loaded checkpoint contains a trained fight
head; representation-only checkpoints remain explicitly marked untrained.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def env_report():
    import platform
    print(f"python: {platform.python_version()}")
    try:
        import numpy
        print(f"numpy: {numpy.__version__}")
    except ImportError:
        print("numpy: MISSING")
    try:
        import torch
        print(f"torch: {torch.__version__}")
        cuda = torch.cuda.is_available()
        print(f"cuda_available: {cuda}")
        if cuda:
            print(f"gpu: {torch.cuda.get_device_name(0)}")
            print(f"cuda_version: {torch.version.cuda}")
    except ImportError:
        print("torch: MISSING - install torch (CPU builds suffice)")
    for mod in ("mmcv", "mmengine", "pyskl", "cv2", "ultralytics"):
        try:
            m = __import__(mod)
            print(f"{mod}: {getattr(m, '__version__', 'present')}")
        except ImportError:
            print(f"{mod}: not installed (not required)")


def main(argv=None):
    ap = argparse.ArgumentParser(description="VIGLANCE ST-GCN++ model smoke test")
    ap.add_argument("--checkpoint", default="models/stgcnpp/best_fight_stgcnpp_v1.pth")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--batches", default="1,2,4,8")
    args = ap.parse_args(argv)

    print("== environment ==")
    env_report()

    try:
        import torch  # noqa
    except ImportError:
        print("ABORT: torch unavailable; see report above.")
        return 2
    import numpy as np
    from core.stgcn_adapter import STGCNPPAdapter

    ad = STGCNPPAdapter(device=args.device)
    dev_requested = ad.resolve_device()
    print(f"== device: {dev_requested} ==")

    t0 = time.perf_counter()
    try:
        info = ad.load_model(args.checkpoint, device=args.device)
        print(f"checkpoint: {args.checkpoint} LOADED "
              f"(missing={len(info.missing_keys)}, "
              f"unexpected={len(info.unexpected_keys)}, "
              f"ignored_head={info.ignored_head_keys})")
    except FileNotFoundError as exc:
        print(f"checkpoint: MISSING - {exc}")
        print("continuing with randomly initialized backbone (forward still real)")
    load_ms = (time.perf_counter() - t0) * 1000.0
    print(f"model load time: {load_ms:.1f} ms")

    torch.manual_seed(0)
    np.random.seed(0)
    print("== forward / batch benchmark (CPU" +
          (", CUDA" if dev_requested == "cuda" else "") + ") ==")
    for n in [int(x) for x in args.batches.split(",")]:
        x = np.random.rand(n, 2, ad.temporal_length, 17, 3).astype(np.float32)
        t0 = time.perf_counter()
        preds = ad.predict_detailed([None] * n, x)
        dt = (time.perf_counter() - t0) * 1000.0
        assert all(len(p.logits) == 2 for p in preds)
        p0 = preds[0]
        print(f"batch={n} out=({n},2) total={dt:.1f}ms per-pair={dt / n:.1f}ms "
              f"logits={[round(v, 4) for v in p0.logits]} "
              f"fight_probability={p0.fight_probability:.4f} [{p0.head_status}]")

    status = "TRAINED_HEAD" if ad.head_trained else "UNTRAINED_HEAD"
    print(f"SMOKE OK: real ST-GCN++ forward executed; head_status={status}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
