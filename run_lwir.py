"""
End-to-end LWIR training and test-set evaluation for all architectures.

Trains each model for 30 epochs on data/lwir/, then evaluates best.pt on
the test split. Results are saved to runs/lwir/results.csv.

Usage:
    python run_lwir.py                 # MPS (Apple Silicon)
    python run_lwir.py --device 0      # CUDA GPU 0
    python run_lwir.py --device cpu
"""

import argparse
import gc
import sys
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd
import torch
from ultralytics import YOLO

from evaluate import evaluate_model


def _log(msg: str):
    print(f"[run_lwir] {msg}", flush=True)


# ── config ────────────────────────────────────────────────────────────────────

ARCHITECTURES = [
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8n-p2",
    "yolo11n.pt",
    "yolo11s.pt",
    "yolov12n.pt",
    "yolo26n.pt",
    "yolo26s.pt",
]

DATA    = "data/lwir/data.yaml"
EPOCHS  = 30
IMGSZ   = 1280       # native LWIR frames are 1440×1080; 1280 = 40×32, no upscaling
BATCH   = 4          # reduced from 8 — larger imgsz needs more VRAM
WORKERS = 2
PROJECT = "runs/lwir"


# ── training ──────────────────────────────────────────────────────────────────

def _train(arch: str, device: str) -> tuple[str, Path, dict]:
    """Train one architecture. Returns (clean_name, weights_path, val_metrics_dict)."""
    _log(f"clearing GPU/MPS cache before loading {arch}")
    torch.cuda.empty_cache()
    gc.collect()

    if "-p2" in arch:
        base  = arch.replace("-p2", "")
        _log(f"loading P2 variant: {arch}.yaml + {base}.pt")
        model = YOLO(f"{arch}.yaml").load(f"{base}.pt")
        clean = arch
    else:
        _log(f"loading pretrained weights: {arch}")
        model = YOLO(arch)
        clean = arch.replace(".pt", "")

    batch = BATCH // 2 if "s.pt" in arch else BATCH
    amp   = "cuda" in str(device)   # AMP stable on CUDA; skip on MPS/CPU
    run_name = f"{clean}_{IMGSZ}px"

    _log(f"starting training — epochs={EPOCHS}  imgsz={IMGSZ}  batch={batch}  amp={amp}  device={device}")
    _log(f"output dir: {PROJECT}/{run_name}")

    t0 = time.time()
    results = model.train(
        data=DATA,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=batch,
        project=PROJECT,
        name=run_name,
        workers=WORKERS,
        amp=amp,
        cache=False,
        plots=True,
        device=device,
        exist_ok=True,   # deterministic path on re-runs
    )
    elapsed = timedelta(seconds=int(time.time() - t0))
    _log(f"training done in {elapsed}")

    weights = Path(PROJECT) / run_name / "weights" / "best.pt"
    _log(f"best weights: {weights}  (exists={weights.exists()})")

    val_metrics = {
        "val_mAP@50":    round(results.box.map50, 4),
        "val_mAP@50-95": round(results.box.map,   4),
        "val_infer_ms":  round(results.speed["inference"], 2),
    }
    _log(f"val mAP@50={val_metrics['val_mAP@50']}  mAP@50-95={val_metrics['val_mAP@50-95']}")
    return clean, weights, val_metrics


# ── evaluation ────────────────────────────────────────────────────────────────

def _evaluate(weights: Path, device: str) -> dict:
    """Evaluate best.pt on the LWIR test split."""
    _log(f"evaluating on test split: {weights}")
    t0 = time.time()
    row = evaluate_model(
        entry={"modality": "lwir", "weights": str(weights), "data": DATA},
        cfg={
            "split":   "test",
            "imgsz":   IMGSZ,
            "batch":   BATCH,
            "workers": WORKERS,
            "plots":   True,
            "device":  device,
        },
    )
    elapsed = timedelta(seconds=int(time.time() - t0))
    _log(f"evaluation done in {elapsed}  —  mAP@50={row['mAP@50']}  mAP@50-95={row['mAP@50-95']}")
    return {
        "test_precision":  row["precision"],
        "test_recall":     row["recall"],
        "test_mAP@50":     row["mAP@50"],
        "test_mAP@50-95":  row["mAP@50-95"],
        "test_infer_ms":   row["inference_ms"],
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main(device: str):
    Path(PROJECT).mkdir(parents=True, exist_ok=True)

    n = len(ARCHITECTURES)
    _log(f"starting LWIR sweep — {n} architectures  epochs={EPOCHS}  imgsz={IMGSZ}  device={device}")
    _log(f"architectures: {ARCHITECTURES}")

    rows = []
    sweep_start = time.time()

    for i, arch in enumerate(ARCHITECTURES, 1):
        print(f"\n{'='*60}")
        print(f"  [{i}/{n}]  LWIR | {arch}")
        print(f"{'='*60}\n")

        try:
            clean, weights, val_m = _train(arch, device)
        except Exception as exc:
            _log(f"ERROR training {arch}: {exc}")
            print(f"ERROR training {arch}: {exc}", file=sys.stderr)
            continue

        row = {"architecture": clean, "weights": str(weights), **val_m}

        if weights.exists():
            try:
                row.update(_evaluate(weights, device))
            except Exception as exc:
                _log(f"ERROR evaluating {arch}: {exc}")
                print(f"ERROR evaluating {arch}: {exc}", file=sys.stderr)
        else:
            _log(f"WARNING: weights not found at {weights}")

        rows.append(row)

        elapsed_total = timedelta(seconds=int(time.time() - sweep_start))
        _log(f"[{i}/{n}] {arch} complete — total elapsed: {elapsed_total}")

    if not rows:
        _log("no results — all architectures failed")
        print("No results.", file=sys.stderr)
        return

    df = pd.DataFrame(rows)
    sort_col = "test_mAP@50-95" if "test_mAP@50-95" in df.columns else "val_mAP@50-95"
    df = df.sort_values(by=sort_col, ascending=False).reset_index(drop=True)

    total_time = timedelta(seconds=int(time.time() - sweep_start))
    _log(f"sweep complete — {len(rows)}/{n} architectures succeeded  total time: {total_time}")

    print("\n=== LWIR Results ===")
    try:
        print(df.to_markdown(index=False))
    except ImportError:
        print(df.to_string(index=False))

    out = Path(PROJECT) / "results.csv"
    df.to_csv(out, index=False)
    _log(f"results saved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full LWIR training and evaluation sweep.")
    parser.add_argument(
        "--device", default="mps",
        help="Compute device: mps, cpu, 0 (CUDA GPU index). Default: mps",
    )
    args = parser.parse_args()
    main(args.device)
