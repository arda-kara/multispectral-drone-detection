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
from pathlib import Path

import pandas as pd
import torch
from ultralytics import YOLO

from evaluate import evaluate_model


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
    torch.cuda.empty_cache()
    gc.collect()

    if "-p2" in arch:
        base  = arch.replace("-p2", "")
        model = YOLO(f"{arch}.yaml").load(f"{base}.pt")
        clean = arch
    else:
        model = YOLO(arch)
        clean = arch.replace(".pt", "")

    batch = BATCH // 2 if "s.pt" in arch else BATCH
    amp   = "cuda" in str(device)   # AMP stable on CUDA; skip on MPS/CPU

    results = model.train(
        data=DATA,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=batch,
        project=PROJECT,
        name=f"{clean}_{IMGSZ}px",
        workers=WORKERS,
        amp=amp,
        cache=False,
        plots=True,
        device=device,
        exist_ok=True,   # deterministic path on re-runs
    )

    weights = Path(PROJECT) / f"{clean}_{IMGSZ}px" / "weights" / "best.pt"
    val_metrics = {
        "val_mAP@50":    round(results.box.map50, 4),
        "val_mAP@50-95": round(results.box.map,   4),
        "val_infer_ms":  round(results.speed["inference"], 2),
    }
    return clean, weights, val_metrics


# ── evaluation ────────────────────────────────────────────────────────────────

def _evaluate(weights: Path, device: str) -> dict:
    """Evaluate best.pt on the LWIR test split."""
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
    rows = []

    for arch in ARCHITECTURES:
        print(f"\n{'='*60}")
        print(f"  LWIR | {arch}")
        print(f"{'='*60}\n")

        try:
            clean, weights, val_m = _train(arch, device)
        except Exception as exc:
            print(f"ERROR training {arch}: {exc}", file=sys.stderr)
            continue

        row = {"architecture": clean, "weights": str(weights), **val_m}

        if weights.exists():
            try:
                row.update(_evaluate(weights, device))
            except Exception as exc:
                print(f"ERROR evaluating {arch}: {exc}", file=sys.stderr)
        else:
            print(f"  WARNING: weights not found at {weights}", file=sys.stderr)

        rows.append(row)

    if not rows:
        print("No results.", file=sys.stderr)
        return

    df = pd.DataFrame(rows)
    sort_col = "test_mAP@50-95" if "test_mAP@50-95" in df.columns else "val_mAP@50-95"
    df = df.sort_values(by=sort_col, ascending=False).reset_index(drop=True)

    print("\n=== LWIR Results ===")
    try:
        print(df.to_markdown(index=False))
    except ImportError:
        print(df.to_string(index=False))

    out = Path(PROJECT) / "results.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full LWIR training and evaluation sweep.")
    parser.add_argument(
        "--device", default="mps",
        help="Compute device: mps, cpu, 0 (CUDA GPU index). Default: mps",
    )
    args = parser.parse_args()
    main(args.device)
