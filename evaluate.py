"""Evaluate saved YOLO weights on a dataset split (default: test)."""

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml
from ultralytics import YOLO


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def evaluate_model(entry: dict, cfg: dict) -> dict:
    weights = Path(entry["weights"])
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    model = YOLO(str(weights))

    val_kwargs = dict(
        data=entry["data"],
        split=cfg.get("split", "test"),
        imgsz=cfg.get("imgsz", 640),
        batch=cfg.get("batch", 8),
        workers=cfg.get("workers", 2),
        plots=cfg.get("plots", True),
        save_json=False,
        verbose=False,
    )
    if "device" in cfg:
        val_kwargs["device"] = cfg["device"]

    results = model.val(**val_kwargs)

    return {
        "modality": entry["modality"],
        "weights": str(weights),
        "precision": round(results.box.mp, 4),
        "recall": round(results.box.mr, 4),
        "mAP@50": round(results.box.map50, 4),
        "mAP@50-95": round(results.box.map, 4),
        "inference_ms": round(results.speed["inference"], 2),
    }


def main(cfg: dict):
    rows = []
    for entry in cfg["evaluations"]:
        modality = entry["modality"]
        print(f"\n{'='*50}")
        print(f"  {modality.upper()} | {entry['weights']}")
        print(f"{'='*50}")
        try:
            row = evaluate_model(entry, cfg)
            rows.append(row)
        except Exception as exc:
            print(f"ERROR: {modality} failed — {exc}", file=sys.stderr)

    if not rows:
        print("No results.", file=sys.stderr)
        return

    df = (
        pd.DataFrame(rows)
        .sort_values(by="mAP@50-95", ascending=False)
        .reset_index(drop=True)
    )

    print("\n=== Evaluation Results ===")
    try:
        print(df.to_markdown(index=False))
    except ImportError:
        print(df.to_string(index=False))

    out = Path(cfg.get("project", "runs")) / f"eval_{cfg.get('split', 'test')}.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLO models on a dataset split.")
    parser.add_argument("--config", default="configs/eval.yaml", help="Path to eval config YAML")
    args = parser.parse_args()
    main(load_config(args.config))
