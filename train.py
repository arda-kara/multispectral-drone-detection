"""Train one or more YOLO architectures on a single modality dataset."""

import argparse
import gc
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from ultralytics import YOLO


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def train_architecture(model_name: str, cfg: dict, run_dir: Path) -> dict:
    torch.cuda.empty_cache()
    gc.collect()

    if "-p2" in model_name:
        base = model_name.replace("-p2", "")
        model = YOLO(f"{model_name}.yaml").load(f"{base}.pt")
        clean = model_name
    else:
        model = YOLO(model_name)
        clean = model_name.replace(".pt", "")

    batch = cfg["batch"] // 2 if "s.pt" in model_name else cfg["batch"]

    train_kwargs = dict(
        data=cfg["data"],
        epochs=cfg["epochs"],
        imgsz=cfg["imgsz"],
        batch=batch,
        project=str(run_dir.resolve()),
        name=f"{clean}_{cfg['imgsz']}px",
        cache=cfg["cache"],
        workers=cfg["workers"],
        amp=cfg["amp"],
        plots=cfg["plots"],
        exist_ok=False,
    )
    if "device" in cfg:
        train_kwargs["device"] = cfg["device"]
    if "fraction" in cfg:
        train_kwargs["fraction"] = cfg["fraction"]
    results = model.train(**train_kwargs)

    return {
        "architecture": clean,
        "modality": cfg["modality"],
        "mAP@50": round(results.box.map50, 4),
        "mAP@50-95": round(results.box.map, 4),
        "inference_ms": round(results.speed["inference"], 2),
    }


def main(cfg: dict):
    run_dir = Path(cfg["project"]) / cfg["modality"]
    run_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for model_name in cfg["architectures"]:
        print(f"\n{'='*50}")
        print(f"  {cfg['modality'].upper()} | {model_name}")
        print(f"{'='*50}")
        try:
            row = train_architecture(model_name, cfg, run_dir)
            rows.append(row)
        except Exception as exc:
            print(f"ERROR: {model_name} failed — {exc}", file=sys.stderr)

    if not rows:
        print("No results.", file=sys.stderr)
        return

    df = (
        pd.DataFrame(rows)
        .sort_values(by=["mAP@50-95", "inference_ms"], ascending=[False, True])
        .reset_index(drop=True)
    )

    print("\n=== Results ===")
    try:
        print(df.to_markdown(index=False))
    except ImportError:
        print(df.to_string(index=False))

    out = run_dir / "comparison.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO on a single modality.")
    parser.add_argument("--config", default="configs/train.yaml", help="Path to config YAML")
    args = parser.parse_args()

    main(load_config(args.config))
