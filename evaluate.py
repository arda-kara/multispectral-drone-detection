"""Evaluate saved YOLO weights on a dataset split (default: test)."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import yaml
from ultralytics import YOLO

_MODALITY_COLOR = {"rgb": "#4daafc", "lwir": "#f0883e", "uv": "#bc8cff"}
_DEFAULT_COLOR = "#8b949e"


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def evaluate_model(entry: dict, cfg: dict) -> dict:
    weights = Path(entry["weights"])
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    model = YOLO(str(weights))
    imgsz = entry.get("imgsz", cfg.get("imgsz", 640))

    val_kwargs = dict(
        data=entry["data"],
        split=cfg.get("split", "test"),
        imgsz=imgsz,
        batch=cfg.get("batch", 8),
        workers=cfg.get("workers", 2),
        plots=cfg.get("plots", False),
        save_json=False,
        verbose=False,
    )
    if "device" in cfg:
        val_kwargs["device"] = cfg["device"]

    results = model.val(**val_kwargs)

    return {
        "modality": entry["modality"],
        "model": weights.stem,
        "weights": str(weights),
        "precision": round(results.box.mp, 4),
        "recall": round(results.box.mr, 4),
        "mAP@50": round(results.box.map50, 4),
        "mAP@50-95": round(results.box.map, 4),
        "inference_ms": round(results.speed["inference"], 2),
    }


# ── chart helpers ─────────────────────────────────────────────────────────────

def _bar_colors(df: pd.DataFrame) -> list[str]:
    return [_MODALITY_COLOR.get(m, _DEFAULT_COLOR) for m in df["modality"]]


def _setup_dark_ax(ax):
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#c9d1d9", labelsize=9)
    ax.xaxis.label.set_color("#c9d1d9")
    ax.yaxis.label.set_color("#c9d1d9")
    ax.title.set_color("#58a6ff")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")


def _save_map_bars(df: pd.DataFrame, out_dir: Path) -> None:
    labels = df["model"].tolist()
    x = np.arange(len(labels))
    w = 0.35
    colors = _bar_colors(df)

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.9), 5))
    fig.patch.set_facecolor("#0d1117")
    _setup_dark_ax(ax)

    bars50 = ax.bar(x - w / 2, df["mAP@50"], w, label="mAP@50", color=colors, alpha=0.9)
    bars95 = ax.bar(x + w / 2, df["mAP@50-95"], w, label="mAP@50-95",
                    color=colors, alpha=0.55, hatch="//")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Detection Performance — mAP per Model")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.legend(facecolor="#21262d", labelcolor="#c9d1d9", fontsize=9)

    # Annotate bars
    for bar in bars50:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom",
                fontsize=7, color="#c9d1d9")

    # Modality legend patches
    from matplotlib.patches import Patch
    modalities = df["modality"].unique()
    extra = [Patch(facecolor=_MODALITY_COLOR.get(m, _DEFAULT_COLOR), label=m.upper())
             for m in modalities]
    ax.legend(handles=extra + ax.get_legend_handles_labels()[0][:2],
              facecolor="#21262d", labelcolor="#c9d1d9", fontsize=9,
              loc="upper right")

    fig.tight_layout()
    fig.savefig(out_dir / "map_bars.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def _save_pr_bars(df: pd.DataFrame, out_dir: Path) -> None:
    labels = df["model"].tolist()
    x = np.arange(len(labels))
    w = 0.35
    colors = _bar_colors(df)

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.9), 5))
    fig.patch.set_facecolor("#0d1117")
    _setup_dark_ax(ax)

    ax.bar(x - w / 2, df["precision"], w, label="Precision", color=colors, alpha=0.9)
    ax.bar(x + w / 2, df["recall"], w, label="Recall", color=colors, alpha=0.55, hatch="//")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Precision & Recall per Model")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.legend(facecolor="#21262d", labelcolor="#c9d1d9", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_dir / "precision_recall.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def _save_speed_bars(df: pd.DataFrame, out_dir: Path) -> None:
    labels = df["model"].tolist()
    colors = _bar_colors(df)

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.9), 4))
    fig.patch.set_facecolor("#0d1117")
    _setup_dark_ax(ax)

    bars = ax.bar(labels, df["inference_ms"], color=colors, alpha=0.9)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{bar.get_height():.1f}", ha="center", va="bottom",
                fontsize=7, color="#c9d1d9")

    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Inference (ms / frame)")
    ax.set_title("Inference Speed per Model")

    fig.tight_layout()
    fig.savefig(out_dir / "inference_speed.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def _save_table(df: pd.DataFrame, out_dir: Path) -> None:
    display_cols = ["modality", "model", "mAP@50", "mAP@50-95", "precision", "recall", "inference_ms"]
    tdf = df[display_cols].copy()
    tdf.columns = ["Modality", "Model", "mAP@50", "mAP@50-95", "Precision", "Recall", "Infer ms"]

    fig_h = max(2.5, 0.4 * (len(tdf) + 1))
    fig, ax = plt.subplots(figsize=(11, fig_h))
    fig.patch.set_facecolor("#0d1117")
    ax.axis("off")
    ax.set_facecolor("#0d1117")

    cell_colors = []
    for _, row in tdf.iterrows():
        mod = row["Modality"]
        base = _MODALITY_COLOR.get(mod, _DEFAULT_COLOR)
        cell_colors.append([base + "40"] * len(tdf.columns))

    tbl = ax.table(
        cellText=tdf.values,
        colLabels=tdf.columns,
        cellLoc="center",
        loc="center",
        cellColours=cell_colors,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#30363d")
        if r == 0:
            cell.set_facecolor("#21262d")
            cell.set_text_props(color="#58a6ff", fontweight="bold")
        else:
            cell.set_text_props(color="#c9d1d9")

    ax.set_title("Individual Model Evaluation Results", color="#58a6ff",
                 fontsize=12, fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(out_dir / "summary_table.png", dpi=150, facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    plt.close(fig)


def _save_charts(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_map_bars(df, out_dir)
    _save_pr_bars(df, out_dir)
    _save_speed_bars(df, out_dir)
    _save_table(df, out_dir)
    print(f"Charts saved to {out_dir}/")


# ── main ──────────────────────────────────────────────────────────────────────

def main(cfg: dict) -> pd.DataFrame:
    out_dir = Path(cfg.get("project", "results/individual"))
    out_dir.mkdir(parents=True, exist_ok=True)

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
        return pd.DataFrame()

    df = (
        pd.DataFrame(rows)
        .sort_values(by=["modality", "mAP@50-95"], ascending=[True, False])
        .reset_index(drop=True)
    )

    print("\n=== Evaluation Results ===")
    try:
        print(df[["modality", "model", "mAP@50", "mAP@50-95", "precision",
                   "recall", "inference_ms"]].to_markdown(index=False))
    except ImportError:
        print(df.to_string(index=False))

    csv_path = out_dir / f"eval_{cfg.get('split', 'test')}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved to {csv_path}")

    _save_charts(df, out_dir)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLO models on a dataset split.")
    parser.add_argument("--config", default="configs/eval.yaml", help="Path to eval config YAML")
    args = parser.parse_args()
    main(load_config(args.config))
