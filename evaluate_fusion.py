"""
Fusion evaluation on spatiotemporally synced 3-band videos.

No ground truth required. Evaluates the fusion algorithm using
confidence-based metrics computed across all frames of the synced streams.

Variants evaluated:
  rgb      — RGB detection only
  lwir     — LWIR detection only (warped to RGB space)
  uv       — UV detection only  (warped to RGB space)
  wbf      — Weighted Boxes Fusion, all 3 bands
  bayesian — Bayesian fusion, all 3 bands (chained)
  spatial  — Center-distance spatial clustering, all 3 bands

Metrics (no GT needed):
  mean_conf          — mean confidence of all detections across all frames
  mean_dets_per_frame — average number of detections per frame
  detection_rate     — fraction of frames with at least one detection
  conf_stability     — 1 - std(per-frame mean confidence); higher = more stable
  multi_band_rate    — fraction of detections confirmed by 2+ bands (spatial only)
  all_band_rate      — fraction of detections confirmed by all 3 bands (spatial only)
"""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import yaml
from ultralytics import YOLO

from fusion import filter_lwir, filter_rgb, fuse, load_homography, warp_boxes

_VARIANT_COLOR = {
    "rgb":      "#4daafc",
    "lwir":     "#f0883e",
    "uv":       "#bc8cff",
    "wbf":      "#3fb950",
    "bayesian": "#f78166",
    "spatial":  "#ffa657",
}


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── chart helpers ─────────────────────────────────────────────────────────────

def _setup_dark_ax(ax):
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#c9d1d9", labelsize=9)
    ax.xaxis.label.set_color("#c9d1d9")
    ax.yaxis.label.set_color("#c9d1d9")
    ax.title.set_color("#58a6ff")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")


def _bar_chart(variants, values, title, ylabel, out_path, ylim=None, fmt=".3f"):
    colors = [_VARIANT_COLOR.get(v, "#8b949e") for v in variants]
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0d1117")
    _setup_dark_ax(ax)
    bars = ax.bar(variants, values, color=colors, alpha=0.9, width=0.6)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:{fmt}}", ha="center", va="bottom", fontsize=9, color="#c9d1d9")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def _grouped_bar_chart(variants, values_a, values_b, label_a, label_b, title, ylabel, out_path):
    colors = [_VARIANT_COLOR.get(v, "#8b949e") for v in variants]
    x = np.arange(len(variants))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0d1117")
    _setup_dark_ax(ax)
    ax.bar(x - w / 2, values_a, w, label=label_a, color=colors, alpha=0.9)
    ax.bar(x + w / 2, values_b, w, label=label_b, color=colors, alpha=0.55, hatch="//")
    ax.set_xticks(x)
    ax.set_xticklabels(variants)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, max(max(values_a), max(values_b)) * 1.15)
    ax.legend(facecolor="#21262d", labelcolor="#c9d1d9", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def _agreement_bar(df_spatial: pd.DataFrame, out_path: Path) -> None:
    categories = ["Single-band", "2-band", "3-band"]
    multi_band = df_spatial["multi_band_rate"].iloc[0]
    if np.isnan(multi_band):
        return
    single = 1.0 - multi_band
    multi  = multi_band - df_spatial["all_band_rate"].iloc[0]
    all3   = df_spatial["all_band_rate"].iloc[0]
    values = [single, multi, all3]
    colors = ["#484f58", "#3fb950", "#58a6ff"]

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#0d1117")
    _setup_dark_ax(ax)
    bars = ax.bar(categories, values, color=colors, alpha=0.9, width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.1%}", ha="center", va="bottom", fontsize=10, color="#c9d1d9")
    ax.set_ylabel("Fraction of detections")
    ax.set_title("Spatial Fusion — Band Agreement per Detection")
    ax.set_ylim(0, 1.1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def _save_table(df: pd.DataFrame, out_path: Path) -> None:
    display_cols = ["variant", "mean_conf", "mean_dets_per_frame",
                    "detection_rate", "conf_stability", "multi_band_rate", "all_band_rate"]
    tdf = df[display_cols].copy()
    tdf.columns = ["Variant", "Mean Conf", "Dets/Frame",
                   "Det Rate", "Conf Stability", "≥2-Band Rate", "3-Band Rate"]

    # Format floats
    for col in tdf.columns[1:]:
        tdf[col] = tdf[col].apply(lambda v: f"{v:.3f}" if isinstance(v, float) else v)

    fig, ax = plt.subplots(figsize=(13, max(2.5, 0.5 * (len(tdf) + 1))))
    fig.patch.set_facecolor("#0d1117")
    ax.axis("off")

    cell_colors = []
    for _, row in df.iterrows():
        base = _VARIANT_COLOR.get(row["variant"], "#8b949e")
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
    tbl.scale(1, 1.6)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#30363d")
        if r == 0:
            cell.set_facecolor("#21262d")
            cell.set_text_props(color="#58a6ff", fontweight="bold")
        else:
            cell.set_text_props(color="#c9d1d9")

    ax.set_title("Fusion Confidence Evaluation — Synced 3-Band Streams",
                 color="#58a6ff", fontsize=11, fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main(cfg: dict) -> pd.DataFrame:
    out_dir = Path(cfg.get("project", "results/fusion"))
    out_dir.mkdir(parents=True, exist_ok=True)

    rgb_model  = YOLO(cfg["rgb_model"])
    lwir_model = YOLO(cfg["lwir_model"])
    uv_model   = YOLO(cfg["uv_model"]) if cfg.get("uv_model") else None

    H_lwir = load_homography(cfg.get("lwir_homography"))
    H_uv   = load_homography(cfg.get("uv_homography"))

    iou_thr  = float(cfg.get("iou_thr",  0.55))
    skip_thr = float(cfg.get("skip_thr", 0.01))
    imgsz    = int(cfg.get("imgsz",     960))
    device   = cfg.get("device", "cpu")
    max_frames = cfg.get("max_frames", None)

    infer_kw = dict(imgsz=imgsz, device=device, verbose=False, conf=skip_thr)

    rgb_cap  = cv2.VideoCapture(cfg["rgb_video"])
    lwir_cap = cv2.VideoCapture(cfg["lwir_video"])
    uv_cap   = cv2.VideoCapture(cfg["uv_video"]) if cfg.get("uv_video") else None

    # Read canvas dimensions from first frame
    ok_r, _rf = rgb_cap.read()
    rgb_wh = (_rf.shape[1], _rf.shape[0]) if ok_r else (1440, 1080)
    rgb_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    ok_l, _lf = lwir_cap.read()
    lwir_wh = (_lf.shape[1], _lf.shape[0]) if ok_l else (1440, 1080)
    lwir_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    uv_wh = None
    if uv_cap is not None:
        ok_u, _uf = uv_cap.read()
        uv_wh = (_uf.shape[1], _uf.shape[0]) if ok_u else (1440, 1080)
        uv_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    total = int(min(
        rgb_cap.get(cv2.CAP_PROP_FRAME_COUNT),
        lwir_cap.get(cv2.CAP_PROP_FRAME_COUNT),
        uv_cap.get(cv2.CAP_PROP_FRAME_COUNT) if uv_cap else float("inf"),
    ))
    if max_frames is not None:
        total = min(total, max_frames)

    variants = ["rgb", "lwir", "uv", "wbf", "bayesian", "spatial"]

    # Per-frame accumulators
    frame_confs: dict[str, list[float]]  = {v: [] for v in variants}
    frame_counts: dict[str, list[int]]   = {v: [] for v in variants}
    # Spatial-only: per-detection sensor counts
    spatial_sensor_counts: list[int] = []

    print(f"Evaluating {total} frames across 3 synced bands ...")

    for frame_i in range(total):
        ok_r, rgb_frame  = rgb_cap.read()
        ok_l, lwir_frame = lwir_cap.read()
        ok_u, uv_frame   = (uv_cap.read() if uv_cap else (False, None))

        if not ok_r and not ok_l:
            break

        # Inference
        rgb_res  = rgb_model(rgb_frame,   **infer_kw)[0] if ok_r else None
        lwir_res = lwir_model(lwir_frame, **infer_kw)[0] if ok_l else None
        uv_res   = (uv_model(uv_frame, **infer_kw)[0]
                    if uv_model and ok_u and uv_frame is not None else None)

        _empty = (np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32))
        boxes_r,     scores_r = filter_rgb(rgb_res)   if rgb_res  is not None else _empty
        boxes_l_raw, scores_l = filter_lwir(lwir_res) if lwir_res is not None else _empty
        boxes_u_raw, scores_u = filter_lwir(uv_res)   if uv_res   is not None else _empty

        # Warp LWIR and UV detections to RGB canvas space
        boxes_l = warp_boxes(boxes_l_raw, H_lwir, lwir_wh, rgb_wh) if H_lwir is not None else boxes_l_raw
        boxes_u = (warp_boxes(boxes_u_raw, H_uv, uv_wh, rgb_wh)
                   if H_uv is not None and uv_wh is not None else boxes_u_raw)

        # All 6 variants in RGB space
        boxes_wbf, scores_wbf = fuse(
            boxes_r, scores_r, boxes_l, scores_l,
            boxes3=boxes_u, scores3=scores_u,
            method="wbf", iou_thr=iou_thr, skip_thr=skip_thr,
        )
        boxes_bay, scores_bay = fuse(
            boxes_r, scores_r, boxes_l, scores_l,
            boxes3=boxes_u, scores3=scores_u,
            method="bayesian", iou_thr=iou_thr,
        )
        boxes_spa, scores_spa, details_spa = fuse(
            boxes_r, scores_r, boxes_l, scores_l,
            boxes3=boxes_u, scores3=scores_u,
            method="spatial", image_wh=rgb_wh, skip_thr=skip_thr,
            source_names=["rgb", "lwir", "uv"],
            return_details=True,
        )

        frame_preds = {
            "rgb":      scores_r,
            "lwir":     scores_l,
            "uv":       scores_u,
            "wbf":      scores_wbf,
            "bayesian": scores_bay,
            "spatial":  scores_spa,
        }

        for v, scores in frame_preds.items():
            frame_counts[v].append(len(scores))
            frame_confs[v].append(float(scores.mean()) if len(scores) > 0 else float("nan"))

        # Spatial: record how many bands confirmed each detection
        if details_spa:
            for d in details_spa:
                spatial_sensor_counts.append(d["sensor_count"])

        if (frame_i + 1) % 100 == 0:
            print(f"  {frame_i + 1}/{total} frames", flush=True)

    rgb_cap.release()
    lwir_cap.release()
    if uv_cap is not None:
        uv_cap.release()

    # ── aggregate metrics ─────────────────────────────────────────────────────
    rows = []
    for v in variants:
        counts = np.array(frame_counts[v])
        confs  = np.array([c for c in frame_confs[v] if not np.isnan(c)])

        mean_conf   = float(confs.mean())  if len(confs)  > 0 else 0.0
        conf_std    = float(confs.std())   if len(confs)  > 1 else 0.0
        mean_dets   = float(counts.mean()) if len(counts) > 0 else 0.0
        det_rate    = float((counts > 0).mean()) if len(counts) > 0 else 0.0
        stability   = round(1.0 - conf_std, 4)

        if v == "spatial" and spatial_sensor_counts:
            sc = np.array(spatial_sensor_counts)
            multi_band = float((sc >= 2).mean())
            all_band   = float((sc >= 3).mean())
        else:
            multi_band = float("nan")
            all_band   = float("nan")

        rows.append({
            "variant":            v,
            "mean_conf":          round(mean_conf, 4),
            "mean_dets_per_frame": round(mean_dets, 2),
            "detection_rate":     round(det_rate,  4),
            "conf_stability":     stability,
            "multi_band_rate":    round(multi_band, 4) if not np.isnan(multi_band) else float("nan"),
            "all_band_rate":      round(all_band,   4) if not np.isnan(all_band)   else float("nan"),
        })

    df = pd.DataFrame(rows)

    print("\n=== Fusion Confidence Evaluation ===")
    try:
        print(df.to_markdown(index=False))
    except ImportError:
        print(df.to_string(index=False))

    csv_path = out_dir / "eval_fusion.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved to {csv_path}")

    # ── charts ────────────────────────────────────────────────────────────────
    vs = df["variant"].tolist()

    _bar_chart(vs, df["mean_conf"].tolist(),
               "Mean Detection Confidence per Variant", "Confidence (0–1)",
               out_dir / "fusion_mean_conf.png", ylim=(0, 1.05))

    _grouped_bar_chart(vs,
               df["mean_dets_per_frame"].tolist(),
               df["detection_rate"].tolist(),
               "Dets / Frame", "Detection Rate",
               "Detections per Frame vs Detection Rate",
               "Value", out_dir / "fusion_dets.png")

    _bar_chart(vs, df["conf_stability"].tolist(),
               "Confidence Stability per Variant (1 − std)",
               "Stability (higher = more consistent)",
               out_dir / "fusion_stability.png", ylim=(0, 1.05))

    spa_row = df[df["variant"] == "spatial"]
    if not spa_row.empty and not np.isnan(spa_row["multi_band_rate"].iloc[0]):
        _agreement_bar(spa_row, out_dir / "fusion_agreement.png")

    _save_table(df, out_dir / "fusion_table.png")

    print(f"Charts saved to {out_dir}/")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate fusion variants on synced 3-band streams (confidence-based, no GT)."
    )
    parser.add_argument("--config", default="configs/eval_fusion.yaml")
    args = parser.parse_args()
    main(load_config(args.config))
