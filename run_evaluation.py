"""
Master evaluation pipeline.

Steps:
  1. Run per-modality individual eval (evaluate.py) on all RGB + LWIR models.
  2. Auto-select the best model per modality by mAP@50.
  3. Inject best models into the fusion config.
  4. Run fusion eval (evaluate_fusion.py) with all 6 variants.
  5. Print a final summary to stdout.
"""

import argparse
import sys
from pathlib import Path

import yaml

from evaluate import load_config as load_eval_cfg, main as run_individual
from evaluate_fusion import load_config as load_fusion_cfg, main as run_fusion


def _best_per_modality(df) -> dict[str, str]:
    """Return {modality: weights_path} for the highest mAP@50 model in each group."""
    best = {}
    for modality, group in df.groupby("modality"):
        idx = group["mAP@50"].idxmax()
        best[modality] = group.loc[idx, "weights"]
    return best


def _print_separator(title: str) -> None:
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def main(eval_cfg_path: str, fusion_cfg_path: str) -> None:
    # ── Step 1: individual eval ───────────────────────────────────────────────
    _print_separator("STEP 1 — Individual Model Evaluation")
    eval_cfg = load_eval_cfg(eval_cfg_path)
    ind_df = run_individual(eval_cfg)

    if ind_df.empty:
        print("Individual eval produced no results — aborting.", file=sys.stderr)
        sys.exit(1)

    # ── Step 2: pick best models ──────────────────────────────────────────────
    _print_separator("STEP 2 — Selecting Best Model per Modality")
    best = _best_per_modality(ind_df)
    for mod, weights in best.items():
        print(f"  {mod.upper()}: {weights}")

    # ── Step 3: inject into fusion config ─────────────────────────────────────
    _print_separator("STEP 3 — Fusion Evaluation (6 variants, 3 bands)")
    fusion_cfg = load_fusion_cfg(fusion_cfg_path)
    fusion_cfg["rgb_model"]  = best.get("rgb",  fusion_cfg.get("rgb_model"))
    fusion_cfg["lwir_model"] = best.get("lwir", fusion_cfg.get("lwir_model"))

    if fusion_cfg.get("rgb_model") is None or fusion_cfg.get("lwir_model") is None:
        print("ERROR: rgb_model or lwir_model still unset after individual eval.",
              file=sys.stderr)
        sys.exit(1)

    # ── Step 4: fusion eval ───────────────────────────────────────────────────
    fus_df = run_fusion(fusion_cfg)

    # ── Step 5: summary ───────────────────────────────────────────────────────
    _print_separator("FINAL SUMMARY")
    print("\nBest individual models:")
    for mod, weights in best.items():
        row = ind_df[ind_df["weights"] == weights].iloc[0]
        print(f"  {mod.upper():4s}  {Path(weights).stem:20s}  "
              f"mAP@50={row['mAP@50']:.4f}  mAP@50-95={row['mAP@50-95']:.4f}  "
              f"P={row['precision']:.4f}  R={row['recall']:.4f}")

    print("\nFusion results:")
    for _, row in fus_df.sort_values("mAP@50", ascending=False).iterrows():
        print(f"  {row['variant']:10s}  mAP@50={row['mAP@50']:.4f}  "
              f"P={row['precision']:.4f}  R={row['recall']:.4f}")

    ind_out = Path(eval_cfg.get("project", "results/individual"))
    fus_out = Path(fusion_cfg.get("project", "results/fusion"))
    print(f"\nResults saved to:")
    print(f"  Individual: {ind_out}/")
    print(f"  Fusion:     {fus_out}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run full evaluation: individual models + fusion benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="configs/eval.yaml",
        help="Individual eval config YAML",
    )
    parser.add_argument(
        "--fusion-config",
        default="configs/eval_fusion.yaml",
        help="Fusion eval config YAML",
    )
    args = parser.parse_args()
    main(args.config, args.fusion_config)
