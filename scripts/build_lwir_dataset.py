"""
Extract LWIR frames from mp4 videos and pair with YOLO label files.

Sources (data_annotations_lwir/):
  Senaryo1_Train_edited_lwir.mp4  + Senaryo1_lwir_annotations.zip  -> train
  Senaryo2_edited_lwir.mp4        + Senaryo2_lwir_annotations.zip  -> train
  Senaryo3_Test_edited_lwir.mp4   + Senaryo3_lwir_annotations.zip  -> test

Output (data/lwir/):
  train/images/, train/labels/
  val/images/,   val/labels/
  test/images/,  test/labels/
  data.yaml
"""

import random
import shutil
import subprocess
import zipfile
from pathlib import Path

# ── config ────────────────────────────────────────────────────────────────────
SRC      = Path("data_annotations_lwir")
DST      = Path("data/lwir")
VAL_FRAC = 0.15
SEED     = 42
IMG_EXT  = "jpg"

SCENARIOS = [
    ("Senaryo1_Train_edited_lwir.mp4", "Senaryo1_lwir_annotations.zip", "train"),
    ("Senaryo2_edited_lwir.mp4",        "Senaryo2_lwir_annotations.zip", "train"),
    ("Senaryo3_Test_edited_lwir.mp4",   "Senaryo3_lwir_annotations.zip", "test"),
]

# ── helpers ───────────────────────────────────────────────────────────────────
def extract_labels(zip_path: Path, label_dir: Path):
    """Unzip only the label txt files into label_dir."""
    with zipfile.ZipFile(zip_path) as zf:
        for entry in zf.namelist():
            if entry.startswith("labels/") and entry.endswith(".txt"):
                stem = Path(entry).name
                target = label_dir / stem
                target.write_bytes(zf.read(entry))


def frame_number(label_stem: str) -> int:
    """frame_000407 -> 407"""
    return int(label_stem.replace("frame_", ""))


def extract_frames(video: Path, frame_numbers: list[int], out_dir: Path):
    """
    Extract specific frames from a video using ffmpeg select filter.
    Emits frame_{n:06d}.jpg for each requested frame number.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # Build select expression: eq(n,407)+eq(n,681)+...
    select = "+".join(f"eq(n\\,{n})" for n in sorted(frame_numbers))
    cmd = [
        "ffmpeg", "-y", "-i", str(video),
        "-vf", f"select='{select}'",
        "-vsync", "vfr",
        "-frame_pts", "1",      # use presentation timestamp as filename
        "-q:v", "2",
        str(out_dir / f"frame_%06d.{IMG_EXT}"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("ffmpeg stderr:", result.stderr[-500:])
        raise RuntimeError(f"ffmpeg failed for {video.name}")


def extract_frames_bulk(video: Path, frame_numbers: set[int], out_dir: Path):
    """
    Faster: extract ALL frames, keep only the needed ones, delete the rest.
    Preferable when the labeled fraction is high (>50%).
    """
    tmp = out_dir / "_tmp"
    tmp.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y", "-i", str(video),
        "-vsync", "0",
        "-q:v", "2",
        str(tmp / f"frame_%06d.{IMG_EXT}"),
    ]
    print(f"  Extracting all frames from {video.name} ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr[-300:]}")

    kept = 0
    for img in tmp.glob(f"*.{IMG_EXT}"):
        # ffmpeg names files starting at 1: frame_000001 = frame index 0
        idx = int(img.stem.split("_")[1]) - 1   # 0-based frame index
        if idx in frame_numbers:
            stem = f"frame_{idx:06d}"
            img.rename(out_dir / f"{stem}.{IMG_EXT}")
            kept += 1
        else:
            img.unlink()

    shutil.rmtree(tmp)
    print(f"  Kept {kept}/{len(frame_numbers)} expected labeled frames")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    random.seed(SEED)

    for split in ("train", "val", "test"):
        for sub in ("images", "labels"):
            (DST / split / sub).mkdir(parents=True, exist_ok=True)

    train_stems = []   # collect all train stems to split into train/val

    for video_name, zip_name, role in SCENARIOS:
        video   = SRC / video_name
        zip_path = SRC / zip_name
        scenario = zip_name.replace("_lwir_annotations.zip", "")
        print(f"\n{'='*55}")
        print(f"  {scenario}  ({role})")
        print(f"{'='*55}")

        # 1. Extract labels to a temp staging dir
        stage_labels = DST / "_stage" / scenario / "labels"
        stage_labels.mkdir(parents=True, exist_ok=True)
        print("  Extracting labels ...")
        extract_labels(zip_path, stage_labels)

        stems = [p.stem for p in stage_labels.glob("*.txt")]
        frame_nums = {frame_number(s) for s in stems}
        print(f"  Found {len(stems)} labeled frames")

        # 2. Extract matching frames from video
        stage_images = DST / "_stage" / scenario / "images"
        stage_images.mkdir(parents=True, exist_ok=True)
        extract_frames_bulk(video, frame_nums, stage_images)

        # 3. Collect (image, label) pairs — prefix with scenario to avoid collisions
        prefix = scenario.replace("_", "").lower()  # e.g. "senaryo1"
        pairs = []
        for lbl in stage_labels.glob("*.txt"):
            img = stage_images / f"{lbl.stem}.{IMG_EXT}"
            if img.exists():
                pairs.append((img, lbl, prefix))

        missing = len(stems) - len(pairs)
        if missing:
            print(f"  WARNING: {missing} labels have no matching image")

        if role == "test":
            for img, lbl, pfx in pairs:
                name = f"{pfx}_{img.name}"
                shutil.copy(img, DST / "test" / "images" / name)
                shutil.copy(lbl, DST / "test" / "labels" / f"{pfx}_{lbl.name}")
            print(f"  -> test: {len(pairs)} pairs")
        else:
            train_stems.extend(pairs)

    # 4. Split train pool into train / val
    random.shuffle(train_stems)
    n_val   = int(len(train_stems) * VAL_FRAC)
    val_set = train_stems[:n_val]
    trn_set = train_stems[n_val:]

    for img, lbl, pfx in trn_set:
        name = f"{pfx}_{img.name}"
        shutil.copy(img, DST / "train" / "images" / name)
        shutil.copy(lbl, DST / "train" / "labels" / f"{pfx}_{lbl.name}")

    for img, lbl, pfx in val_set:
        name = f"{pfx}_{img.name}"
        shutil.copy(img, DST / "val" / "images" / name)
        shutil.copy(lbl, DST / "val" / "labels" / f"{pfx}_{lbl.name}")

    print(f"\n  -> train: {len(trn_set)}, val: {len(val_set)}")

    # 5. Cleanup staging area
    shutil.rmtree(DST / "_stage")

    # 6. Write data.yaml
    yaml_content = (
        f"path: {DST.resolve()}\n"
        "train: train/images\n"
        "val:   val/images\n"
        "test:  test/images\n"
        "nc: 1\n"
        "names:\n"
        "  0: drone\n"
    )
    (DST / "data.yaml").write_text(yaml_content)

    # 7. Summary
    print("\n=== Final dataset (data/lwir/) ===")
    for split in ("train", "val", "test"):
        imgs = len(list((DST / split / "images").glob(f"*.{IMG_EXT}")))
        lbls = len(list((DST / split / "labels").glob("*.txt")))
        print(f"  {split:5s}: {imgs} images, {lbls} labels")


if __name__ == "__main__":
    main()
