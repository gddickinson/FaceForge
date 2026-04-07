"""Generate virtual scanner cross-section images for the README.

Loads the scene headlessly, collects meshes, and runs the ScannerEngine
at various orientations/modes to produce 2D cross-section images.

Usage::

    PYTHONPATH=src python -m tools.generate_scanner_images
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, ".")

import numpy as np
from PIL import Image, ImageDraw

OUTPUT_DIR = Path("docs/images")

# Scan plane orientations (from scanner_window.py)
ORIENTATIONS = {
    "axial": {
        "normal": np.array([0, 0, -1], dtype=np.float32),
        "right":  np.array([1, 0, 0], dtype=np.float32),
        "up":     np.array([0, -1, 0], dtype=np.float32),
    },
    "coronal": {
        "normal": np.array([0, -1, 0], dtype=np.float32),
        "right":  np.array([1, 0, 0], dtype=np.float32),
        "up":     np.array([0, 0, -1], dtype=np.float32),
    },
    "sagittal": {
        "normal": np.array([1, 0, 0], dtype=np.float32),
        "right":  np.array([0, -1, 0], dtype=np.float32),
        "up":     np.array([0, 0, -1], dtype=np.float32),
    },
}


def stitch_horizontal(images: list[Image.Image], gap: int = 8,
                      bg: tuple = (24, 26, 32)) -> Image.Image:
    """Stitch PIL images horizontally."""
    if not images:
        return Image.new("RGB", (1, 1))
    h = max(img.height for img in images)
    w = sum(img.width for img in images) + gap * (len(images) - 1)
    out = Image.new("RGB", (w, h), bg)
    x = 0
    for img in images:
        out.paste(img, (x, (h - img.height) // 2))
        x += img.width + gap
    return out


def stitch_grid(images: list[Image.Image], cols: int, gap: int = 8,
                labels: list[str] | None = None,
                bg: tuple = (24, 26, 32)) -> Image.Image:
    """Stitch PIL images in a grid with optional labels."""
    if not images:
        return Image.new("RGB", (1, 1))
    cw = max(img.width for img in images)
    ch = max(img.height for img in images)
    lh = 28 if labels else 0
    rows = (len(images) + cols - 1) // cols
    w = cols * cw + (cols - 1) * gap
    h = rows * (ch + lh) + (rows - 1) * gap
    out = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(out)
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        x = c * (cw + gap) + (cw - img.width) // 2
        y = r * (ch + lh + gap)
        if labels and i < len(labels):
            tw = draw.textlength(labels[i])
            draw.text((c * (cw + gap) + (cw - tw) // 2, y + 4),
                       labels[i], fill=(200, 200, 210))
            y += lh
        out.paste(img, (x, y))
    return out


def array_to_pil(arr: np.ndarray, mode: str = "ct") -> Image.Image:
    """Convert a scanner output array to a PIL Image."""
    if arr.ndim == 3:
        # Anatomical mode: RGB float32 → uint8
        rgb = np.clip(arr * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(rgb, "RGB")
    else:
        # Grayscale float32 → uint8
        gray = np.clip(arr * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(gray, "L").convert("RGB")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    from tools.headless_loader import load_headless_scene, load_layer
    from faceforge.scanner.engine import ScannerEngine
    from faceforge.scanner.tissue_map import TissueMapper

    print("Loading headless scene...")
    t0 = time.time()
    hs = load_headless_scene()
    print(f"  Scene loaded in {time.time() - t0:.1f}s")

    # Collect meshes from scene graph (same as app.py does)
    meshes = hs.scene.collect_meshes()
    print(f"  Meshes: {len(meshes)}")

    # Load additional layers for richer scans
    for layer_name in ["back_muscles", "torso_muscles", "shoulder_muscles"]:
        try:
            load_layer(hs, layer_name)
            print(f"  Loaded {layer_name}")
        except Exception as e:
            print(f"  {layer_name}: {e}")

    # Re-collect after loading layers
    meshes = hs.scene.collect_meshes()
    print(f"  Meshes after layers: {len(meshes)}")

    # Initialize scanner
    tissue_mapper = TissueMapper()
    engine = ScannerEngine(tissue_mapper)
    engine.cache_meshes(meshes)
    print(f"  Scanner cached {len(engine._cache)} mesh groups\n")

    resolution = 256

    # ── 1. Head axial slices at different Z levels ──────────────────
    print("[1/4] Head axial slices...")
    orient = ORIENTATIONS["axial"]
    z_levels = [
        ("Eyes", 8.0),
        ("Jaw", -2.0),
        ("Neck", -12.0),
    ]
    imgs, labels = [], []
    for label, z in z_levels:
        origin = np.array([0, 0, z], dtype=np.float32)
        result = engine.scan(
            origin=origin, **orient,
            width=40.0, height=40.0, depth=5.0,
            resolution=resolution, mode="ct", reduction="mean",
        )
        imgs.append(array_to_pil(result))
        labels.append(f"Axial — {label} (Z={z:.0f})")

    stitch_grid(imgs, cols=3, labels=labels).save(
        str(OUTPUT_DIR / "scanner_axial.png"))
    print("  OK")

    # ── 2. Different scan modes on same slice ──────────────────────
    print("[2/4] Scan modes comparison...")
    origin = np.array([0, 0, -5], dtype=np.float32)  # mid-head
    modes = [
        ("CT", "ct", "mean"),
        ("MRI T1", "mri_t1", "mean"),
        ("MRI T2", "mri_t2", "mean"),
        ("Anatomical", "anatomical", "mean"),
    ]
    imgs, labels = [], []
    for label, mode, reduction in modes:
        result = engine.scan(
            origin=origin, **orient,
            width=40.0, height=40.0, depth=5.0,
            resolution=resolution, mode=mode, reduction=reduction,
        )
        imgs.append(array_to_pil(result, mode))
        labels.append(label)

    stitch_grid(imgs, cols=4, labels=labels).save(
        str(OUTPUT_DIR / "scanner_modes.png"))
    print("  OK")

    # ── 3. Body cross-sections (different orientations) ────────────
    print("[3/4] Body cross-sections...")
    body_scans = [
        ("Axial — Chest", "axial",    [0, 0, -30],   120, 80, 10),
        ("Coronal — Torso", "coronal", [0, 5, -60],   100, 150, 10),
        ("Sagittal — Side", "sagittal", [0, 0, -60],   80, 150, 10),
    ]
    imgs, labels = [], []
    for label, orient_name, origin_list, w, h, d in body_scans:
        orient = ORIENTATIONS[orient_name]
        origin = np.array(origin_list, dtype=np.float32)
        result = engine.scan(
            origin=origin, **orient,
            width=float(w), height=float(h), depth=float(d),
            resolution=resolution, mode="ct", reduction="mean",
        )
        imgs.append(array_to_pil(result))
        labels.append(label)

    stitch_grid(imgs, cols=3, labels=labels).save(
        str(OUTPUT_DIR / "scanner_body.png"))
    print("  OK")

    # ── 4. X-ray projections ───────────────────────────────────────
    print("[4/4] X-ray projections...")
    xray_scans = [
        ("X-ray — Head Front", "coronal", [0, 0, 0], 50, 50, 40),
        ("X-ray — Chest Front", "coronal", [0, 0, -35], 120, 100, 40),
        ("X-ray — Hand", "coronal", [35, 0, -85], 30, 30, 20),
    ]
    imgs, labels = [], []
    for label, orient_name, origin_list, w, h, d in xray_scans:
        orient = ORIENTATIONS[orient_name]
        origin = np.array(origin_list, dtype=np.float32)
        result = engine.scan(
            origin=origin, **orient,
            width=float(w), height=float(h), depth=float(d),
            resolution=resolution, mode="xray", reduction="sum",
        )
        imgs.append(array_to_pil(result, "xray"))
        labels.append(label)

    stitch_grid(imgs, cols=3, labels=labels).save(
        str(OUTPUT_DIR / "scanner_xray.png"))
    print("  OK")

    print(f"\nAll scanner images saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
