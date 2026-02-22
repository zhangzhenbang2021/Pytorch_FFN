import numpy as np
from pathlib import Path
import imageio.v2 as imageio


# ======================
# User config (edit here only)
# ======================
NPZ_PATH = "./tmp/ffn_output/0/0/seg-0_0_0.npz"     # Input NPZ path
OUT_DIR = "./tmp/ffn_output/sample_pngs_inference"   # Output directory
KEY = None                             # None = auto-pick; or "segmentation"/"seg"/"prob"
MODE = "labels"                       # "labels" or "gray"
AXIS = 0                              # 0=z, 1=y, 2=x
START = 0                             # Start slice index
END = -1                              # End slice index (-1 = to end)
SEED = 0                              # Random seed for label colors


# ======================
# Utility functions
# ======================
def pick_key(npz: np.lib.npyio.NpzFile, key: str | None) -> str:
    print("Available keys in NPZ:", npz.files)
    if key is not None:
        if key not in npz.files:
            raise KeyError(f"Key '{key}' not found. Available keys: {npz.files}")
        return key
    preferred = ["segmentation", "seg", "labels", "label", "prob", "im", "image"]
    for k in preferred:
        if k in npz.files:
            return k
    if not npz.files:
        raise ValueError("NPZ file contains no arrays.")
    return npz.files[0]


def to_uint8_gray(slice2d: np.ndarray) -> np.ndarray:
    x = slice2d.astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    mn, mx = float(x.min()), float(x.max())
    if mx <= mn:
        return np.zeros_like(x, dtype=np.uint8)
    x = (x - mn) / (mx - mn)
    return (x * 255.0).clip(0, 255).astype(np.uint8)


def labels_to_rgb(slice2d: np.ndarray, seed: int = 0, max_labels: int = 200000) -> np.ndarray:
    lab = slice2d.astype(np.int64, copy=False)
    h, w = lab.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    uniq = np.unique(lab)
    if uniq.size > max_labels:
        gray = to_uint8_gray(lab)
        return np.stack([gray, gray, gray], axis=-1)

    rng = np.random.default_rng(seed)
    colors = {0: (0, 0, 0)}
    for u in uniq:
        if u == 0:
            continue
        colors[u] = tuple(rng.integers(30, 256, size=3, dtype=np.uint8).tolist())

    for u, c in colors.items():
        if u == 0:
            continue
        rgb[lab == u] = c
    return rgb


# ======================
# Main logic
# ======================
def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(NPZ_PATH) as npz:
        key = pick_key(npz, KEY)
        vol = npz[key]
    print('shape of loaded volume:', vol.shape)
    
    if vol.ndim != 3:
        raise ValueError(f"Expected a 3D array, got shape {vol.shape} for key '{key}'")

    # Move slice axis to dimension 0
    vol = np.moveaxis(vol, AXIS, 0)   # [num_slices, H, W]
    n = vol.shape[0]

    start = max(0, START)
    end = n if END == -1 else min(n, END)
    if start >= end:
        raise ValueError(f"Invalid slice range: start={start}, end={end}, total={n}")

    print(f"Loaded: {NPZ_PATH}")
    print(f"Using key: {key}")
    print(f"Volume shape after axis move: {vol.shape}")
    print(f"Saving slices [{start}:{end}) to {out_dir.resolve()}")

    digits = len(str(end - 1))
    for i in range(start, end):
        sl = vol[i]
        if MODE == "labels":
            img = labels_to_rgb(sl, seed=SEED)
        else:
            img = to_uint8_gray(sl)

        fname = out_dir / f"{i:0{digits}d}.png"
        imageio.imwrite(fname, img)

    print("Done.")


if __name__ == "__main__":
    main()
