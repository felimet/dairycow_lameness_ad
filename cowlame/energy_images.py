"""Reference equations (1)-(5): GEI, CGI, FDEI and HEI from binary silhouettes."""
import logging
import numpy as np
from PIL import Image
from .config import Config, EnergyConfig

LOGGER = logging.getLogger(__name__)


def centroid(mask):
    """Return (x, y) foreground centroid, or None for an empty frame."""
    rows, columns = mask.sum(axis=1), mask.sum(axis=0)
    count = rows.sum()
    if count == 0:
        return None
    return float(columns @ np.arange(mask.shape[1]) / count), float(rows @ np.arange(mask.shape[0]) / count)


def bounds(mask):
    y = np.flatnonzero(mask.any(axis=1))
    x = np.flatnonzero(mask.any(axis=0))
    if len(x) == 0:
        raise ValueError("No foreground in the selected window")
    return int(x[0]), int(y[0]), int(x[-1] + 1), int(y[-1] + 1)


def align_centroids(frames):
    centers = [centroid(frame) for frame in frames]
    regions = []
    for mask, center in zip(frames, centers):
        if center is None:
            regions.append(None)
            continue
        x0, y0, x1, y1 = bounds(mask)
        # Half-up rounding preserves integer-translation equivariance; bankers'
        # rounding would shift half-pixel centroids differently on odd/even frames.
        cx, cy = np.floor(np.asarray(center) + 0.5).astype(int)
        regions.append((x0, y0, x1, y1, cx, cy))
    valid = [region for region in regions if region is not None]
    if not valid:
        raise ValueError("All masks in the window are empty")
    left, top = min(r[0] - r[4] for r in valid), min(r[1] - r[5] for r in valid)
    right, bottom = max(r[2] - r[4] for r in valid), max(r[3] - r[5] for r in valid)
    aligned = np.zeros((len(frames), bottom - top, right - left), dtype=np.float32)
    for index, region in enumerate(regions):
        if region is None:
            continue
        x0, y0, x1, y1, cx, cy = region
        dx, dy = x0 - cx - left, y0 - cy - top
        aligned[index, dy:dy + y1 - y0, dx:dx + x1 - x0] = frames[index, y0:y1, x0:x1]
    return aligned, np.array([c is not None for c in centers])


def resized(image, hw):
    """Resize floating energy channels before final PNG quantization."""
    if image.ndim == 3:
        return np.stack([resized(image[..., i], hw) for i in range(image.shape[-1])], axis=-1)
    return np.asarray(Image.fromarray(image.astype(np.float32)).resize(
        (hw[1], hw[0]), Image.Resampling.BILINEAR), dtype=np.float32)


def synthesize(frames, cfg=EnergyConfig()):
    """Return GEI, CGI, FDEI and six HEI channel permutations in [0,1].

    D_c is the segment mean retained at its spatial peak (>= peak). The
    Methods describe the denoising step only qualitatively; this literal
    peak-retention choice is exposed in EnergyConfig and documented in
    docs/IMPLEMENTATION_NOTES.md.
    """
    frames = np.asarray(frames, dtype=bool)
    if frames.ndim != 3 or len(frames) < cfg.temporal_segments:
        raise ValueError("Expected N binary masks, N >= three temporal segments")
    aligned, valid = align_centroids(frames)
    gei = aligned[valid].mean(axis=0)
    segments = np.array_split(np.arange(len(frames)), cfg.temporal_segments)
    energies = [aligned[ids].mean(axis=0) for ids in segments]
    cgi = np.stack(energies, axis=-1)
    # B_0 := B_1: the first frame contributes no artificial entering edge.
    previous = np.concatenate([aligned[:1], aligned[:-1]], axis=0)
    differences = np.maximum(previous - aligned, 0)
    for ids, energy in zip(segments, energies):
        denoised = np.where(energy >= energy.max() * cfg.denoise_peak_fraction, energy, 0)
        differences[ids] += denoised
    fdei = differences.mean(axis=0)
    x0, y0, x1, y1 = bounds(frames.any(axis=0))
    union = frames[:, y0:y1, x0:x1]
    whole = union.mean(axis=0, dtype=np.float32)
    front, rear = np.zeros_like(whole), np.zeros_like(whole)
    centers = [centroid(frame) for frame in frames]
    nonempty = [c for c in centers if c is not None]
    direction = cfg.front_direction
    if direction == "auto":
        direction = "right" if nonempty[-1][0] >= nonempty[0][0] else "left"
    if direction not in ("left", "right"):
        raise ValueError("front_direction must be auto, left or right")
    for mask in union:
        if not mask.any():
            continue
        a, _, b, _ = bounds(mask)
        midpoint = (a + b) // 2
        left_half = mask.copy()
        left_half[:, midpoint:] = False
        right_half = mask.copy()
        right_half[:, :midpoint] = False
        front += right_half if direction == "right" else left_half
        rear += left_half if direction == "right" else right_half
    front /= len(frames)
    rear /= len(frames)
    result = {"GEI": resized(gei, cfg.image_hw), "CGI": resized(cgi, cfg.image_hw),
              "FDEI": resized(fdei, cfg.image_hw)}
    components = [resized(e, cfg.hei_hw) for e in (whole, front, rear)]
    # Lexicographic W/F/L order; combination 6 is L/F/W (rear/front/whole).
    from itertools import permutations
    for index, order in enumerate(permutations(range(cfg.temporal_segments)), 1):
        result[f"HEI_comb{index}"] = np.stack([components[i] for i in order], axis=-1)
    for name, values in zip(("WholeHEI", "FrontHEI", "LatterHEI"), components):
        result[name] = values
    return result


def save_png(path, values):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        values = np.rint(np.clip(values, 0, 1) * Config().pixel_max).astype(np.uint8)
        Image.fromarray(values).save(path)
    except Exception:
        LOGGER.exception("Energy-image PNG output failed: %s", path)
        raise
