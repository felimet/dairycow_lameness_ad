"""All fixed analysis settings and explicit implementation choices."""
from dataclasses import asdict, dataclass
import logging
import os
from pathlib import Path
import random

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Config:
    seed: int = 42
    folds: int = 5
    alpha: float = 0.45
    image_size: int = 128
    pca_components: int = 30
    bank_fraction: float = 0.10
    patchcore_blocks: int = 7
    batch_size: int = 64
    epochs: int = 40
    learning_rate: float = 1e-3
    encoder_channels: tuple = (32, 64, 128, 128)
    ae_kernel: int = 4
    ae_stride: int = 2
    ae_padding: int = 1
    student_layers: int = 4
    student_kernel: int = 3
    student_channel_divisor: int = 2
    imagenet_mean: tuple = (0.485, 0.456, 0.406)
    imagenet_std: tuple = (0.229, 0.224, 0.225)
    pixel_max: float = 255.0
    query_chunk: int = 4096
    bank_chunk: int = 16384
    permutation_count: int = 10000
    confidence: float = 0.95
    max_crowding: int = 3
    normal_score: int = 1
    cpu_threads: int = 4
    device: str = "cuda"
    amp: bool = True
    pretrained: bool = True
    unknown_cow_policy: str = "single-group"
    cache_version: int = 1
    modalities: tuple = ("CGI", "GEI", "FDEI", "HEI")
    detectors: tuple = ("patchcore", "subspace", "convae", "stfpm")


@dataclass(frozen=True)
class EnergyConfig:
    image_hw: tuple = (220, 318)
    hei_hw: tuple = (180, 382)
    temporal_segments: int = 3
    # Window hop is not recorded in mask_meta.json; this default is an exposed assumption (docs/IMPLEMENTATION_NOTES.md item 18).
    window_step: int = 5
    window_origin: int = 1
    denoise_peak_fraction: float = 1.0
    front_direction: str = "auto"


@dataclass(frozen=True)
class SegmentationConfig:
    imgsz: int = 512
    batch: int = 16
    workers: int = 0
    conf: float = 0.001
    iou: float = 0.7
    max_det: int = 300
    half: bool = False
    mask_ratio: int = 4
    overlap_mask: bool = True
    device: int = 0
    display_decimals: int = 3


def setup_runtime(cfg, root=None):
    """Keep downloaded assets/configuration inside the repository."""
    try:
        root = Path(root or Path(__file__).resolve().parents[1])
        for variable, child in {"TORCH_HOME": "torch", "MPLCONFIGDIR": "matplotlib",
                                "YOLO_CONFIG_DIR": "ultralytics", "TEMP": "tmp",
                                "TMP": "tmp", "XDG_CACHE_HOME": "xdg"}.items():
            folder = root / "cache" / child
            folder.mkdir(parents=True, exist_ok=True)
            os.environ[variable] = str(folder)
        import numpy as np
        import torch
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.set_num_threads(cfg.cpu_threads)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        if cfg.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; CPU fallback is forbidden")
        LOGGER.info("Configuration: %s", asdict(cfg))
    except Exception:
        LOGGER.exception("Runtime setup failed")
        raise
