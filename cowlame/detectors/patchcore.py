import logging
import numpy as np
import torch
from torch.nn import functional as F
from .common import backbone, extract_features

LOGGER = logging.getLogger(__name__)


class PatchCore:
    def __init__(self, cfg, channels=None):
        self.cfg = cfg
        self.backbone = backbone(cfg)

    def features(self, x):
        return extract_features(self.backbone, x, self.cfg)

    def fit_features(self, features):
        try:
            patches = features.transpose(0, 2, 3, 1).reshape(-1, features.shape[1])
            count = max(1, int(len(patches) * self.cfg.bank_fraction))
            indices = np.random.default_rng(self.cfg.seed).choice(len(patches), count, replace=False)
            self.bank = torch.as_tensor(patches[indices], device=self.cfg.device)
            LOGGER.info("PatchCore bank=%d patches, dimensions=%d", count, patches.shape[1])
            return self
        except Exception:
            LOGGER.exception("PatchCore bank construction failed")
            raise

    def fit(self, train_x):
        return self.fit_features(self.features(train_x))

    @torch.inference_mode()
    def distance_maps(self, features):
        n, channels, height, width = features.shape
        patches = features.transpose(0, 2, 3, 1).reshape(-1, channels)
        distances = []
        for start in range(0, len(patches), self.cfg.query_chunk):
            query = torch.as_tensor(patches[start:start + self.cfg.query_chunk], device=self.cfg.device)
            nearest = torch.full((len(query),), torch.inf, device=self.cfg.device)
            for bank in self.bank.split(self.cfg.bank_chunk):
                nearest = torch.minimum(nearest, torch.cdist(query, bank).amin(dim=1))
            distances.append(nearest.cpu().numpy())
        return np.concatenate(distances).reshape(n, height, width)

    def score_features(self, features):
        return self.distance_maps(features).reshape(len(features), -1).max(axis=1)

    def score(self, x):
        return self.score_features(self.features(x))

    def localize(self, x):
        maps = torch.from_numpy(self.distance_maps(self.features(x)))[:, None]
        return F.interpolate(maps, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0].numpy()
