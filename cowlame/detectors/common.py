"""Shared batching, pretrained preprocessing and fixed-epoch training."""
import logging
import numpy as np
import torch
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

LOGGER = logging.getLogger(__name__)


def backbone(cfg):
    try:
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if cfg.pretrained else None
        model = mobilenet_v3_large(weights=weights).features[:cfg.patchcore_blocks]
        return model.to(cfg.device).eval().requires_grad_(False)
    except Exception:
        LOGGER.exception("Backbone construction/download failed")
        raise


def tensor(x, cfg):
    return torch.as_tensor(np.array(x, copy=True), device=cfg.device, dtype=torch.float32)


def normalized(x, cfg):
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    if x.shape[1] != 3:
        raise ValueError("Pretrained backbone requires one or three channels")
    mean = x.new_tensor(cfg.imagenet_mean)[None, :, None, None]
    std = x.new_tensor(cfg.imagenet_std)[None, :, None, None]
    return (x - mean) / std


@torch.inference_mode()
def extract_features(model, x, cfg):
    return np.concatenate([model(normalized(tensor(x[start:start + cfg.batch_size], cfg), cfg))
                           .cpu().numpy() for start in range(0, len(x), cfg.batch_size)])


class Learner:
    def fit(self, train_x, epoch_callback=None):
        try:
            torch.manual_seed(self.cfg.seed)
            # The normal training subset fits in GPU RAM; avoid repeated host transfers.
            train = tensor(train_x, self.cfg)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate)
            amp = self.cfg.amp and self.cfg.device == "cuda"
            scaler = torch.amp.GradScaler("cuda", enabled=amp)
            generator = torch.Generator(device=self.cfg.device).manual_seed(self.cfg.seed)
            for epoch in range(1, self.cfg.epochs + 1):
                self.model.train()
                order = torch.randperm(len(train), generator=generator, device=self.cfg.device)
                total = 0.0
                for ids in order.split(self.cfg.batch_size):
                    batch = train[ids]
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=self.cfg.device, enabled=amp):
                        loss = self.loss(batch)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite training loss")
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    total += loss.detach().item() * len(ids)
                self.model.eval()
                mean_loss = total / len(train)
                if epoch_callback is not None:
                    epoch_callback(epoch, mean_loss)
                LOGGER.info("%s epoch %d/%d loss=%.7f", type(self).__name__, epoch,
                            self.cfg.epochs, mean_loss)
            return self
        except Exception:
            LOGGER.exception("%s training failed", type(self).__name__)
            raise

    @torch.inference_mode()
    def localize(self, x):
        self.model.eval()
        maps = []
        for start in range(0, len(x), self.cfg.batch_size):
            batch = tensor(x[start:start + self.cfg.batch_size], self.cfg)
            maps.append(self.anomaly_map(batch).float().cpu().numpy())
        return np.concatenate(maps)

    def score(self, x):
        return self.localize(x).reshape(len(x), -1).mean(axis=1)
