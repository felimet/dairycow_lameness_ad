import torch
from torch import nn
from .common import Learner


class ConvAE(Learner):
    def __init__(self, cfg, channels=1):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        encoder = []
        incoming = channels
        for outgoing in cfg.encoder_channels:
            encoder.extend([nn.Conv2d(incoming, outgoing, cfg.ae_kernel, cfg.ae_stride, cfg.ae_padding),
                            nn.BatchNorm2d(outgoing), nn.ReLU()])
            incoming = outgoing
        decoder = []
        for outgoing in (*reversed(cfg.encoder_channels[:-1]), channels):
            decoder.append(nn.ConvTranspose2d(incoming, outgoing, cfg.ae_kernel,
                                             cfg.ae_stride, cfg.ae_padding))
            decoder.extend([nn.BatchNorm2d(outgoing), nn.ReLU()] if outgoing != channels
                           else [nn.Sigmoid()])
            incoming = outgoing
        self.model = nn.Sequential(*encoder, *decoder).to(cfg.device)

    def loss(self, x):
        return (self.model(x) - x).square().mean()

    def anomaly_map(self, x):
        return (self.model(x) - x).square().mean(dim=1)
