import logging
import math
import torch
from torch import nn
from torch.nn import functional as F
from .common import Learner, backbone, normalized

LOGGER = logging.getLogger(__name__)


class STFPM(Learner):
    def __init__(self, cfg, channels=1):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        self.teacher = backbone(cfg)
        with torch.no_grad():
            output = self.teacher(torch.zeros(1, 3, cfg.image_size, cfg.image_size, device=cfg.device))
        teacher_channels = output.shape[1]
        stride = cfg.image_size // output.shape[-1]
        downsample_layers = int(math.log2(stride))
        if 2 ** downsample_layers != stride or downsample_layers > cfg.student_layers:
            raise ValueError("Teacher stride cannot be matched by the four-layer student")
        layers = []
        incoming = 3
        for index in range(cfg.student_layers):
            outgoing = (teacher_channels if index == cfg.student_layers - 1
                        else teacher_channels // cfg.student_channel_divisor)
            step = 2 if index < downsample_layers else 1
            layers.append(nn.Conv2d(incoming, outgoing, cfg.student_kernel, step,
                                    cfg.student_kernel // 2))
            if index < cfg.student_layers - 1:
                layers.append(nn.ReLU())
            incoming = outgoing
        self.model = nn.Sequential(*layers).to(cfg.device)
        LOGGER.info("STFPM student: %s", self.model)

    def difference(self, x):
        x = normalized(x, self.cfg)
        with torch.no_grad():
            teacher = self.teacher(x)
        student = self.model(x)
        return F.normalize(student.float(), dim=1) - F.normalize(teacher.float(), dim=1)

    def loss(self, x):
        return self.difference(x).square().mean()

    def anomaly_map(self, x):
        maps = self.difference(x).square().sum(dim=1).sqrt()
        return F.interpolate(maps[:, None], size=x.shape[-2:], mode="nearest")[:, 0]
