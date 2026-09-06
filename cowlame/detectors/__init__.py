from .subspace import Subspace
from .patchcore import PatchCore
from .convae import ConvAE
from .stfpm import STFPM

DETECTORS = {"subspace": Subspace, "patchcore": PatchCore, "convae": ConvAE, "stfpm": STFPM}
