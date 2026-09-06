import logging
import numpy as np
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

LOGGER = logging.getLogger(__name__)


class Subspace:
    def __init__(self, cfg, channels=None):
        self.cfg = cfg

    def fit(self, train_x):
        try:
            if len(train_x) < 2:
                raise ValueError("PCA requires at least two normal training passes")
            flat = np.asarray(train_x).reshape(len(train_x), -1)
            count = min(self.cfg.pca_components, len(flat) - 1, flat.shape[1])
            self.model = PCA(n_components=count, svd_solver="full", random_state=self.cfg.seed)
            with threadpool_limits(limits=self.cfg.cpu_threads):
                self.model.fit(flat)
            return self
        except Exception:
            LOGGER.exception("Subspace fitting failed")
            raise

    def localize(self, x):
        flat = np.asarray(x).reshape(len(x), -1)
        residual = flat - self.model.inverse_transform(self.model.transform(flat))
        return np.square(residual.reshape(x.shape)).mean(axis=1)

    def score(self, x):
        return self.localize(x).reshape(len(x), -1).mean(axis=1)
