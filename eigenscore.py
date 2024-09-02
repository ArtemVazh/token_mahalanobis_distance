import os
import copy
import numpy as np

from typing import Dict, List

from lm_polygraph.estimators.estimator import Estimator

class EigenScore(Estimator):
    def __init__(
        self,
        alpha:float = 1e-3
    ):
        super().__init__(["sample_embeddings"], "sequence")
        self.alpha = alpha

    def __str__(self):
        return f"EigenScore"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        sample_embeddings = stats[f"sample_embeddings"]
        ue = []
        for embeddings in sample_embeddings:
            sentence_embeddings = np.array(embeddings)
            dim = sentence_embeddings.shape[-1]
            J_d = np.eye(dim) - 1 / dim * np.ones((dim, dim))
            covariance = sentence_embeddings @ J_d @ sentence_embeddings.T
            reg_covariance = covariance + self.alpha * np.eye(covariance.shape[0])
            eigenvalues, _ = np.linalg.eig(reg_covariance)
            ue.append(np.mean(np.log([val if val > 0 else 1e-10 for val in eigenvalues])))
        return np.array(ue)