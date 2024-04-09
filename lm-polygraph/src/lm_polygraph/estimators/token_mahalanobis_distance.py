import os
import numpy as np
import torch

from typing import Dict

from .estimator import Estimator

from .mahalanobis_distance import (
    compute_inv_covariance,
    mahalanobis_distance_with_known_centroids_sigma_inv,
    MahalanobisDistanceSeq,
    create_cuda_tensor_from_numpy,
    JITTERS
)

class TokenMahalanobisDistance(Estimator):
    def __init__(
        self,
        embeddings_type: str = "decoder",
        parameters_path: str = None,
        normalize: bool = False,
        metric_thr: float = 0.0,
        aggregation: str = "mean"
    ):
        super().__init__(["token_embeddings", "train_token_embeddings", "train_token_metrics"], "sequence")
        self.centroid = None
        self.sigma_inv = None
        self.parameters_path = parameters_path
        self.embeddings_type = embeddings_type
        self.normalize = normalize
        self.min = 1e100
        self.max = -1e100
        self.is_fitted = False
        self.metric_thr = metric_thr
        self.aggregation = aggregation

        if self.parameters_path is not None:
            self.full_path = f"{self.parameters_path}/tmd_{self.embeddings_type}_{self.aggregation}_{self.metric_thr}"
            os.makedirs(self.full_path, exist_ok=True)

            if os.path.exists(f"{self.full_path}/centroid.pt"):
                self.centroid = torch.load(f"{self.full_path}/centroid.pt")
                self.sigma_inv = torch.load(f"{self.full_path}/sigma_inv.pt")
                self.max = torch.load(f"{self.full_path}/max.pt")
                self.min = torch.load(f"{self.full_path}/min.pt")
                self.is_fitted = True

    def __str__(self):
        return f"TokenMahalanobisDistance_{self.embeddings_type} ({self.aggregation}, {self.metric_thr})"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        # take the embeddings
        embeddings = create_cuda_tensor_from_numpy(
            stats[f"token_embeddings_{self.embeddings_type}"]
        )

        # compute centroids if not given
        if not self.is_fitted:
            train_embeddings = create_cuda_tensor_from_numpy(
                stats[f"train_token_embeddings_{self.embeddings_type}"]
            )
            train_token_metrics = np.array(stats["train_token_metrics"]).flatten()
            train_embeddings = train_embeddings[train_token_metrics > self.metric_thr]
            
            self.centroid = train_embeddings.mean(axis=0)
            if self.parameters_path is not None:
                torch.save(self.centroid, f"{self.full_path}/centroid.pt")

        # compute inverse covariance matrix if not given
        if not self.is_fitted:
            train_embeddings = create_cuda_tensor_from_numpy(
                stats[f"train_token_embeddings_{self.embeddings_type}"]
            )
            train_token_metrics = np.array(stats["train_token_metrics"]).flatten()
            train_embeddings = train_embeddings[train_token_metrics > self.metric_thr]
            self.sigma_inv, _ = compute_inv_covariance(
                self.centroid.unsqueeze(0), train_embeddings
            )
            if self.parameters_path is not None:
                torch.save(self.sigma_inv, f"{self.full_path}/sigma_inv.pt")
            self.is_fitted = True

        if torch.cuda.is_available():
            if not self.centroid.is_cuda:
                self.centroid = self.centroid.cuda()
            if not self.sigma_inv.is_cuda:
                self.sigma_inv = self.sigma_inv.cuda()

        # compute MD given centroids and inverse covariance matrix
        dists = mahalanobis_distance_with_known_centroids_sigma_inv(
            self.centroid.float(),
            None,
            self.sigma_inv.float(),
            embeddings.float(),
        )[:, 0]
        
        k = 0
        agg_dists = []
        for tokens in stats["greedy_tokens"]:
            dists_i = dists[k:k+len(tokens)].cpu().detach().numpy()
            k += len(tokens)
            if self.aggregation == "mean":
                agg_dists.append(np.mean(dists_i))
            elif self.aggregation == "sum":
                agg_dists.append(np.sum(dists_i))
        agg_dists = np.array(agg_dists)
    
        if self.max < agg_dists.max():
            self.max = agg_dists.max()
            if self.parameters_path is not None:
                torch.save(self.max, f"{self.full_path}/max.pt")
        if self.min > agg_dists.min():
            self.min = agg_dists.min()
            if self.parameters_path is not None:
                torch.save(self.min, f"{self.full_path}/min.pt")

        # norlmalise if required
        if self.normalize:
            agg_dists = torch.clip((self.max - agg_dists) / (self.max - self.min), min=0, max=1)

        return agg_dists
