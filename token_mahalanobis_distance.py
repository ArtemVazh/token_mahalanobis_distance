import os
import numpy as np
import torch

from typing import Dict

from lm_polygraph.estimators.estimator import Estimator
from lm_polygraph.generation_metrics.aggregated_metric import AggregatedMetric

from lm_polygraph.estimators.mahalanobis_distance import (
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
        aggregation: str = "mean",
        hidden_layer: int = -1,
        metric = None,
        metric_name: str = "",
        aggregated: bool = False,
    ):
        self.hidden_layer = hidden_layer
        if self.hidden_layer == -1:
            super().__init__(["token_embeddings", "train_token_embeddings", "train_greedy_tokens", "train_target_texts"], "sequence")
        else:
            super().__init__([f"token_embeddings_{self.hidden_layer}", f"train_token_embeddings_{self.hidden_layer}", "train_greedy_tokens", "train_target_texts"], "sequence")
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
        self.metric_name = metric_name
        if metric is not None:
            self.metric = metric
            if aggregated:
                self.metric = AggregatedMetric(base_metric=self.metric)

        if self.parameters_path is not None:
            self.full_path = f"{self.parameters_path}/tmd_{self.hidden_layer}_{self.embeddings_type}_{self.aggregation}_{self.metric_name}_{self.metric_thr}"
            os.makedirs(self.full_path, exist_ok=True)

            if os.path.exists(f"{self.full_path}/centroid.pt"):
                self.centroid = torch.load(f"{self.full_path}/centroid.pt")
                self.sigma_inv = torch.load(f"{self.full_path}/sigma_inv.pt")
                self.max = torch.load(f"{self.full_path}/max.pt")
                self.min = torch.load(f"{self.full_path}/min.pt")
                self.is_fitted = True

    def __str__(self):
        hidden_layer = "" if self.hidden_layer==-1 else f"_{self.hidden_layer}"
        return f"TokenMahalanobisDistance_{self.embeddings_type}{hidden_layer} ({self.aggregation}, {self.metric_name}, {self.metric_thr})"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        # take the embeddings
        if self.hidden_layer == -1:
            hidden_layer = ""
        else:
            hidden_layer = f"_{self.hidden_layer}"
        embeddings = create_cuda_tensor_from_numpy(
            stats[f"token_embeddings_{self.embeddings_type}{hidden_layer}"]
        )

        # compute centroids if not given
        if not self.is_fitted:
            train_embeddings = create_cuda_tensor_from_numpy(
                stats[f"train_token_embeddings_{self.embeddings_type}{hidden_layer}"]
            )
            if self.metric_thr > 0:
                train_greedy_texts = stats[f"train_greedy_texts"]
                train_greedy_tokens = stats[f"train_greedy_tokens"]
                train_target_texts = stats[f"train_target_texts"]
                self.train_token_metrics = np.concatenate([[self.metric({"greedy_texts": [x], "target_texts": [y]}, [y], [y])[0]] * len(x_t) 
                                                           for x, y, x_t in zip(train_greedy_texts, train_target_texts, train_greedy_tokens)])
                
                if (self.train_token_metrics >= self.metric_thr).sum() > 10:
                    train_embeddings = train_embeddings[self.train_token_metrics >= self.metric_thr]

            self.centroid = train_embeddings.mean(axis=0)
            if self.parameters_path is not None:
                torch.save(self.centroid, f"{self.full_path}/centroid.pt")

        # compute inverse covariance matrix if not given
        if not self.is_fitted:
            train_embeddings = create_cuda_tensor_from_numpy(
                stats[f"train_token_embeddings_{self.embeddings_type}{hidden_layer}"]
            )

            if self.metric_thr > 0:
                if (self.train_token_metrics >= self.metric_thr).sum() > 10:
                    train_embeddings = train_embeddings[self.train_token_metrics >= self.metric_thr]
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
        if self.aggregation == "none":
            agg_dists = dists.cpu().detach().numpy()
        else:
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
