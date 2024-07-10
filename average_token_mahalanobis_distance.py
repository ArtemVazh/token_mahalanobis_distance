import os
import numpy as np
import torch

from typing import Dict, List

from lm_polygraph.estimators.estimator import Estimator
from lm_polygraph.generation_metrics.aggregated_metric import AggregatedMetric

from lm_polygraph.estimators.mahalanobis_distance import (
    compute_inv_covariance,
    mahalanobis_distance_with_known_centroids_sigma_inv,
    MahalanobisDistanceSeq,
    create_cuda_tensor_from_numpy,
    JITTERS
)

from lm_polygraph.generation_metrics.openai_fact_check import OpenAIFactCheck
from token_mahalanobis_distance import TokenMahalanobisDistance



class AverageTokenMahalanobisDistance(Estimator):
    def __init__(
        self,
        embeddings_type: str = "decoder",
        parameters_path: str = None,
        normalize: bool = False,
        metric_thr: float = 0.0,
        aggregation: str = "mean",
        hidden_layers: List[int] = [0, -1],
        metric = None,
        metric_name: str = "",
        aggregated: bool = False,
    ):
        self.hidden_layers = hidden_layers
        self.tmds = []
        dependencies = ["train_greedy_tokens", "train_target_texts"]
        for layer in self.hidden_layers:
            if layer == -1:
                dependencies += ["token_embeddings", "train_token_embeddings"]
            else:
                dependencies += [f"token_embeddings_{layer}", f"train_token_embeddings_{layer}"]

            self.tmds.append(TokenMahalanobisDistance(
                embeddings_type, parameters_path, normalize=False, metric_thr=metric_thr, metric=metric, metric_name=metric_name, aggregation="none", hidden_layer=layer, aggregated=aggregated
            ))
        
        super().__init__(dependencies, "sequence")
        self.is_fitted = False
        self.metric_thr = metric_thr
        self.aggregation = aggregation
        self.metric_name = metric_name
        self.embeddings_type=embeddings_type
    
    def __str__(self):
        hidden_layers = ",".join([str(x) for x in self.hidden_layers])
        return f"AverageTokenMahalanobisDistance_{self.embeddings_type}{hidden_layers} ({self.aggregation}, {self.metric_name}, {self.metric_thr})"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        # take the embeddings
        mds = []
        for MD in self.tmds:
            md = MD(stats)
            mds.append(md)
        dists = np.mean(mds, axis=0)
        
        k = 0
        agg_dists = []
        for tokens in stats["greedy_tokens"]:
            dists_i = dists[k:k+len(tokens)]
            k += len(tokens)
            if self.aggregation == "mean":
                agg_dists.append(np.mean(dists_i))
            elif self.aggregation == "sum":
                agg_dists.append(np.sum(dists_i))

        return agg_dists
