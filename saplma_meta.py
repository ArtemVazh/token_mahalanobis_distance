import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import itertools
from sklearn.model_selection import KFold

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
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import train_test_split
from saplma import SAPLMA

class SAPLMA_meta(Estimator):
    def __init__(
        self,
        embeddings_type: str = "decoder",
        parameters_path: str = None,
        normalize: bool = False,
        aggregation: str = "mean",
        hidden_layer: int = -1,
        metric = None,
        metric_name: str = "",
        aggregated: bool = False,
        device: str = "cuda",
        cv_hp: bool = False
    ):
        self.hidden_layers = hidden_layer
        self.saplmas = []
        dependencies = ["train_greedy_tokens", "train_target_texts"]
        for layer in self.hidden_layers:
            if layer == -1:
                dependencies += ["token_embeddings", "train_token_embeddings"]
            else:
                dependencies += [f"token_embeddings_{layer}", f"train_token_embeddings_{layer}"]

            self.saplmas.append(SAPLMA(embeddings_type, parameters_path=parameters_path, metric=metric, metric_name=metric_name, 
                                       aggregated=aggregated, hidden_layer=layer, device=device, cv_hp=cv_hp))
        super().__init__(dependencies, "sequence")
        self.centroid = None
        self.sigma_inv = None
        self.parameters_path = parameters_path
        self.embeddings_type = embeddings_type
        self.normalize = normalize
        self.min = 1e100
        self.max = -1e100
        self.is_fitted = False
        self.aggregation = aggregation
        self.metric_name = metric_name
        self.device = device
        self.cv_hp = cv_hp
        self.regression = True if metric_name!="Accuracy" else False
        self.ue_predictor = Ridge()
        self.aggregated = aggregated
        if metric is not None:
            self.metric = metric
            if aggregated:
                self.metric = AggregatedMetric(base_metric=self.metric)

    def __str__(self):
        hidden_layers = ",".join([str(x) for x in self.hidden_layers])
        cv = "cv, " if self.cv_hp else ""
        return f"SAPLMA_meta_{self.embeddings_type}{hidden_layers} ({cv}{self.metric_name})"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
       
        # compute centroids if not given
        if not self.is_fitted:
            train_greedy_texts = stats[f"train_greedy_texts"]
            train_greedy_tokens = stats[f"train_greedy_tokens"]
            train_target_texts = stats[f"train_target_texts"]
            metric_key = f"train_seq_{self.metric_name}_{len(train_greedy_texts)}"
            if metric_key in stats.keys():
                self.train_seq_metrics = stats[metric_key]
            else:   
                metrics = []
                for x, y, x_t in zip(train_greedy_texts, train_target_texts, train_greedy_tokens):
                    if isinstance(y, list) and (not self.aggregated):
                        y_ = y[0]
                    elif isinstance(y, str) and (self.aggregated):
                        y_ = [y]
                    else:
                        y_ = y
                    metrics.append(self.metric({"greedy_texts": [x], "target_texts": [y_]}, [y_])[0])
                self.train_seq_metrics = np.array(metrics)
                stats[metric_key] = self.train_seq_metrics
            self.train_seq_metrics[np.isnan(self.train_seq_metrics)] = 0
            train_saplmas = []
            
            dev_size = 0.5 
            train_idx, dev_idx = train_test_split(list(range(len(train_greedy_texts))), test_size=dev_size, random_state=42)
                
            for layer in self.hidden_layers:                        
                if layer == -1:
                    train_embeddings = stats[f"train_embeddings_{self.embeddings_type}"]
                    train_stats = {"train_greedy_tokens": [train_greedy_tokens[k] for k in train_idx], 
                                   "train_greedy_texts": [train_greedy_texts[k] for k in train_idx],
                                   "train_target_texts": [train_target_texts[k] for k in train_idx],
                                   f"train_embeddings_{self.embeddings_type}": [train_embeddings[k] for k in train_idx],
                                   f"embeddings_{self.embeddings_type}": [train_embeddings[k] for k in dev_idx],
                                  }                
                else:
                    train_embeddings = stats[f"train_embeddings_{self.embeddings_type}_{layer}"]
                    train_stats = {"train_greedy_tokens": [train_greedy_tokens[k] for k in train_idx], 
                                   "train_greedy_texts": [train_greedy_texts[k] for k in train_idx],
                                   "train_target_texts": [train_target_texts[k] for k in train_idx],
                                   f"train_embeddings_{self.embeddings_type}_{layer}": [train_embeddings[k] for k in train_idx],
                                   f"embeddings_{self.embeddings_type}_{layer}": [train_embeddings[k] for k in dev_idx],
                                  }
                score = self.saplmas[layer](train_stats).reshape(-1)
                self.saplmas[layer].is_fitted = False
                train_saplmas.append(score)
            train_scores = np.array(train_saplmas).T
            self.ue_predictor.fit(train_scores, 1 - self.train_seq_metrics[dev_idx])
            self.is_fitted = True

        eval_scores = []
        for layer in self.hidden_layers:
            score = self.saplmas[layer](stats).reshape(-1)
            eval_scores.append(score)
        eval_scores = np.array(eval_scores).T
        eval_scores[np.isnan(eval_scores)] = 0
        ue = self.ue_predictor.predict(eval_scores)

        return ue