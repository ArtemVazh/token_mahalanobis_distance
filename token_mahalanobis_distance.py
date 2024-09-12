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

from lm_polygraph.generation_metrics.openai_fact_check import OpenAIFactCheck


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
        device: str = "cuda",
        storage_device: str = "cuda",
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
        self.device = device
        self.storage_device = storage_device
        self.aggregated = aggregated
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

    def __call__(self, stats: Dict[str, np.ndarray], save_data: bool = True) -> np.ndarray:
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
            train_greedy_texts = stats[f"train_greedy_texts"]
            centroid_key = f"centroid{hidden_layer}_{self.metric_name}_{self.metric_thr}_{len(train_greedy_texts)}"
            if (centroid_key in stats.keys()): # to reduce number of stored centroid for multiple methods used the same data
                self.centroid = stats[centroid_key]
            else:
                train_embeddings = create_cuda_tensor_from_numpy(
                    stats[f"train_token_embeddings_{self.embeddings_type}{hidden_layer}"]
                )
                if self.metric_thr > 0:
                    train_greedy_tokens = stats[f"train_greedy_tokens"]
                    train_target_texts = stats[f"train_target_texts"]
                    
                    metric_key = f"train_{self.metric_name}_{len(train_greedy_texts)}"
                    if metric_key in stats.keys():
                        self.train_token_metrics = stats[metric_key]
                    else:
                        metrics = []
                        for x, y, x_t in zip(train_greedy_texts, train_target_texts, train_greedy_tokens):
                            if isinstance(y, list) and (not self.aggregated):
                                y_ = y[0]
                            elif isinstance(y, str) and (self.aggregated):
                                y_ = [y]
                            else:
                                y_ = y
                            metrics.append([self.metric({"greedy_texts": [x], "target_texts": [y_]}, [y_], [y_])[0]] * len(x_t))
                            
                        self.train_token_metrics = np.concatenate(metrics)
                        stats[metric_key] = self.train_token_metrics
                        
                    if (self.train_token_metrics >= self.metric_thr).sum() > 10:
                        train_embeddings = train_embeddings[self.train_token_metrics >= self.metric_thr]

                self.centroid = train_embeddings.mean(axis=0)
                
                if self.storage_device == "cpu":
                    self.centroid = self.centroid.cpu()
                if self.parameters_path is not None:
                    torch.save(self.centroid, f"{self.full_path}/centroid.pt")
                if save_data:
                    stats[centroid_key] = self.centroid

        # compute inverse covariance matrix if not given
        if not self.is_fitted:
            covariance_key = f"covariance{hidden_layer}_{self.metric_name}_{self.metric_thr}_{len(train_greedy_texts)}"
            if (covariance_key in stats.keys()): # to reduce number of stored centroid for multiple methods used the same data
                self.sigma_inv = stats[covariance_key]
            else:
                train_embeddings = create_cuda_tensor_from_numpy(
                    stats[f"train_token_embeddings_{self.embeddings_type}{hidden_layer}"]
                )

                if self.metric_thr > 0:
                    if (self.train_token_metrics >= self.metric_thr).sum() > 10:
                        train_embeddings = train_embeddings[self.train_token_metrics >= self.metric_thr]
                self.sigma_inv, _ = compute_inv_covariance(
                    self.centroid.unsqueeze(0), train_embeddings
                )
                if self.storage_device == "cpu":
                    self.sigma_inv = self.sigma_inv.cpu()
                    
                if self.parameters_path is not None:
                    torch.save(self.sigma_inv, f"{self.full_path}/sigma_inv.pt")
                
                if save_data:
                    stats[covariance_key] = self.sigma_inv
            self.is_fitted = True

        # compute MD given centroids and inverse covariance matrix
        if self.device == "cuda" and self.storage_device == "cpu":
            if embeddings.shape[0] < 20:
                # force compute on cpu, since for a small number of embeddings it will be faster than move to cuda 
                dists = mahalanobis_distance_with_known_centroids_sigma_inv(
                    self.centroid.float(),
                    None,
                    self.sigma_inv.float(),
                    embeddings.cpu().float(),
                )[:, 0]
            else:
                dists = mahalanobis_distance_with_known_centroids_sigma_inv(
                    self.centroid.cuda().float(),
                    None,
                    self.sigma_inv.cuda().float(),
                    embeddings.float(),
                )[:, 0]
        elif self.device == "cuda" and self.storage_device == "cuda":
            dists = mahalanobis_distance_with_known_centroids_sigma_inv(
                self.centroid.float(),
                None,
                self.sigma_inv.float(),
                embeddings.float(),
            )[:, 0]
        else:
            raise NotImplementedError
        
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


class TokenMahalanobisDistanceClaim(Estimator):
    def __init__(
        self,
        embeddings_type: str = "decoder",
        parameters_path: str = None,
        normalize: bool = False,
        metric_thr: float = 0.0,
        aggregation: str = "mean",
        hidden_layer: int = -1,
        device: str = "cuda",
        storage_device: str = "cuda",
    ):
        self.hidden_layer = hidden_layer
        if self.hidden_layer == -1:
            super().__init__(["token_embeddings", "train_token_embeddings", "train_greedy_tokens", "train_target_texts", "claims", "train_claims", "train_input_texts"], "claim")
        else:
            super().__init__([f"token_embeddings_{self.hidden_layer}", f"train_token_embeddings_{self.hidden_layer}", "train_greedy_tokens", "train_target_texts", "claims", "train_claims", "train_input_texts"], "claim")
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
        self.device = device
        self.storage_device = storage_device
        self.factcheck = OpenAIFactCheck(openai_model="gpt-4o")
        
        # if self.parameters_path is not None:
        #     self.full_path = f"{self.parameters_path}/tmd_{self.hidden_layer}_{self.embeddings_type}_{self.aggregation}_{self.metric_name}_{self.metric_thr}"
        #     os.makedirs(self.full_path, exist_ok=True)

        #     if os.path.exists(f"{self.full_path}/centroid.pt"):
        #         self.centroid = torch.load(f"{self.full_path}/centroid.pt")
        #         self.sigma_inv = torch.load(f"{self.full_path}/sigma_inv.pt")
        #         self.max = torch.load(f"{self.full_path}/max.pt")
        #         self.min = torch.load(f"{self.full_path}/min.pt")
        #         self.is_fitted = True

    def __str__(self):
        hidden_layer = "" if self.hidden_layer==-1 else f"_{self.hidden_layer}"
        return f"TokenMahalanobisDistanceClaim_{self.embeddings_type}{hidden_layer} ({self.aggregation}, {self.metric_thr})"

    def _get_targets(self, greedy_tokens, claims, factcheck):
        targets = []
        for j in range(len(greedy_tokens)):
            target = np.zeros_like(greedy_tokens[j]) + 1.0
            true_tokens = []
            false_tokens = []
            for i, claim in enumerate(claims[j]):
                if not np.isnan(factcheck[j][i]):
                    for t in claim.aligned_token_ids:
                         if factcheck[j][i] == 1:
                             false_tokens.append(t)
                         else:
                             true_tokens.append(t)
            final_true_tokens = np.array(list(set(true_tokens) - set(false_tokens)))
            final_false_tokens = np.array(list(set(false_tokens) - set(true_tokens)))
            if len(final_true_tokens):
                target[final_true_tokens] = 1.0
            if len(final_false_tokens):
                target[final_false_tokens] = 0.0
            target = np.clip(target, 0, 1)
            targets.append(target)
        return targets

    def __call__(self, stats: Dict[str, np.ndarray], save_data=True) -> np.ndarray:
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
            train_greedy_texts = stats[f"train_greedy_texts"]
            centroid_key = f"centroid{hidden_layer}_{self.metric_thr}_{len(train_greedy_texts)}"
            if (centroid_key in stats.keys()): # to reduce number of stored centroid for multiple methods used the same data
                self.centroid = stats[centroid_key]
            else:
                train_embeddings = create_cuda_tensor_from_numpy(
                    stats[f"train_token_embeddings_{self.embeddings_type}{hidden_layer}"]
                )
                if self.metric_thr > 0:
                    train_greedy_texts = stats[f"train_greedy_texts"]
                    train_greedy_tokens = stats[f"train_greedy_tokens"]
                    train_input_texts = stats[f"train_input_texts"]
                    train_claims = stats[f"train_claims"]
                    train_stats = {"claims": train_claims, "input_texts": train_input_texts}
    
                    if "factcheck" in stats.keys():
                        factcheck = stats["factcheck"]
                        self.train_token_metrics = stats["train_token_metrics"]
                    else:
                        factcheck = self.factcheck(train_stats, None, None)
                        self.train_token_metrics = np.concatenate(self._get_targets(train_greedy_tokens, train_claims, factcheck))
                        stats["factcheck"] = factcheck
                        stats["train_token_metrics"] = self.train_token_metrics
                                
                    if (self.train_token_metrics >= self.metric_thr).sum() > 10:
                        train_embeddings = train_embeddings[self.train_token_metrics <= self.metric_thr]
    
                self.centroid = train_embeddings.mean(axis=0)
                if self.storage_device == "cpu":
                    self.centroid = self.centroid.cpu()
                if self.parameters_path is not None:
                    torch.save(self.centroid, f"{self.full_path}/centroid.pt")
                if save_data:
                    stats[centroid_key] = self.centroid

        # compute inverse covariance matrix if not given
        if not self.is_fitted:
            covariance_key = f"covariance{hidden_layer}_{self.metric_thr}_{len(train_greedy_texts)}"
            if (covariance_key in stats.keys()): # to reduce number of stored centroid for multiple methods used the same data
                self.sigma_inv = stats[covariance_key]
            else:
                train_embeddings = create_cuda_tensor_from_numpy(
                    stats[f"train_token_embeddings_{self.embeddings_type}{hidden_layer}"]
                )
    
                if self.metric_thr > 0:
                    if (self.train_token_metrics >= self.metric_thr).sum() > 10:
                        train_embeddings = train_embeddings[self.train_token_metrics >= self.metric_thr]
                self.sigma_inv, _ = compute_inv_covariance(
                    self.centroid.unsqueeze(0), train_embeddings
                )
                if self.storage_device == "cpu":
                    self.sigma_inv = self.sigma_inv.cpu()
                if self.parameters_path is not None:
                    torch.save(self.sigma_inv, f"{self.full_path}/sigma_inv.pt")
                    
                if save_data:
                    stats[covariance_key] = self.sigma_inv
            self.is_fitted = True


        # compute MD given centroids and inverse covariance matrix
        if self.device == "cuda" and self.storage_device == "cpu":
            if embeddings.shape[0] < 20:
                # force compute on cpu, since for a small number of embeddings it will be faster than move to cuda 
                dists = mahalanobis_distance_with_known_centroids_sigma_inv(
                    self.centroid.float(),
                    None,
                    self.sigma_inv.float(),
                    embeddings.cpu().float(),
                )[:, 0]
            else:
                dists = mahalanobis_distance_with_known_centroids_sigma_inv(
                    self.centroid.cuda().float(),
                    None,
                    self.sigma_inv.cuda().float(),
                    embeddings.float(),
                )[:, 0]
        elif self.device == "cuda" and self.storage_device == "cuda":
            dists = mahalanobis_distance_with_known_centroids_sigma_inv(
                self.centroid.float(),
                None,
                self.sigma_inv.float(),
                embeddings.float(),
            )[:, 0]
        else:
            raise NotImplementedError

        k = 0
        tmd_scores = []
        claims = stats["claims"]
        for idx, tokens in enumerate(stats["greedy_tokens"]):
            dists_i = dists[k:k+len(tokens)].cpu().detach().numpy()
            k += len(tokens)

            tmd_scores.append([])
            for claim in claims[idx]:
                tokens = np.array(claim.aligned_token_ids)
                claim_p_i = dists_i[tokens]
                
                if self.aggregation == "mean":
                    tmd_scores[-1].append(claim_p_i.mean())
                elif self.aggregation  == "sum":
                    tmd_scores[-1].append(claim_p_i.sum())
        if self.aggregation == "none":
            tmd_scores = dists.cpu().detach().numpy()
        else:
            tmd_scores = np.array(tmd_scores)

        return tmd_scores
