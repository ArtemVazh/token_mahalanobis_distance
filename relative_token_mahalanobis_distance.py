import os
import numpy as np
import torch

from typing import Dict

from lm_polygraph.estimators.estimator import Estimator
from lm_polygraph.estimators.mahalanobis_distance import (
    compute_inv_covariance,
    mahalanobis_distance_with_known_centroids_sigma_inv,
    MahalanobisDistanceSeq,
    create_cuda_tensor_from_numpy,
)

from token_mahalanobis_distance import TokenMahalanobisDistance, TokenMahalanobisDistanceClaim

def save_array(array, filename):
    with open(filename, "wb") as f:
        np.save(f, array)


def load_array(filename):
    with open(filename, "rb") as f:
        array = np.load(f)
    return array


NAMING_MAP = {"bert-base-uncased": "bert_base", 
              "bert-large-uncased": "bert_large", 
              "google/electra-small-discriminator": "electra_base", 
              "roberta-base": "roberta_base", 
              "roberta-large": "roberta_large",
              "meta-llama/Llama-3.2-1B": "llama1b", 
              "meta-llama/Llama-3.2-3B": "llama3b", 
              "meta-llama/Llama-3.1-8B": "llama8b"}


class RelativeTokenMahalanobisDistance(Estimator):
    """
    Ren et al. (2023) showed that it might be useful to adjust the Mahalanobis distance score by subtracting
    from it the other Mahalanobis distance MD_0(x) computed for some large general purpose dataset covering many domain.
    RMD(x) = MD(x) - MD_0(x)
    """

    def __init__(
        self,
        embeddings_type: str = "decoder",
        parameters_path: str = None,
        normalize: bool = False,
        metric_thr: float = 0.0,
        aggregation: str = "mean",
        metric = None,
        aggregated: bool = False,
        metric_name: str = "",
        hidden_layer: int = -1,
        device: str = "cuda",
        storage_device: str = "cuda",
        is_proxy_model: bool = False,
        proxy_model_name: str = "bert-base-uncased",
    ):
        self.hidden_layer = hidden_layer
        self.is_proxy_model = is_proxy_model
        self.proxy = f"proxy_{NAMING_MAP[proxy_model_name]}_" if self.is_proxy_model else ""
        train_greedy_tokens = f"train_{self.proxy}tokens" if self.is_proxy_model else f"train_greedy_tokens"
        if self.hidden_layer == -1:
            super().__init__([f"{self.proxy}token_embeddings", f"train_{self.proxy}token_embeddings", f"background_train_{self.proxy}token_embeddings", train_greedy_tokens, "train_target_texts"], "sequence")
        else:
            super().__init__([f"{self.proxy}token_embeddings_{self.hidden_layer}", f"train_{self.proxy}token_embeddings_{self.hidden_layer}", f"background_train_{self.proxy}token_embeddings_{self.hidden_layer}", train_greedy_tokens, "train_target_texts"], "sequence")
        self.centroid_0 = None
        self.sigma_inv_0 = None
        self.parameters_path = parameters_path
        self.embeddings_type = embeddings_type
        self.normalize = normalize
        self.min = 1e100
        self.max = -1e100
        self.metric_name = metric_name
        self.MD = TokenMahalanobisDistance(
            embeddings_type, parameters_path, normalize=False, metric_thr=metric_thr, metric=metric, metric_name=metric_name, aggregation="none", hidden_layer=self.hidden_layer, aggregated=aggregated, device=device, storage_device=storage_device, is_proxy_model=is_proxy_model, proxy_model_name=proxy_model_name
        )
        self.is_fitted = False
        self.metric_thr = metric_thr
        self.aggregation = aggregation
        self.metric = metric
        self.device = device
        self.storage_device = storage_device
        
        if self.parameters_path is not None:
            self.full_path = f"{self.parameters_path}/rtmd_{self.hidden_layer}_{self.embeddings_type}_{self.aggregation}_{self.metric_name }_{self.metric_thr}"
            os.makedirs(self.full_path, exist_ok=True)
            if os.path.exists(f"{self.full_path}/centroid_0.pt"):
                self.centroid_0 = torch.load(f"{self.full_path}/centroid_0.pt")
                self.sigma_inv_0 = torch.load(f"{self.full_path}/sigma_inv_0.pt")
                self.max = load_array(f"{self.full_path}/max_0.npy")
                self.min = load_array(f"{self.full_path}/min_0.npy")
                self.is_fitted = True

    def __str__(self):
        hidden_layer = "" if self.hidden_layer==-1 else f"_{self.hidden_layer}"
        return f"RelativeTokenMahalanobisDistance_{self.proxy}{self.embeddings_type}{hidden_layer} ({self.aggregation}, {self.metric_name}, {self.metric_thr})"

    def __call__(self, stats: Dict[str, np.ndarray], save_data: bool = True) -> np.ndarray:
        # take the embeddings
        if self.hidden_layer == -1:
            hidden_layer = ""
        else:
            hidden_layer = f"_{self.hidden_layer}"
        embeddings = create_cuda_tensor_from_numpy(
            stats[f"{self.proxy}token_embeddings_{self.embeddings_type}{hidden_layer}"]
        )

        # since we want to adjust resulting reasure on baseline MD on train part
        # we have to compute average train centroid and inverse cavariance matrix
        # to obtain MD_0

        if not self.is_fitted:
            train_greedy_texts = stats[f"train_greedy_texts"]
            centroid_key = f"background_{self.proxy}centroid{hidden_layer}_{self.metric_name}_{self.metric_thr}_{len(train_greedy_texts)}"
            if (centroid_key in stats.keys()): # to reduce number of stored centroid for multiple methods used the same data
                self.centroid_0 = stats[centroid_key]
                if self.storage_device == "cpu":
                    self.centroid_0 = self.centroid_0.cpu()
                elif self.storage_device == "cuda":
                    self.centroid_0 = self.centroid_0.cuda()
            else:
                background_train_embeddings = create_cuda_tensor_from_numpy(
                    stats[f"background_train_{self.proxy}token_embeddings_{self.embeddings_type}{hidden_layer}"]
                )
                self.centroid_0 = background_train_embeddings.mean(axis=0)
                
                if self.storage_device == "cpu":
                    self.centroid_0 = self.centroid_0.cpu()
                    
                if self.parameters_path is not None:
                    torch.save(self.centroid_0, f"{self.full_path}/centroid_0.pt")
                if save_data:
                    stats[centroid_key] = self.centroid_0

        if not self.is_fitted:
            covariance_key = f"background_{self.proxy}covariance{hidden_layer}_{self.metric_name}_{self.metric_thr}_{len(train_greedy_texts)}"
            if (covariance_key in stats.keys()): # to reduce number of stored centroid for multiple methods used the same data
                self.sigma_inv_0 = stats[covariance_key]
                if self.storage_device == "cpu":
                    self.sigma_inv_0 = self.sigma_inv_0.cpu()
                elif self.storage_device == "cuda":
                    self.sigma_inv_0 = self.sigma_inv_0.cuda()
            else:
                background_train_embeddings = create_cuda_tensor_from_numpy(
                    stats[f"background_train_{self.proxy}token_embeddings_{self.embeddings_type}{hidden_layer}"]
                )
                self.sigma_inv_0, _ = compute_inv_covariance(
                    self.centroid_0.unsqueeze(0), background_train_embeddings
                )
                if self.storage_device == "cpu":
                    self.sigma_inv_0 = self.sigma_inv_0.cpu()
                    
                if self.parameters_path is not None:
                    torch.save(self.sigma_inv_0, f"{self.full_path}/sigma_inv_0.pt")
                if save_data:
                    stats[covariance_key] = self.sigma_inv_0
                
            self.is_fitted = True
            
        # compute MD_0

        if self.device == "cuda" and self.storage_device == "cpu":
            if embeddings.shape[0] < 20:
                # force compute on cpu, since for a small number of embeddings it will be faster than move to cuda 
                dists_0 = (
                    mahalanobis_distance_with_known_centroids_sigma_inv(
                        self.centroid_0.float(),
                        None,
                        self.sigma_inv_0.float(),
                        embeddings.cpu().float(),
                    )[:, 0]
                    .cpu()
                    .detach()
                    .numpy()
                )
            else:
                dists_0 = (
                    mahalanobis_distance_with_known_centroids_sigma_inv(
                        self.centroid_0.cuda().float(),
                        None,
                        self.sigma_inv_0.cuda().float(),
                        embeddings.float(),
                    )[:, 0]
                    .cpu()
                    .detach()
                    .numpy()
                )
        elif self.device == "cuda" and self.storage_device == "cuda":
            dists_0 = (
                    mahalanobis_distance_with_known_centroids_sigma_inv(
                        self.centroid_0.float(),
                        None,
                        self.sigma_inv_0.float(),
                        embeddings.float(),
                    )[:, 0]
                    .cpu()
                    .detach()
                    .numpy()
                )
        else:
            raise NotImplementedError

        # compute original MD

        md = self.MD(stats, save_data=save_data)

        # RMD calculation

        dists = md - dists_0
        
        agg_dists = []
        k = 0
        greedy_tokens = stats[f"{self.proxy}tokens"] if self.is_proxy_model else stats[f"greedy_tokens"]
        for tokens in greedy_tokens:
            dists_i = dists[k:k+len(tokens)]
            k += len(tokens)
            if self.aggregation == "mean":
                agg_dists.append(np.mean(dists_i))
            elif self.aggregation == "sum":
                agg_dists.append(np.sum(dists_i))
        if self.aggregation == "none":
            agg_dists = dists
            
        agg_dists = np.array(agg_dists)
        
        if self.max < agg_dists.max():
            self.max = agg_dists.max()
            if self.parameters_path is not None:
                save_array(self.max, f"{self.full_path}/max_0.npy")
        if self.min > agg_dists.min():
            self.min = agg_dists.min()
            if self.parameters_path is not None:
                save_array(self.min, f"{self.full_path}/min_0.npy")

        if self.normalize:
            agg_dists = np.clip(
                (self.max - agg_dists) / (self.max - self.min), a_min=0, a_max=1
            )

        return agg_dists



class RelativeTokenMahalanobisDistanceClaim(Estimator):
    """
    Ren et al. (2023) showed that it might be useful to adjust the Mahalanobis distance score by subtracting
    from it the other Mahalanobis distance MD_0(x) computed for some large general purpose dataset covering many domain.
    RMD(x) = MD(x) - MD_0(x)
    """

    def __init__(
        self,
        embeddings_type: str = "decoder",
        parameters_path: str = None,
        normalize: bool = False,
        metric_thr: float = 0.0,
        aggregation: str = "mean",
        metric = None,
        hidden_layer: int = -1,
        device: str = "cuda",
        storage_device: str = "cuda",
    ):
        self.hidden_layer = hidden_layer            
        if self.hidden_layer == -1:
            super().__init__(["token_embeddings", "train_token_embeddings", "background_train_token_embeddings", "train_greedy_tokens", "train_target_texts"], "claim")
        else:
            super().__init__([f"token_embeddings_{self.hidden_layer}", f"train_token_embeddings_{self.hidden_layer}", f"background_train_token_embeddings_{self.hidden_layer}", "train_greedy_tokens", "train_target_texts"], "claim")
        self.centroid_0 = None
        self.sigma_inv_0 = None
        self.parameters_path = parameters_path
        self.embeddings_type = embeddings_type
        self.normalize = normalize
        self.min = 1e100
        self.max = -1e100
        self.MD = TokenMahalanobisDistanceClaim(
            embeddings_type, parameters_path, normalize=False, metric_thr=metric_thr, aggregation="none", hidden_layer=self.hidden_layer
        )
        self.is_fitted = False
        self.metric_thr = metric_thr
        self.aggregation = aggregation
        self.metric = metric
        self.device = device
        self.storage_device = storage_device

    def __str__(self):
        hidden_layer = "" if self.hidden_layer==-1 else f"_{self.hidden_layer}"
        return f"RelativeTokenMahalanobisDistanceClaim_{self.embeddings_type}{hidden_layer} ({self.aggregation}, {self.metric_thr})"

    def __call__(self, stats: Dict[str, np.ndarray], save_data: bool = True) -> np.ndarray:
        # take the embeddings
        if self.hidden_layer == -1:
            hidden_layer = ""
        else:
            hidden_layer = f"_{self.hidden_layer}"
        embeddings = create_cuda_tensor_from_numpy(
            stats[f"token_embeddings_{self.embeddings_type}{hidden_layer}"]
        )

        # since we want to adjust resulting reasure on baseline MD on train part
        # we have to compute average train centroid and inverse cavariance matrix
        # to obtain MD_0

        if not self.is_fitted:
            train_greedy_texts = stats[f"train_greedy_texts"]
            centroid_key = f"background_centroid{hidden_layer}_{self.metric_thr}_{len(train_greedy_texts)}"
            if (centroid_key in stats.keys()): # to reduce number of stored centroid for multiple methods used the same data
                self.centroid_0 = stats[centroid_key]
            else:
                background_train_embeddings = create_cuda_tensor_from_numpy(
                    stats[f"background_train_token_embeddings_{self.embeddings_type}{hidden_layer}"]
                )
                self.centroid_0 = background_train_embeddings.mean(axis=0)
                if self.storage_device == "cpu":
                    self.centroid_0 = self.centroid_0.cpu()
                if self.parameters_path is not None:
                    torch.save(self.centroid_0, f"{self.full_path}/centroid_0.pt")
                if save_data:
                    stats[centroid_key] = self.centroid_0

        if not self.is_fitted:
            covariance_key = f"background_covariance{hidden_layer}_{self.metric_thr}_{len(train_greedy_texts)}"
            if (covariance_key in stats.keys()): # to reduce number of stored centroid for multiple methods used the same data
                self.sigma_inv_0 = stats[covariance_key]
            else:
                background_train_embeddings = create_cuda_tensor_from_numpy(
                    stats[f"background_train_token_embeddings_{self.embeddings_type}{hidden_layer}"]
                )
                self.sigma_inv_0, _ = compute_inv_covariance(
                    self.centroid_0.unsqueeze(0), background_train_embeddings
                )
                if self.storage_device == "cpu":
                    self.sigma_inv_0 = self.sigma_inv_0.cpu()
                if self.parameters_path is not None:
                    torch.save(self.sigma_inv_0, f"{self.full_path}/sigma_inv_0.pt")

                if save_data:
                    stats[covariance_key] = self.sigma_inv_0
            self.is_fitted = True

        # compute MD_0

        if self.device == "cuda" and self.storage_device == "cpu":
            if embeddings.shape[0] < 20:
                # force compute on cpu, since for a small number of embeddings it will be faster than move to cuda 
                dists_0 = (
                    mahalanobis_distance_with_known_centroids_sigma_inv(
                        self.centroid_0.float(),
                        None,
                        self.sigma_inv_0.float(),
                        embeddings.cpu().float(),
                    )[:, 0]
                    .cpu()
                    .detach()
                    .numpy()
                )
            else:
                dists_0 = (
                    mahalanobis_distance_with_known_centroids_sigma_inv(
                        self.centroid_0.cuda().float(),
                        None,
                        self.sigma_inv_0.cuda().float(),
                        embeddings.float(),
                    )[:, 0]
                    .cpu()
                    .detach()
                    .numpy()
                )
        elif self.device == "cuda" and self.storage_device == "cuda":
            dists_0 = (
                    mahalanobis_distance_with_known_centroids_sigma_inv(
                        self.centroid_0.float(),
                        None,
                        self.sigma_inv_0.float(),
                        embeddings.float(),
                    )[:, 0]
                    .cpu()
                    .detach()
                    .numpy()
                )
        else:
            raise NotImplementedError
        # compute original MD

        md = self.MD(stats, save_data=save_data)

        # RMD calculation

        dists = md - dists_0

        k = 0
        tmd_scores = []
        claims = stats["claims"]
        for idx, tokens in enumerate(stats["greedy_tokens"]):
            dists_i = dists[k:k+len(tokens)]
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
            tmd_scores = np.array(dists)
        else:
            tmd_scores = np.array(tmd_scores)
            
        return tmd_scores
