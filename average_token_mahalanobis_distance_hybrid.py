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

from lm_polygraph.estimators.max_probability import MaximumSequenceProbability
from lm_polygraph.stat_calculators.entropy import EntropyCalculator

from lm_polygraph.estimators.claim.max_probability import MaximumClaimProbability
from lm_polygraph.estimators.claim.token_entropy import MaxTokenEntropyClaim

from lm_polygraph.generation_metrics.openai_fact_check import OpenAIFactCheck

from lm_polygraph.generation_metrics.openai_fact_check import OpenAIFactCheck
from token_mahalanobis_distance import TokenMahalanobisDistance, TokenMahalanobisDistanceClaim
from relative_token_mahalanobis_distance import RelativeTokenMahalanobisDistance, RelativeTokenMahalanobisDistanceClaim
from average_token_mahalanobis_distance import StableLinReg

from sklearn.linear_model import Ridge, RidgeCV, Lasso
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from lm_polygraph.ue_metrics.ue_metric import get_random_scores
from lm_polygraph.ue_metrics.pred_rej_area import PredictionRejectionArea
from lm_polygraph.estimators.claim.claim_conditioned_probability import ClaimConditionedProbabilityClaim

import scipy
import scipy.cluster.hierarchy as sch
from sklearn.decomposition import PCA

prr = PredictionRejectionArea()

NAMING_MAP = {"bert-base-uncased": "bert_base", 
              "bert-large-uncased": "bert_large", 
              "google/electra-small-discriminator": "electra_base", 
              "roberta-base": "roberta_base", 
              "roberta-large": "roberta_large",
              "meta-llama/Llama-3.2-1B": "llama1b", 
              "meta-llama/Llama-3.2-3B": "llama3b", 
              "meta-llama/Llama-3.1-8B": "llama8b"}

def get_prr(ue, metric):
    mean_val = prr(ue, metric) 
    oracle = prr(-metric, metric)
    random = get_random_scores(prr, metric)
    final_score = (mean_val - random) / (oracle - random)
    return final_score

class MLP_NN(nn.Module):
    def __init__(self,
                 n_features: int = 1603, 
                 n_dim: int = 512, 
                 n_layers: int = 4, 
                 dropout: float = 0.1,):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(n_features, n_dim)] + [nn.Linear(n_dim, n_dim) for i in range(n_layers - 2)] + [nn.Linear(n_dim, 1)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class MLP:
    def __init__(self, 
                 n_features: int = 1603, 
                 n_dim: int = 512, 
                 n_layers: int = 4, 
                 dropout: float = 0.1,
                 n_epochs: int = 20,
                 lr: float = 1e-5,
                 batch_size: int = 256):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.loss = nn.MarginRankingLoss(margin=0.1)
        self.mse = nn.MSELoss() 
        self.model = MLP_NN(n_features, n_dim, n_layers, dropout)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def fit(self, X, y):
        X_torch = torch.tensor(X, dtype=torch.float32)
        y_torch = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
        batch_start = torch.arange(0, len(X), self.batch_size)
        self.model.to(self.device)
        for epoch in tqdm(range(self.n_epochs)):
            self.model.train()
            for start in batch_start:
                X_batch = X_torch[start:start+self.batch_size].to(self.device)
                y_batch = y_torch[start:start+self.batch_size].to(self.device)
                y_pred = self.model(X_batch)
                x1, x2, target = [], [], []
                
                for i in range(len(y_pred)):
                    for j in range(i+1, len(y_pred)):
                        if y_batch[i] > y_batch[j]:
                            target.append(1)
                        elif y_batch[i] < y_batch[j]:
                            target.append(-1)
                        else:
                            continue
                        x1.append(y_pred[i])
                        x2.append(y_pred[j])

                x1 = torch.stack(x1).reshape(-1).to(self.device)
                x2 = torch.stack(x2).reshape(-1).to(self.device)
                target = torch.Tensor(target).to(self.device)
                
                loss = self.loss(x1, x2, target) + self.mse(y_pred, y_batch)       
                #loss = self.loss(y_pred, y_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
    def predict(self, X):
        X_torch = torch.tensor(X, dtype=torch.float32)
        batch_start = torch.arange(0, len(X), self.batch_size)
        self.model.eval()
        prediction = []
        if next(self.model.parameters()).device.type != self.device.type:
            self.model.to(self.device)
        for start in batch_start:
            X_batch = X_torch[start:start+self.batch_size].to(self.device)
            y_pred = self.model(X_batch)
            prediction.append(y_pred.cpu().detach().flatten())
        prediction = np.concatenate(prediction)
        return prediction

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
        train_greedy_tokens = f"train_{self.proxy}tokens" if self.is_proxy_model else f"train_greedy_tokens"
        dependencies = [train_greedy_tokens, "train_target_texts"]
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


class LinRegTokenMahalanobisDistance_Hybrid(Estimator):
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

        metric_md = None,
        metric_md_name: str = "",
        
        aggregated: bool = False,
        positive: bool = True,
        ue: str = "TokenMahalanobis",

        meta_model: str = "LinReg",
        norm: str = "norm",

        tgt_norm: bool = False,
        remove_corr: bool = False,
        remove_alg: int = 2,  

        use_tad: bool = False,

        device: str = "cuda",
        storage_device: str = "cuda",
        
        is_proxy_model: bool = False,
        proxy_model_name: str = "bert-base-uncased",

        sim_pca: bool = False,
    ):
        self.ue = ue
        self.hidden_layers = hidden_layers
        self.device = device
        self.storage_device = storage_device
        self.tmds = {}
        self.is_proxy_model = is_proxy_model
        self.proxy = f"proxy_{NAMING_MAP[proxy_model_name]}_" if self.is_proxy_model else ""

        self.sim_pca = sim_pca
        self.sim_pca_name = f", sim_pca" if self.sim_pca else ""

        train_greedy_tokens = f"train_{self.proxy}tokens" if self.is_proxy_model else f"train_greedy_tokens"
        dependencies = [train_greedy_tokens, "train_target_texts"]
        # dependencies += ["attention_features", "train_attention_features", "train_greedy_log_likelihoods"]

        for layer in self.hidden_layers:
            if layer == -1:
                dependencies += [f"{self.proxy}token_embeddings", f"train_{self.proxy}token_embeddings"]
                if "relative" in ue.lower():
                    dependencies += [f"background_train_{self.proxy}token_embeddings", f"background_train_{self.proxy}token_embeddings"]
            else:
                dependencies += [f"{self.proxy}token_embeddings_{layer}", f"train_{self.proxy}token_embeddings_{layer}"]
                if "relative" in ue.lower():
                    dependencies += [f"background_train_{self.proxy}token_embeddings_{layer}"]
            if ue == "TokenMahalanobis":
                self.tmds[layer] = TokenMahalanobisDistance(
                    embeddings_type, None, normalize=False, metric_thr=metric_thr, metric=metric_md, metric_name=metric_md_name, aggregation="none", hidden_layer=layer, aggregated=aggregated, device=self.device, storage_device=self.storage_device, is_proxy_model=is_proxy_model, proxy_model_name=proxy_model_name
                )
            elif ue == "RelativeTokenMahalanobis":
                self.tmds[layer] = RelativeTokenMahalanobisDistance(
                    embeddings_type, None, normalize=False, metric_thr=metric_thr, metric=metric_md, metric_name=metric_md_name, aggregation="none", hidden_layer=layer, aggregated=aggregated, device=self.device, storage_device=self.storage_device, is_proxy_model=is_proxy_model, proxy_model_name=proxy_model_name
                )
        super().__init__(dependencies, "sequence")
        self.parameters_path=parameters_path
        self.is_fitted = False
        self.metric_thr = metric_thr
        self.aggregated=aggregated
        if metric is not None:
            self.metric = metric
            if aggregated:
                self.metric = AggregatedMetric(base_metric=self.metric)
        self.aggregation = aggregation
        self.metric_name = metric_name
        self.metric_md_name = metric_md_name
        self.embeddings_type=embeddings_type
        self.positive=positive
        self.meta_model=meta_model
        self.norm=norm
        self.tgt_norm=tgt_norm
        self.remove_corr=remove_corr
        self.remove_alg=remove_alg
        self.msp = MaximumSequenceProbability()
        self.ent = EntropyCalculator()
        self.use_tad=use_tad

    def __str__(self):
        hidden_layers = ",".join([str(x) for x in self.hidden_layers])
        positive = "pos" if self.positive else ""
        tgt_norm = "tgt_norm" if self.tgt_norm else ""
        remove_corr = f"remove_corr_{self.remove_alg}" if self.remove_corr else ""
        use_tad = f"+tad" if self.use_tad else ""
        return f"Hybrid{self.meta_model}{self.ue}Distance_{self.proxy}{self.embeddings_type}{hidden_layers}{use_tad} ({self.aggregation}, {self.metric_name}, {self.metric_md_name}, {self.metric_thr}, {positive}, {self.norm}, {tgt_norm}, {remove_corr}{self.sim_pca_name})"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        
        if not self.is_fitted: 
            train_greedy_texts = stats[f"train_greedy_texts"]
            train_greedy_tokens = stats[f"train_{self.proxy}tokens"] if self.is_proxy_model else stats[f"train_greedy_tokens"]
            train_target_texts = stats[f"train_target_texts"]
            train_greedy_log_probs = stats[f"train_greedy_log_probs"]
            train_greedy_log_likelihoods = stats[f"train_greedy_log_likelihoods"]

            metric_key = f"{self.proxy}train_seq_{self.metric_name}_{len(train_greedy_texts)}"
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

            train_mds = []
            dev_size = 0.5 
            train_idx, dev_idx = train_test_split(list(range(len(train_greedy_texts))), test_size=dev_size, shuffle=True, random_state=42)
            lens = np.array([0]+[len(tokens) for tokens in train_greedy_tokens])
            tokens_before = np.cumsum(lens)
            token_train_idx = np.concatenate([np.arange(tokens_before[i], tokens_before[i+1]) for i in train_idx])
            token_dev_idx = np.concatenate([np.arange(tokens_before[i], tokens_before[i+1]) for i in dev_idx])
            print("Mean score for train/dev: ", np.mean(self.train_seq_metrics[train_idx]), np.mean(self.train_seq_metrics[dev_idx]))

            centroids = []
            for layer in tqdm(self.hidden_layers):
                if layer == -1:
                    train_token_embeddings = stats[f"train_{self.proxy}token_embeddings_{self.embeddings_type}"]
                    train_stats = {f"train_{self.proxy}tokens": [train_greedy_tokens[k] for k in train_idx], 
                                   f"train_greedy_tokens": [train_greedy_tokens[k] for k in train_idx], 
                                   "train_greedy_texts":[train_greedy_texts[k] for k in train_idx],
                                   f"{self.proxy}tokens": [train_greedy_tokens[k] for k in dev_idx], 
                                   f"greedy_tokens": [train_greedy_tokens[k] for k in dev_idx], 
                                   "train_target_texts": [train_target_texts[k] for k in train_idx],
                                   f"train_{self.proxy}token_embeddings_{self.embeddings_type}": [train_token_embeddings[k] for k in token_train_idx], #train_token_embeddings[token_train_idx],
                                   f"{self.proxy}token_embeddings_{self.embeddings_type}": [train_token_embeddings[k] for k in token_dev_idx], #train_token_embeddings[token_dev_idx],
                                  }
                    if "relative" in self.ue.lower(): 
                        train_stats[f"background_train_{self.proxy}token_embeddings_{self.embeddings_type}"] = stats[f"background_train_{self.proxy}token_embeddings_{self.embeddings_type}"]
                else:
                    train_token_embeddings = stats[f"train_{self.proxy}token_embeddings_{self.embeddings_type}_{layer}"]
                    train_stats = {f"train_{self.proxy}tokens": [train_greedy_tokens[k] for k in train_idx], 
                                   f"train_greedy_tokens": [train_greedy_tokens[k] for k in train_idx], 
                                   "train_greedy_texts": [train_greedy_texts[k] for k in train_idx],
                                   f"{self.proxy}tokens": [train_greedy_tokens[k] for k in dev_idx], 
                                   f"greedy_tokens": [train_greedy_tokens[k] for k in dev_idx], 
                                   "train_target_texts": [train_target_texts[k] for k in train_idx],
                                   f"train_{self.proxy}token_embeddings_{self.embeddings_type}_{layer}": [train_token_embeddings[k] for k in token_train_idx], #train_token_embeddings[token_train_idx],
                                   f"{self.proxy}token_embeddings_{self.embeddings_type}_{layer}": [train_token_embeddings[k] for k in token_dev_idx], #train_token_embeddings[token_dev_idx],
                                  }
                    if "relative" in self.ue.lower(): 
                        train_stats[f"background_train_{self.proxy}token_embeddings_{self.embeddings_type}_{layer}"] = stats[f"background_train_{self.proxy}token_embeddings_{self.embeddings_type}_{layer}"]
                    
                metric_key = f"{self.proxy}train_{self.metric_md_name}_{len(train_greedy_texts)}"
                if metric_key in stats.keys():
                    train_stats[f"{self.proxy}train_{self.metric_md_name}_{len(train_idx)}"] = stats[metric_key][token_train_idx]

                if layer == -1:
                    hidden_layer = ""
                else:
                    hidden_layer = f"_{layer}"
                    
                centroid_key_ = f"{self.proxy}centroid{hidden_layer}_{self.metric_name}_{self.metric_thr}_{len(train_idx)}"
                covariance_key_ = f"{self.proxy}covariance{hidden_layer}_{self.metric_name}_{self.metric_thr}_{len(train_idx)}"

                background_centroid_key_ = f"{self.proxy}background_centroid{hidden_layer}_{self.metric_name}_{self.metric_thr}_{len(train_idx)}"
                background_covariance_key_ = f"{self.proxy}background_covariance{hidden_layer}_{self.metric_name}_{self.metric_thr}_{len(train_idx)}"

                if centroid_key_ in stats.keys():
                    train_stats[centroid_key_] = stats[centroid_key_]
                    centroids.append(stats[centroid_key_].cpu().detach().numpy())
                if covariance_key_ in stats.keys():
                    train_stats[covariance_key_] = stats[covariance_key_]
                if background_centroid_key_ in stats.keys():
                    train_stats[background_centroid_key_] = stats[background_centroid_key_]
                if background_covariance_key_ in stats.keys():
                    train_stats[background_covariance_key_] = stats[background_covariance_key_]
                
                md = self.tmds[layer](train_stats, save_data=False).reshape(-1)

                if "Relative" in self.ue:
                    if centroid_key_ not in stats.keys():
                        stats[centroid_key_] = self.tmds[layer].MD.centroid
                        centroids.append(stats[centroid_key_].cpu().detach().numpy())
                    if covariance_key_ not in stats.keys():
                        stats[covariance_key_] = self.tmds[layer].MD.sigma_inv  
                    if background_centroid_key_ not in stats.keys():
                        stats[background_centroid_key_] = self.tmds[layer].centroid_0
                    if background_covariance_key_ not in stats.keys():
                        stats[background_covariance_key_] = self.tmds[layer].sigma_inv_0
                else:
                    if centroid_key_ not in stats.keys():
                        stats[centroid_key_] = self.tmds[layer].centroid
                        centroids.append(stats[centroid_key_].cpu().detach().numpy())
                    if covariance_key_ not in stats.keys():
                        stats[covariance_key_] = self.tmds[layer].sigma_inv

                self.tmds[layer].is_fitted = False
                k = 0
                mean_md = []
                for tokens in [train_greedy_tokens[k] for k in dev_idx]:
                    dists_i = md[k:k+len(tokens)]
                    k += len(tokens)
                    mean_md.append(np.mean(dists_i))
                train_mds.append(mean_md)
            train_dists = np.array(train_mds).T
            train_dists[np.isnan(train_dists)] = 0
            if self.meta_model == "LinReg":
                self.regressor = Ridge(positive=self.positive)
            elif self.meta_model == "Lasso":
                self.regressor = Lasso()
            elif self.meta_model == "StableLinReg":
                self.regressor = StableLinReg()
            elif self.meta_model == "WeightedStableLinReg":
                self.regressor = StableLinReg(weighted=True)
            elif self.meta_model == "MLP":                
                self.regressor = MLP(n_features=train_dists.shape[1])
            elif self.meta_model == "weights":
                scores = []
                for i in range(train_dists.shape[-1]):
                    scores.append(get_prr(train_dists[:, i], self.train_seq_metrics[dev_idx]))
                self.weights = np.array(scores)
                self.weights /= np.abs(self.weights).sum()
                
            X = np.zeros_like(train_dists)
            for col in range(train_dists.shape[1]):
                X[:, col] = rankdata(train_dists[:, col])
                if self.norm == "norm":
                    X[:, col] /= X[:, col].max()
            if self.norm == "orig":
                X = train_dists
            elif self.norm == "scaler":
                scaler = StandardScaler()
                X = scaler.fit_transform(train_dists)

            if self.remove_corr:
                feats = np.arange(X.shape[1])
                if self.remove_alg == 1:
                    removed = np.zeros_like(feats, dtype=bool)
                    added = np.zeros_like(feats, dtype=bool)
                    
                    for f in feats:
                        if removed[f]:
                            continue
                        added[f] = True
                        corr_idx = np.argwhere((np.abs(np.corrcoef(X.T)[f]) > 0.8) & (np.arange(X.shape[1]) != f)).flatten()
                        removed[corr_idx] = True
                    self.added = added
                    X = X[:, self.added]

                if self.remove_alg == 2:
                    X_corr = np.corrcoef(X.T)
                    d = sch.distance.pdist(X_corr, metric="cosine")
                    L = sch.linkage(d, method='complete')
                    clusters = sch.fcluster(L, 0.3*d.max(), 'distance')
    
                    features = []
                    for cluster in np.unique(clusters):
                        cls_features = []
                        cls_prr = []
                        for f in np.argwhere(clusters == cluster).flatten():
                            cls_features.append(f)
                            cls_prr.append(np.abs(get_prr(train_dists[:, f], self.train_seq_metrics[dev_idx])))
                        features.append(cls_features[np.argmax(cls_prr)]) 
                    self.added = np.zeros_like(feats, dtype=bool)
                    self.added[features] = True
                    X = X[:, self.added]

                if self.remove_alg == 3:
                    if self.sim_pca:
                        from sklearn.metrics.pairwise import cosine_similarity
                        centroids = np.array(centroids)
                        sim = cosine_similarity(centroids)

                        self.L = np.linalg.cholesky(sim)
                        self.scaler = StandardScaler()
                        X = self.scaler.fit_transform(X)
                        _, _, self.v = np.linalg.svd(X.dot(self.L), full_matrices=False)
                        X = X @ self.L @ self.v.T[:, :10]
                        
                    else:
                        self.pca = PCA(n_components=10)
                        X = self.pca.fit_transform(X)
                    
                if self.remove_alg == 4:
                    self.pca = PCA(n_components=X.shape[1])
                    self.pca.fit(X)
                    
                    n_components = np.argwhere(np.cumsum(self.pca.explained_variance_ratio_) > 0.99).min()
                    self.pca = PCA(n_components=n_components)
                    X = self.pca.fit_transform(X)

            msp = np.array(self.msp({"greedy_log_likelihoods": [train_greedy_log_likelihoods[i] for i in dev_idx]}))
            ent = np.array([np.mean(x) for x in self.ent({"greedy_log_probs": [train_greedy_log_probs[i] for i in dev_idx]})["entropy"]])
            if self.use_tad:
                pass
                # tad_stats = {
                #     "tokenizer": stats["tokenizer"],
                #     "greedy_texts": stats["train_greedy_texts"][dev_samples:],
                #     "greedy_tokens": stats["train_greedy_tokens"][dev_samples:],
                #     "greedy_log_probs": stats["train_greedy_log_probs"][dev_samples:],
                #     "greedy_log_likelihoods": stats["train_greedy_log_likelihoods"][dev_samples:],
                #     "attention_features": stats["train_attention_features"][dev_tokens-dev_samples:],
                #     "train_input_texts": stats["train_input_texts"][:dev_samples],
                #     "train_target_texts": stats["train_target_texts"][:dev_samples],
                #     "train_greedy_texts": stats["train_greedy_texts"][:dev_samples],
                #     "train_greedy_tokens": stats["train_greedy_tokens"][:dev_samples],
                #     "train_greedy_log_probs": stats["train_greedy_log_probs"][:dev_samples],
                #     "train_greedy_log_likelihoods": stats["train_greedy_log_likelihoods"][:dev_samples],
                #     "train_attention_features": stats["train_attention_features"][:dev_tokens-dev_samples],
                # }
                # tad = np.array(self.tad(tad_stats))
                # self.tad.is_fitted = False
                # X = np.hstack([X, msp.reshape(-1, 1), ent.reshape(-1, 1), tad.reshape(-1, 1)])
            else:
                X = np.hstack([X, msp.reshape(-1, 1), ent.reshape(-1, 1)])
            
            target = self.train_seq_metrics[dev_idx]
            target[np.isnan(target)] = 0
            if self.tgt_norm:
                y = 1 - rankdata(target)
            else:
                y = 1 - target
            
            if self.meta_model != "weights":
                self.regressor.fit(X, y)
            self.is_fitted = True


        eval_mds = []
        greedy_tokens = stats[f"{self.proxy}tokens"] if self.is_proxy_model else stats[f"greedy_tokens"]
        for layer in self.tmds.keys():
            md = self.tmds[layer](stats).reshape(-1)
            k = 0
            mean_md = []
            for tokens in greedy_tokens:
                dists_i = md[k:k+len(tokens)]
                k += len(tokens)
                mean_md.append(np.mean(dists_i))
            eval_mds.append(mean_md)
        eval_dists = np.array(eval_mds).T
        eval_dists[np.isnan(eval_dists)] = 0
        if self.norm == "scaler":
            eval_dists = scaler.transform(eval_dists)
            
        if self.meta_model != "weights":
            if self.remove_corr and (self.remove_alg < 3):
                eval_dists = eval_dists[:, self.added]
            elif self.remove_corr:
                if self.sim_pca:
                    eval_dists = self.scaler.transform(eval_dists)
                    eval_dists = eval_dists @ self.L @ self.v.T[:, :10]
                else:
                    eval_dists = self.pca.transform(eval_dists)

            msp = np.array(self.msp({"greedy_log_likelihoods": stats["greedy_log_likelihoods"]}))
            ent = np.array([np.mean(x) for x in self.ent({"greedy_log_probs": stats["greedy_log_probs"]})["entropy"]])

            if self.use_tad:
                tad = np.array(self.tad(stats))
                eval_dists = np.hstack([eval_dists, msp.reshape(-1, 1), ent.reshape(-1, 1), tad.reshape(-1, 1)])
            else:
                eval_dists = np.hstack([eval_dists, msp.reshape(-1, 1), ent.reshape(-1, 1)])
            
            ues = self.regressor.predict(eval_dists)
        else:
            ues = eval_dists @ self.weights
        return ues

class LinRegTokenMahalanobisDistance_Hybrid_Claim(Estimator):
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

        metric_md = None,
        metric_md_name: str = "",
        
        aggregated: bool = False,
        positive: bool = True,
        ue: str = "TokenMahalanobis",

        meta_model: str = "LinReg",
        norm: str = "norm",

        tgt_norm: bool = False,
        remove_corr: bool = False,
        remove_alg: int = 2,

        use_ccp: bool = False,
        ccp_context: str = "no_context",
        
        device: str = "cuda",
        storage_device: str = "cuda",
        
    ):
        self.ue = ue
        self.hidden_layers = hidden_layers
        self.device = device
        self.storage_device = storage_device
        self.tmds = {}
        dependencies = ["train_greedy_tokens", "train_target_texts",  "claims", "train_claims"]
        self.use_ccp = use_ccp
        if self.use_ccp:
            dependencies += ["greedy_tokens", "greedy_tokens_alternatives", "greedy_tokens_alternatives_nli", "greedy_tokens_alternatives_fact_pref_nli", 
                             "train_greedy_tokens_alternatives", "train_greedy_tokens_alternatives_nli", "train_greedy_tokens_alternatives_fact_pref_nli"]

        for layer in self.hidden_layers:
            if layer == -1:
                dependencies += ["token_embeddings", "train_token_embeddings"]
                if "relative" in ue.lower():
                    dependencies += ["background_train_token_embeddings", "background_train_embeddings"]
            else:
                dependencies += [f"token_embeddings_{layer}", f"train_token_embeddings_{layer}"]
                if "relative" in ue.lower():
                    dependencies += [f"background_train_token_embeddings_{layer}", f"background_train_embeddings_{layer}"]
            if ue == "TokenMahalanobis":
                self.tmds[layer] = TokenMahalanobisDistanceClaim(
                    embeddings_type, None, normalize=False, metric_thr=metric_thr, aggregation="none", hidden_layer=layer, device=self.device, storage_device=self.storage_device
                )
            elif ue == "RelativeTokenMahalanobis":
                self.tmds[layer] = RelativeTokenMahalanobisDistanceClaim(
                    embeddings_type, None, normalize=False, metric_thr=metric_thr, aggregation="none", hidden_layer=layer, device=self.device, storage_device=self.storage_device
                )
                
        super().__init__(dependencies, "claim")
        self.parameters_path=parameters_path
        self.is_fitted = False
        self.metric_thr = metric_thr
        if metric is not None:
            self.metric = metric
            if aggregated:
                self.metric = AggregatedMetric(base_metric=self.metric)
        self.aggregation = aggregation
        self.metric_name = metric_name
        self.metric_md_name = metric_md_name
        self.embeddings_type=embeddings_type
        self.positive=positive
        self.meta_model=meta_model
        self.norm=norm
        self.tgt_norm=tgt_norm
        self.remove_corr=remove_corr
        self.remove_alg=remove_alg
        self.factcheck = OpenAIFactCheck(openai_model="gpt-4o-mini")
        os.makedirs(self.parameters_path, exist_ok=True)
        self.msp = MaximumClaimProbability()
        self.ent_calc = EntropyCalculator()
        self.ent = MaxTokenEntropyClaim()        
        if self.use_ccp:
            self.ccp_context = ccp_context
            self.ccp = ClaimConditionedProbabilityClaim(nli_context = self.ccp_context)

        
    def __str__(self):
        hidden_layers = ",".join([str(x) for x in self.hidden_layers])
        positive = "pos" if self.positive else ""
        tgt_norm = "tgt_norm" if self.tgt_norm else ""
        remove_corr = f"remove_corr_{self.remove_alg}" if self.remove_corr else ""
        ccp = f"+{str(self.ccp)}" if self.use_ccp else ""
        return f"Hybrid{self.meta_model}{self.ue}Distance{ccp}_{self.embeddings_type}{hidden_layers} Claim ({self.aggregation}, {self.metric_name}, {self.metric_md_name}, {self.metric_thr}, {positive}, {self.norm}, {tgt_norm}, {remove_corr})"

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

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        
        if not self.is_fitted: 

            train_greedy_texts = stats[f"train_greedy_texts"]
            train_greedy_tokens = stats[f"train_greedy_tokens"]
            train_input_texts = stats[f"train_input_texts"]
            train_target_texts = stats[f"train_target_texts"]
            train_claims = stats[f"train_claims"]
            train_greedy_log_probs = stats[f"train_greedy_log_probs"]
            train_greedy_log_likelihoods = stats[f"train_greedy_log_likelihoods"]
            train_stats = {"claims": train_claims, "input_texts": train_input_texts}

            if "factcheck" in stats.keys():
                self.factcheck_score = stats["factcheck"]
                self.train_token_metrics = stats["train_token_metrics"]
            else:
                self.factcheck_score = self.factcheck(train_stats, None, None)
                self.train_token_metrics = np.concatenate(self._get_targets(train_greedy_tokens, train_claims, self.factcheck_score))
                stats["factcheck"] = self.factcheck_score
                stats["train_token_metrics"] = self.train_token_metrics
            
            train_mds = []
            dev_size = 0.5 
            train_idx, dev_idx = train_test_split(list(range(len(train_greedy_texts))), test_size=dev_size, random_state=42, shuffle=False)
            lens = np.array([0]+[len(tokens) for tokens in train_greedy_tokens])
            tokens_before = np.cumsum(lens)
            token_train_idx = np.concatenate([np.arange(tokens_before[i], tokens_before[i+1]) for i in train_idx]).astype(int)
            token_dev_idx = np.concatenate([np.arange(tokens_before[i], tokens_before[i+1]) for i in dev_idx]).astype(int)           
            for layer in self.hidden_layers:
                if layer == -1:
                    train_token_embeddings = stats[f"train_token_embeddings_{self.embeddings_type}"]
                    train_stats = {"train_greedy_tokens": [train_greedy_tokens[k] for k in train_idx], 
                                   "train_greedy_texts":[train_greedy_texts[k] for k in train_idx],
                                   "greedy_tokens": [train_greedy_tokens[k] for k in dev_idx], 
                                   "train_target_texts": [train_target_texts[k] for k in train_idx],
                                   f"train_token_embeddings_{self.embeddings_type}": [train_token_embeddings[k] for k in token_train_idx],
                                   f"token_embeddings_{self.embeddings_type}": [train_token_embeddings[k] for k in token_dev_idx],
                                   "claims": [train_claims[k] for k in dev_idx],
                                   "train_claims": [train_claims[k] for k in train_idx],
                                   "train_input_texts": [train_input_texts[k] for k in train_idx],
                                  }
                    if "relative" in self.ue.lower(): 
                        train_stats[f"background_train_token_embeddings_{self.embeddings_type}"] = stats[f"background_train_token_embeddings_{self.embeddings_type}"]
                else:
                    train_token_embeddings = stats[f"train_token_embeddings_{self.embeddings_type}_{layer}"]
                    train_stats = {"train_greedy_tokens": [train_greedy_tokens[k] for k in train_idx], 
                                   "train_greedy_texts": [train_greedy_texts[k] for k in train_idx],
                                   "greedy_tokens": [train_greedy_tokens[k] for k in dev_idx], 
                                   "train_target_texts": [train_target_texts[k] for k in train_idx],
                                   f"train_token_embeddings_{self.embeddings_type}_{layer}": [train_token_embeddings[k] for k in token_train_idx],
                                   f"token_embeddings_{self.embeddings_type}_{layer}": [train_token_embeddings[k] for k in token_dev_idx],
                                    "claims": [train_claims[k] for k in dev_idx],
                                   "train_claims": [train_claims[k] for k in train_idx],
                                   "train_input_texts": [train_input_texts[k] for k in train_idx],
                                  }
                    if "relative" in self.ue.lower(): 
                        train_stats[f"background_train_token_embeddings_{self.embeddings_type}_{layer}"] = stats[f"background_train_token_embeddings_{self.embeddings_type}_{layer}"]
                    
                train_stats["factcheck"] = [self.factcheck_score[i] for i in train_idx]
                train_stats["train_token_metrics"] = self.train_token_metrics[token_train_idx]


                if layer == -1:
                    hidden_layer = ""
                else:
                    hidden_layer = f"_{layer}"
                    
                centroid_key_ = f"centroid{hidden_layer}_{self.metric_thr}_{len(train_idx)}"
                covariance_key_ = f"covariance{hidden_layer}_{self.metric_thr}_{len(train_idx)}"

                background_centroid_key_ = f"background_centroid{hidden_layer}_{self.metric_thr}_{len(train_idx)}"
                background_covariance_key_ = f"background_covariance{hidden_layer}_{self.metric_thr}_{len(train_idx)}"

                if centroid_key_ in stats.keys():
                    train_stats[centroid_key_] = stats[centroid_key_]
                if covariance_key_ in stats.keys():
                    train_stats[covariance_key_] = stats[covariance_key_]
                if background_centroid_key_ in stats.keys():
                    train_stats[background_centroid_key_] = stats[background_centroid_key_]
                if background_covariance_key_ in stats.keys():
                    train_stats[background_covariance_key_] = stats[background_covariance_key_]
                
                md = self.tmds[layer](train_stats, save_data=False).reshape(-1)

                if "Relative" in self.ue:
                    if centroid_key_ not in stats.keys():
                        stats[centroid_key_] = self.tmds[layer].MD.centroid
                    if covariance_key_ not in stats.keys():
                        stats[covariance_key_] = self.tmds[layer].MD.sigma_inv  
                    if background_centroid_key_ not in stats.keys():
                        stats[background_centroid_key_] = self.tmds[layer].centroid_0
                    if background_covariance_key_ not in stats.keys():
                        stats[background_covariance_key_] = self.tmds[layer].sigma_inv_0
                else:
                    if centroid_key_ not in stats.keys():
                        stats[centroid_key_] = self.tmds[layer].centroid
                    if covariance_key_ not in stats.keys():
                        stats[covariance_key_] = self.tmds[layer].sigma_inv

                self.tmds[layer].is_fitted = False
                train_mds.append(md)
            train_dists = np.array(train_mds).T
            tmd_scores = []
            k = 0
            for idx, tokens in enumerate([train_greedy_tokens[k] for k in dev_idx]):
                dists_i = train_dists[k:k+len(tokens)]
                k += len(tokens)
    
                tmd_scores.append([])
                for claim in [train_claims[k] for k in dev_idx][idx]:
                    tokens = np.array(claim.aligned_token_ids)
                    claim_p_i = dists_i[tokens]
                    
                    if self.aggregation == "mean":
                        tmd_scores[-1].append(claim_p_i.mean(axis=0))
                    elif self.aggregation  == "sum":
                        tmd_scores[-1].append(claim_p_i.sum(axis=0))

            train_dists = np.concatenate(tmd_scores)
            train_dists[np.isnan(train_dists)] = 0
            if self.meta_model == "LinReg":
                self.regressor = Ridge(positive=self.positive)
            elif self.meta_model == "MLP":                
                self.regressor = MLP(n_features=train_dists.shape[1])
            elif self.meta_model == "weights":
                scores = []
                for i in range(train_dists.shape[-1]):
                    scores.append(get_prr(train_dists[:, i], self.train_token_metrics[dev_idx]))
                self.weights = np.array(scores)
                self.weights /= np.abs(self.weights).sum()
                
            X = np.zeros_like(train_dists)
            for col in range(train_dists.shape[1]):
                X[:, col] = rankdata(train_dists[:, col])
                if self.norm == "norm":
                    X[:, col] /= X[:, col].max()
            if self.norm == "orig":
                X = train_dists
            elif self.norm == "scaler":
                scaler = StandardScaler()
                X = scaler.fit_transform(train_dists)

            if self.remove_corr:
                feats = np.arange(X.shape[1])
                if self.remove_alg == 1:
                    removed = np.zeros_like(feats, dtype=bool)
                    added = np.zeros_like(feats, dtype=bool)
                    
                    for f in feats:
                        if removed[f]:
                            continue
                        added[f] = True
                        corr_idx = np.argwhere((np.abs(np.corrcoef(X.T)[f]) > 0.8) & (np.arange(X.shape[1]) != f)).flatten()
                        removed[corr_idx] = True
                    self.added = added
                    X = X[:, self.added]

                if self.remove_alg == 2:
                    X_corr = np.corrcoef(X.T)
                    d = sch.distance.pdist(X_corr, metric="cosine")
                    L = sch.linkage(d, method='complete')
                    clusters = sch.fcluster(L, 0.3*d.max(), 'distance')
    
                    features = []
                    for cluster in np.unique(clusters):
                        cls_features = []
                        cls_prr = []
                        for f in np.argwhere(clusters == cluster).flatten():
                            cls_features.append(f)
                            cls_prr.append(np.abs(get_prr(train_dists[:, f], self.train_seq_metrics[dev_idx])))
                        features.append(cls_features[np.argmax(cls_prr)]) 
                    self.added = np.zeros_like(feats, dtype=bool)
                    self.added[features] = True
                    X = X[:, self.added]

                if self.remove_alg == 3:
                    self.pca = PCA(n_components=10)
                    X = self.pca.fit_transform(X)
                    
                if self.remove_alg == 4:
                    self.pca = PCA(n_components=X.shape[1])
                    self.pca.fit(X)
                    
                    n_components = np.argwhere(np.cumsum(self.pca.explained_variance_ratio_) > 0.99).min()
                    self.pca = PCA(n_components=n_components)
                    X = self.pca.fit_transform(X)

            msp = np.concatenate(self.msp({"greedy_log_likelihoods": [stats[f"train_greedy_log_likelihoods"][i] for i in dev_idx],
                                           "claims": [train_claims[i] for i in dev_idx]}))
            
            ent_token = self.ent_calc({"greedy_log_probs": [train_greedy_log_probs[i] for i in dev_idx]})["entropy"]
            ent = np.concatenate(self.ent({"entropy": ent_token, "claims": [train_claims[i] for i in dev_idx]}))

            if self.use_ccp:
                if self.ccp_context == "fact_pref":
                    train_greedy_tokens_alternatives = stats[f"train_greedy_tokens_alternatives"]
                    train_greedy_tokens_alternatives_fact_pref_nli = stats[f"train_greedy_tokens_alternatives_fact_pref_nli"]
    
                    train_stats = {"greedy_tokens": [train_greedy_tokens[i] for i in dev_idx], 
                                   "greedy_tokens_alternatives": [train_greedy_tokens_alternatives[i] for i in dev_idx],
                                   "greedy_tokens_alternatives_fact_pref_nli": [train_greedy_tokens_alternatives_fact_pref_nli[i] for i in dev_idx],
                                   "claims": [train_claims[i] for i in dev_idx],}

                else:
                    train_greedy_tokens_alternatives = stats[f"train_greedy_tokens_alternatives"]
                    train_greedy_tokens_alternatives_nli = stats[f"train_greedy_tokens_alternatives_nli"]
    
                    train_stats = {"greedy_tokens": [train_greedy_tokens[i] for i in dev_idx], 
                                   "greedy_tokens_alternatives": [train_greedy_tokens_alternatives[i] for i in dev_idx],
                                   "greedy_tokens_alternatives_nli": [train_greedy_tokens_alternatives_nli[i] for i in dev_idx],
                                   "claims": [train_claims[i] for i in dev_idx],}
                    
                ccp = np.concatenate(self.ccp(train_stats))
                X = np.hstack([X, msp.reshape(-1, 1), ent.reshape(-1, 1), ccp.reshape(-1, 1)])
            else:
                X = np.hstack([X, msp.reshape(-1, 1), ent.reshape(-1, 1)])
            
            n_claims = len(np.concatenate([train_claims[i] for i in dev_idx]))
            target = np.concatenate([self.factcheck_score[i] for i in dev_idx])
            target[np.isnan(target)] = 0
            if self.tgt_norm:
                y = rankdata(target)
            else:
                y = target
            
            if self.meta_model != "weights":
                self.regressor.fit(X, y)
                self.y_preds = self.regressor.predict(X)
            self.is_fitted = True


        eval_mds = []
        for layer in self.tmds.keys():
            md = self.tmds[layer](stats).reshape(-1)
            eval_mds.append(md)
        eval_dists = np.array(eval_mds).T
        eval_dists[np.isnan(eval_dists)] = 0

        tmd_scores = []
        k = 0
        claims = stats["claims"]
        for idx, tokens in enumerate(stats["greedy_tokens"]):
            dists_i = eval_dists[k:k+len(tokens)]
            k += len(tokens)

            tmd_scores.append([])
            for claim in claims[idx]:
                tokens = np.array(claim.aligned_token_ids)
                claim_p_i = dists_i[tokens]
                
                if self.aggregation == "mean":
                    tmd_scores[-1].append(claim_p_i.mean(axis=0))
                elif self.aggregation  == "sum":
                    tmd_scores[-1].append(claim_p_i.sum(axis=0))
        eval_dists = np.concatenate(tmd_scores)

        
        if self.norm == "scaler":
            eval_dists = scaler.transform(eval_dists)
            
        if self.meta_model != "weights":
            if self.remove_corr and (self.remove_alg < 3):
                eval_dists = eval_dists[:, self.added]
            elif self.remove_corr:
                eval_dists = self.pca.transform(eval_dists)
                
            msp = np.concatenate(self.msp({"greedy_log_likelihoods": stats["greedy_log_likelihoods"], "claims": stats["claims"]}))
            ent_token = self.ent_calc({"greedy_log_probs": stats["greedy_log_likelihoods"]})["entropy"]
            ent = np.concatenate(self.ent({"entropy": ent_token, "claims":stats["claims"]}))

            if self.use_ccp:
                if self.ccp_context == "fact_pref":
                    greedy_tokens_alternatives = stats[f"greedy_tokens_alternatives"]
                    greedy_tokens_alternatives_fact_pref_nli = stats[f"greedy_tokens_alternatives_fact_pref_nli"]
    
                    ccp_stats = {"greedy_tokens": stats["greedy_tokens"], 
                                 "greedy_tokens_alternatives": greedy_tokens_alternatives,
                                 "greedy_tokens_alternatives_fact_pref_nli": greedy_tokens_alternatives_fact_pref_nli,
                                 "claims": claims,}
                else:
                    greedy_tokens_alternatives = stats[f"greedy_tokens_alternatives"]
                    greedy_tokens_alternatives_nli = stats[f"greedy_tokens_alternatives_nli"]
                
                    ccp_stats = {"greedy_tokens": stats["greedy_tokens"], 
                                 "greedy_tokens_alternatives": greedy_tokens_alternatives,
                                 "greedy_tokens_alternatives_nli": greedy_tokens_alternatives_nli,
                                 "claims": claims,}
                ccp = np.concatenate(self.ccp(ccp_stats))
                eval_dists = np.hstack([eval_dists, msp.reshape(-1, 1), ent.reshape(-1, 1), ccp.reshape(-1, 1)])
            else:
                eval_dists = np.hstack([eval_dists, msp.reshape(-1, 1), ent.reshape(-1, 1)])

            claim_ues = self.regressor.predict(eval_dists)
        else:
            claim_ues = eval_dists @ self.weights
            
        tmd_scores = []
        claims = stats["claims"]
        k = 0
        for idx, tokens in enumerate(stats["greedy_tokens"]):
            tmd_scores.append([])
            for _ in claims[idx]:
                tmd_scores[-1].append(claim_ues[k])
                k += 1
        return tmd_scores
