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

from lm_polygraph.generation_metrics.openai_fact_check import OpenAIFactCheck
from average_token_mahalanobis_distance import LinRegTokenMahalanobisDistance, LinRegTokenMahalanobisDistance_Claim
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import rankdata

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from lm_polygraph.ue_metrics import *

from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from lm_polygraph.ue_metrics.ue_metric import get_random_scores
from lm_polygraph.ue_metrics.pred_rej_area import PredictionRejectionArea
from sklearn.model_selection import train_test_split

import scipy
import scipy.cluster.hierarchy as sch
from sklearn.decomposition import PCA

from lm_polygraph.estimators.max_probability import MaximumSequenceProbability
from lm_polygraph.estimators.claim.max_probability import MaximumClaimProbability
from lm_polygraph.estimators.claim.claim_conditioned_probability import ClaimConditionedProbabilityClaim

prr = PredictionRejectionArea()

def total_uncertainty_linear_step(
    epistemic, aleatoric, threshold_min=0.1, threshold_max=0.9, alpha=0.1
):
    n_preds = len(aleatoric)
    n_lowest = int(n_preds * threshold_min)
    n_max = int(n_preds * threshold_max)

    aleatoric_rank = rankdata(aleatoric)
    epistemic_rank = rankdata(epistemic)

    total_rank = np.zeros_like(epistemic)

    total_rank = (1 - alpha) * epistemic_rank + alpha * aleatoric_rank
    # total_rank[(aleatoric_rank > n_max)] = aleatoric_rank[(aleatoric_rank > n_max)]
    total_rank[epistemic_rank <= n_lowest] = rankdata(
        aleatoric[epistemic_rank <= n_lowest]
    )
    total_rank[
        (aleatoric_rank > n_max) & (epistemic_rank <= n_lowest)
    ] = aleatoric_rank[(aleatoric_rank > n_max) & (epistemic_rank <= n_lowest)]

    return total_rank

def grid_search_hp(
    epistemic,
    aleatoric,
    metrics,
    t_min_min=0.0,
    t_min_max=0.3,
    t_max_min=0.8,
    t_max_max=1.0,
    alpha_min=0.0,
    alpha_max=1.0,
    target_metric=PredictionRejectionArea()
):
    t_min_best = 0
    t_max_best = 1
    alpha_best = 0

    eps = 0.01
    best_prr = target_metric(epistemic, metrics)
    for t_min in np.arange(t_min_min, t_min_max + eps, 0.05):
        for t_max in np.arange(t_max_min, t_max_max + eps, 0.05):
            for alpha in np.arange(alpha_min, alpha_max + eps, 0.1):
                unc = total_uncertainty_linear_step(
                    epistemic, aleatoric, t_min, t_max, alpha
                )
                new_prr = target_metric(unc, metrics)
                if new_prr > best_prr:
                    new_prr = new_prr
                    t_min_best = t_min
                    t_max_best = t_max
                    alpha_best = alpha

    return best_prr, t_min_best, t_max_best, alpha_best

def get_prr(ue, metric):
    mean_val = prr(ue, metric) 
    oracle = prr(-metric, metric)
    random = get_random_scores(prr, metric)
    final_score = (mean_val - random) / (oracle - random)
    return final_score

class HUQ_LRTMD(Estimator):
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

        sim_pca: bool = False,
        n_components: int = 10,
        
    ):
        self.ue = ue
        self.hidden_layers = hidden_layers
        self.device = device
        self.storage_device = storage_device
        self.tmds = {}
        train_greedy_tokens = f"train_greedy_tokens"
        
        dependencies = [train_greedy_tokens, "train_target_texts"]
        # dependencies += ["attention_features", "train_attention_features", "train_greedy_log_likelihoods"]
        
        self.sim_pca = sim_pca
        self.sim_pca_name = f", sim_pca" if self.sim_pca else ""
        train_greedy_tokens = f"train_greedy_tokens"
        for layer in self.hidden_layers:
            if layer == -1:
                dependencies += [f"token_embeddings", f"train_token_embeddings"]
                if "relative" in ue.lower():
                    dependencies += [f"background_train_token_embeddings"]
            else:
                dependencies += [f"token_embeddings_{layer}", f"train_token_embeddings_{layer}"]
                if "relative" in ue.lower():
                    dependencies += [f"background_train_token_embeddings_{layer}"]
        super().__init__(dependencies, "sequence")
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
        self.n_components = n_components
        self.md = LinRegTokenMahalanobisDistance(embeddings_type, parameters_path=parameters_path, metric=metric, metric_name=metric_name, 
                                                 metric_md=metric, metric_md_name=metric_name, aggregated=aggregated, 
                                                 hidden_layers=hidden_layers, metric_thr=metric_thr, aggregation=aggregation,
                                                 ue=ue, positive=positive, meta_model=meta_model, norm=norm, n_components=n_components,
                                                 remove_corr=remove_corr, remove_alg=remove_alg, device=device, storage_device=storage_device, sim_pca=sim_pca)
        self.msp = MaximumSequenceProbability()
        self.use_tad=use_tad
    
    def __str__(self):
        hidden_layers = ",".join([str(x) for x in self.hidden_layers])
        positive = "pos" if self.positive else ""
        tgt_norm = "tgt_norm" if self.tgt_norm else ""
        remove_corr = f"remove_corr_{self.remove_alg}_{self.n_components}_comp" if self.remove_corr else ""
        use_tad = f"+tad" if self.use_tad else ""
        return f"HUQ-{self.meta_model}{self.ue}Distance_{self.embeddings_type}{hidden_layers}{use_tad} ({self.aggregation}, {self.metric_name}, {self.metric_md_name}, {self.metric_thr}, {positive}, {self.norm}, {tgt_norm}, {remove_corr}{self.sim_pca_name})"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        if not self.is_fitted: 
            train_greedy_texts = stats[f"train_greedy_texts"]
            train_greedy_tokens = stats[f"train_greedy_tokens"]
           
            dev_size = 0.5 
            train_idx, dev_idx = train_test_split(list(range(len(train_greedy_texts))), test_size=dev_size, random_state=42)
            lens = np.array([0]+[len(tokens) for tokens in train_greedy_tokens])
            tokens_before = np.cumsum(lens)
            token_train_idx = np.concatenate([np.arange(tokens_before[i], tokens_before[i+1]) for i in train_idx])
            token_dev_idx = np.concatenate([np.arange(tokens_before[i], tokens_before[i+1]) for i in dev_idx])
            
            md_eval = self.md(stats)
            self.train_md = self.md.y_preds
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
                # self.train_msp = np.array(self.tad(tad_stats))
                # self.tad.is_fitted = False
            else:
                self.train_msp = np.array(self.msp({"greedy_log_likelihoods": [stats[f"train_greedy_log_likelihoods"][i] for i in dev_idx]}))

            metrics = self.md.train_seq_metrics[dev_idx]
            best_prr, self.t_min_best, self.t_max_best, self.alpha_best = grid_search_hp(self.train_md, self.train_msp, metrics, target_metric=prr)
            print("HUQ PARAMS", self.t_min_best, self.t_max_best, self.alpha_best)
            self.is_fitted = True            
        else: 
            md_eval = self.md(stats)

        if self.use_tad:
            msp_eval = np.array(self.tad(stats))
        else:
            msp_eval = np.array(self.msp({"greedy_log_likelihoods": stats[f"greedy_log_likelihoods"]}))
        msp_eval_plus = np.concatenate([np.array(msp_eval), self.train_msp])
        md_eval_plus = np.concatenate([np.array(md_eval), self.train_md])
        
        ues = total_uncertainty_linear_step(md_eval_plus, msp_eval_plus, threshold_min=self.t_min_best, threshold_max=self.t_max_best, alpha=self.alpha_best)
        ues = ues[:len(msp_eval)]
        return ues

class HUQ_LRTMD_Claim(Estimator):
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

        self.md = LinRegTokenMahalanobisDistance_Claim(embeddings_type, parameters_path=parameters_path, metric=metric, metric_name=metric_name, 
                                                 metric_md=metric, metric_md_name=metric_name, aggregated=aggregated, 
                                                 hidden_layers=hidden_layers, metric_thr=metric_thr, aggregation=aggregation,
                                                 ue=ue, positive=positive, meta_model=meta_model, norm=norm, 
                                                 remove_corr=remove_corr, remove_alg=remove_alg, device=device, storage_device=storage_device)
        self.msp = MaximumClaimProbability()
        if self.use_ccp:
            self.ccp_context = ccp_context
            self.ccp = ClaimConditionedProbabilityClaim(nli_context = self.ccp_context)
        
    
    def __str__(self):
        hidden_layers = ",".join([str(x) for x in self.hidden_layers])
        positive = "pos" if self.positive else ""
        tgt_norm = "tgt_norm" if self.tgt_norm else ""
        remove_corr = f"remove_corr_{self.remove_alg}" if self.remove_corr else ""
        ccp = f"+{str(self.ccp)}" if self.use_ccp else ""
        return f"HUQ {self.meta_model}{self.ue}Distance{ccp}_{self.embeddings_type}{hidden_layers} Claim ({self.aggregation}, {self.metric_name}, {self.metric_md_name}, {self.metric_thr}, {positive}, {self.norm}, {tgt_norm}, {remove_corr})"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        
        if not self.is_fitted: 

            train_greedy_texts = stats[f"train_greedy_texts"]
            train_greedy_tokens = stats[f"train_greedy_tokens"]
            train_input_texts = stats[f"train_input_texts"]
            train_claims = stats[f"train_claims"]
            
            train_mds = []
            dev_size = 0.5 
            train_idx, dev_idx = train_test_split(list(range(len(train_greedy_texts))), test_size=dev_size, random_state=42)
            lens = np.array([0]+[len(tokens) for tokens in train_greedy_tokens])
            tokens_before = np.cumsum(lens)
            token_train_idx = np.concatenate([np.arange(tokens_before[i], tokens_before[i+1]) for i in train_idx])
            token_dev_idx = np.concatenate([np.arange(tokens_before[i], tokens_before[i+1]) for i in dev_idx])
                
            md_eval = self.md(stats)
            tmd_scores = self.md.y_preds

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
                    
                self.aleatoric = np.concatenate(self.ccp(train_stats))
            else:
                self.aleatoric = np.concatenate(self.msp({"greedy_log_likelihoods": [stats[f"train_greedy_log_likelihoods"][i] for i in dev_idx],
                                                          "claims": [train_claims[i] for i in dev_idx]}))
            self.train_md = tmd_scores

            
            metrics = np.concatenate([self.md.factcheck_score[i] for i in dev_idx])

            best_prr, self.t_min_best, self.t_max_best, self.alpha_best = grid_search_hp(self.train_md, self.aleatoric, metrics, target_metric=ROCAUC())
            print("HUQ PARAMS", self.t_min_best, self.t_max_best, self.alpha_best)
            
            self.is_fitted = True
        else: 
            md_eval = self.md(stats)

        if self.use_ccp:
            if self.ccp_context == "fact_pref":
                greedy_tokens_alternatives = stats[f"greedy_tokens_alternatives"]
                greedy_tokens_alternatives_fact_pref_nli = stats[f"greedy_tokens_alternatives_fact_pref_nli"]

                ccp_stats = {"greedy_tokens": stats["greedy_tokens"], 
                             "greedy_tokens_alternatives": greedy_tokens_alternatives,
                             "greedy_tokens_alternatives_fact_pref_nli": greedy_tokens_alternatives_fact_pref_nli,
                             "claims": stats["claims"],}
            else:
                greedy_tokens_alternatives = stats[f"greedy_tokens_alternatives"]
                greedy_tokens_alternatives_nli = stats[f"greedy_tokens_alternatives_nli"]
            
                ccp_stats = {"greedy_tokens": stats["greedy_tokens"], 
                             "greedy_tokens_alternatives": greedy_tokens_alternatives,
                             "greedy_tokens_alternatives_nli": greedy_tokens_alternatives_nli,
                             "claims": stats["claims"],}
            aleatoric_eval = np.concatenate(self.ccp(ccp_stats))
        else:
            aleatoric_eval = np.concatenate(self.msp({"greedy_log_likelihoods": stats[f"greedy_log_likelihoods"], "claims": stats["claims"]}))
        
        aleatoric_eval_plus = np.concatenate([np.array(aleatoric_eval), self.aleatoric])
        md_eval_plus = np.concatenate([np.concatenate(md_eval), self.train_md])
        
        ues = total_uncertainty_linear_step(md_eval_plus, aleatoric_eval_plus, threshold_min=self.t_min_best, threshold_max=self.t_max_best, alpha=self.alpha_best)
        ues = ues[:len(aleatoric_eval_plus)]
        tmd_scores = []
        k = 0
        claims = stats["claims"]
        for idx, tokens in enumerate(stats["greedy_tokens"]):
            tmd_scores.append([])
            for claim in claims[idx]:
                tmd_scores[-1].append(ues[k])
                k += 1
        return tmd_scores
