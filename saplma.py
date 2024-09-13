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

class MLP_NN(nn.Module):
    def __init__(self, n_features: int = 4096, regression: bool = False):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(n_features, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, 1)])
        if regression:
            self.activation = nn.Identity()
        else:
            self.activation = nn.Sigmoid()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.activation(x)

class MLP:
    def __init__(self, 
                 n_epochs: int = 5,
                 batch_size: int = 64,
                 lr: float = 0.001,
                 n_features: int = 4096, 
                 regression: bool = False
                ):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        if regression:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()
        self.model = MLP_NN(n_features, regression)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def fit(self, X, y):
        if not isinstance(X, torch.Tensor):
            X_torch = torch.tensor(X, dtype=torch.float32)
        else:
            X_torch = X.clone().detach().float()
        if not isinstance(y, torch.Tensor):
            y_torch = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
        else:
            y_torch = y_torch.clone().detach().float()
        batch_start = torch.arange(0, len(X), self.batch_size)
        self.model.to(self.device)
        for epoch in range(self.n_epochs):
            self.model.train()
            for start in batch_start:
                X_batch = X_torch[start:start+self.batch_size].to(self.device)
                y_batch = y_torch[start:start+self.batch_size].to(self.device)
                y_pred = self.model(X_batch)
                loss = self.loss(y_pred, y_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
    def predict(self, X):
        if not isinstance(X, torch.Tensor):
            X_torch = torch.tensor(X, dtype=torch.float32)
        else:
            X_torch = X.clone().detach().float()
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

def cross_val_hp(X, y, model_init, params, regression=False):
    if regression:
        best_score = np.inf
        metric = mean_squared_error
    else: 
        best_score = -np.inf
        metric = roc_auc_score
        
    best_params = None
    for param in tqdm(itertools.product(*params.values())):
        model = model_init(param)
        scores_cv = []
        for i, (train, val) in enumerate(KFold(n_splits=5, random_state=1, shuffle=True).split(list(range(len(X))))):

            X_train = X[train]
            X_val = X[val]
        
            y_train = y[train]
            y_val = y[val]
        
            model.fit(X_train, y_train)
            try:
                scores_cv.append(metric(y_val, model.predict(X_val)))
            except Exception as e: 
                print(f"Skip fold {i} with error: {e}")

        if len(scores_cv):
            scores_mean = np.mean(scores_cv)
        elif regression:
            scores_mean = np.inf
        else:
            scores_mean = -np.inf
            
        if regression:
            if scores_mean < best_score:
                best_score = scores_mean
                best_params = param
        else:
            if best_score < scores_mean:
                best_score = scores_mean
                best_params = param
    print("BEST:", best_params, "BEST SCORE:", scores_mean)
    if best_params is None:
       best_params = list(itertools.product(*params.values()))[0]
    return best_params


class SAPLMA(Estimator):
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
        cv_hp: bool = False,
    ):
        self.hidden_layer = hidden_layer
        if self.hidden_layer == -1:
            super().__init__(["train_embeddings", "embeddings", "train_greedy_tokens", "train_target_texts"], "sequence")
        else:
            super().__init__([f"train_embeddings_{self.hidden_layer}", f"embeddings_{self.hidden_layer}", "train_greedy_tokens", "train_target_texts"], "sequence")
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
        self.ue_predictor = MLP(regression=self.regression)
        self.params = {
                "n_epochs": [5, 10],
                "batch_size": [64, 128],
                "lr": [1e-3, 1e-4, 5e-5, 1e-5, 5e-6],
                "n_features": [4096],
                "regression": [self.regression]
        }
        self.model_init = lambda param: MLP(n_epochs=param[0],
                                            batch_size=param[1],
                                            lr=param[2],
                                            n_features=param[3],
                                            regression=param[4])
        self.aggregated = aggregated
        if metric is not None:
            self.metric = metric
            if aggregated:
                self.metric = AggregatedMetric(base_metric=self.metric)

    def __str__(self):
        hidden_layer = "" if self.hidden_layer==-1 else f"_{self.hidden_layer}"
        cv = "cv, " if self.cv_hp else ""
        return f"SAPLMA_{self.embeddings_type}{hidden_layer} ({cv}{self.metric_name})"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        # take the embeddings
        if self.hidden_layer == -1:
            hidden_layer = ""
        else:
            hidden_layer = f"_{self.hidden_layer}"
        embeddings = create_cuda_tensor_from_numpy(
            stats[f"embeddings_{self.embeddings_type}{hidden_layer}"]
        )
        
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
                    metrics.append(self.metric({"greedy_texts": [x], "target_texts": [y_]}, [y_], [y_])[0])
                self.train_seq_metrics = np.array(metrics)
                stats[metric_key] = self.train_seq_metrics
                
            train_embeddings = stats[f"train_embeddings_{self.embeddings_type}{hidden_layer}"]
            train_embeddings[np.isnan(train_embeddings)] = 0
            self.train_seq_metrics[np.isnan(self.train_seq_metrics)] = 0
            train_embeddings = create_cuda_tensor_from_numpy(
                stats[f"train_embeddings_{self.embeddings_type}{hidden_layer}"]
            )
            if self.cv_hp:
                self.params["n_features"] = [train_embeddings.shape[-1]]
                best_params = cross_val_hp(train_embeddings, 1 - self.train_seq_metrics, self.model_init, self.params, regression=self.regression)
                self.ue_predictor = self.model_init(best_params)
            else:
                self.ue_predictor = MLP(n_features=train_embeddings.shape[-1], regression=self.regression)
                
            self.ue_predictor.fit(train_embeddings, 1 - self.train_seq_metrics)
            self.is_fitted = True
                
        ue = self.ue_predictor.predict(embeddings)

        return ue
    
class SAPLMA_truefalse(Estimator):
    ## use only with original truefalse dataset
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
        device: str = "cuda"
    ):
        self.hidden_layer = hidden_layer
        if self.hidden_layer == -1:
            super().__init__(["train_source_embeddings", "embeddings", "train_target_texts"], "sequence")
        else:
            super().__init__([f"train_source_embeddings_{self.hidden_layer}", f"embeddings_{self.hidden_layer}", "train_target_texts"], "sequence")
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
        self.regression = True if metric_name!="Accuracy" else False
        self.ue_predictor = MLP(regression=self.regression)
        if metric is not None:
            self.metric = metric
            if aggregated:
                self.metric = AggregatedMetric(base_metric=self.metric)

    def __str__(self):
        hidden_layer = "" if self.hidden_layer==-1 else f"_{self.hidden_layer}"
        return f"SAPLMA_truefalse_{self.embeddings_type}{hidden_layer}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        # take the embeddings
        if self.hidden_layer == -1:
            hidden_layer = ""
        else:
            hidden_layer = f"_{self.hidden_layer}"            
        embeddings = create_cuda_tensor_from_numpy(
            stats[f"embeddings_{self.embeddings_type}{hidden_layer}"]
        )
        
        # compute centroids if not given
        if not self.is_fitted:
            self.train_seq_metrics = np.array([int(x) for x in stats[f"train_target_texts"]])
            train_embeddings = create_cuda_tensor_from_numpy(
                stats[f"train_source_embeddings_{self.embeddings_type}{hidden_layer}"]
            )
            self.ue_predictor = MLP(n_features=train_embeddings.shape[-1], regression=self.regression)
            self.ue_predictor.fit(train_embeddings, 1 - self.train_seq_metrics)
            self.is_fitted = True
                
        ue = self.ue_predictor.predict(embeddings)
        return ue