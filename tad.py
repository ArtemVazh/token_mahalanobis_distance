import os
import numpy as np
import torch
import itertools
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, roc_auc_score

from typing import Dict
import json

from lm_polygraph.estimators.estimator import Estimator

from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from lm_polygraph.generation_metrics.alignscore import AlignScore
from lm_polygraph.generation_metrics.aggregated_metric import AggregatedMetric
from lm_polygraph.ue_metrics import PredictionRejectionArea
from lm_polygraph.generation_metrics.openai_fact_check import OpenAIFactCheck

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
                 lr: float = 2e-5,
                 batch_size: int = 128):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.loss = nn.MSELoss() 
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
                loss = self.loss(y_pred, y_batch)
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

def get_tad_ue(model, greedy_log_likelihoods, attention_features, val, val_index, kw_masks, aggregation="mean", attention_only=False):
    tad_scores = []
    k = 0
    for idx in val:
        p_i = [np.exp(greedy_log_likelihoods[idx][0])]
        for j in range(1, len(greedy_log_likelihoods[idx])):                
            pt_jm1 = p_i[-1]
            ptt_j = np.exp(greedy_log_likelihoods[idx][j])
            if attention_only:
                X_test_i = [attention_features[val_index[k]].tolist()]
            else:
                X_test_i = [attention_features[val_index[k]].tolist() + [ptt_j, np.exp(greedy_log_likelihoods[idx][j-1]), pt_jm1]]
            y_pred_i = model.predict(X_test_i)[0]   

            p_i_ = ptt_j*pt_jm1 + y_pred_i * (1 - pt_jm1)
            p_i_ = np.clip(p_i_, 0, 1)
            p_i.append(p_i_)
            k += 1
        
        p_i = np.array(p_i)
        if len(kw_masks):
            if kw_masks[idx].sum():
                p_i = p_i[kw_masks[idx]]
                    
        if aggregation == "mean":
            tad_scores.append(-p_i.mean())
        elif aggregation  == "sum(log(p_i))":
            tad_scores.append(-np.log(p_i + 1e-5).sum())        
    return tad_scores

def cross_val_hp(X, y, model_init, params, attention_features, greedy_log_likelihoods, metrics, aggregation, kw_masks, attention_only, target_metric="prr"):
    prr = PredictionRejectionArea()
    if target_metric == "mse":
        best_prr = np.inf
    else: 
        best_prr = -np.inf
    best_params = None
    for param in tqdm(itertools.product(*params.values())):
        model = model_init(param)
        prr_cv = []
        lens = np.array([0]+[len(ll)-1 for ll in greedy_log_likelihoods])
        tokens_before = np.cumsum(lens)
        for i, (train, val) in enumerate(KFold(n_splits=5, random_state=1, shuffle=True).split(list(range(len(greedy_log_likelihoods))))):

            train_index = np.concatenate([np.arange(tokens_before[i], tokens_before[i+1]) for i in train])
            val_index = np.concatenate([np.arange(tokens_before[i], tokens_before[i+1]) for i in val])
            
            X_train = X[train_index]
            X_val = X[val_index]
        
            y_train = y[train_index]
            y_val = y[val_index]
        
            model.fit(X_train, y_train)
            if target_metric == "prr":
                tad_scores = get_tad_ue(model, greedy_log_likelihoods, attention_features, val, val_index, kw_masks, aggregation, attention_only)
    
                metrics_scores = np.array([metrics[i] for i in val])
                tad_scores = np.array(tad_scores)
                prr_cv.append(prr(tad_scores, metrics_scores))
            elif target_metric == "mse":
                y_preds = model.predict(X_val)
                prr_cv.append(mean_squared_error(y_val, y_preds))
    
        prr_mean = np.mean(prr_cv)
        if target_metric == "prr":
            if prr_mean > best_prr:
                best_prr = prr_mean
                best_params = param
        elif target_metric == "mse":
            if prr_mean < best_prr:
                best_prr = prr_mean
                best_params = param
    print("BEST:", best_params, "BEST SCORE:", best_prr)
    return best_params


alignscorer = AlignScore(batch_size=4)

class TAD(Estimator):
    def __init__(
        self,
        regression_model: str = "LogReg",
        ignore_special_tokens: bool = True,
        aggregation: str = "mean",
        clip_y: int = 2,
        use_alignscore: bool = False,
        aggregated: bool = False,
        use_accuracy: bool = False,
        accuracy = None,
        use_comet: bool = False,
        comet = None,
        cross_val: bool = False,
        target_metric: str = "prr",
        parameters_path: str = "",
        use_idf: bool = False,
        use_keywords: bool = False,
        model_name: str = "meta-llama/Meta-Llama-3-8B",
        attention_only: bool = False,
    ):
        super().__init__(["attention_features", "greedy_log_likelihoods", "train_attention_features", "train_greedy_log_likelihoods", "train_greedy_tokens", "train_target_texts"], "sequence")
        self.ignore_special_tokens = ignore_special_tokens
        self.aggregation = aggregation
        self.clip_y = clip_y
        self.regression_model_name = regression_model
        self.use_alignscore = use_alignscore
        self.use_accuracy = use_accuracy
        self.use_comet = use_comet
        self.cross_val = cross_val
        self.target_metric = target_metric
        self.use_idf = use_idf
        self.use_keywords = use_keywords
        self.attention_only = attention_only

        if self.use_idf:
            import pickle
            path = f"focus_data/token_idf_{model_name.split('/')[-1]}.pkl"
            self.token_idf = pickle.load(open(path, "rb"))

        if self.use_keywords:
            self.NER_type = ['PERSON', 'DATE', 'ORG', "GPE", "NORP", 'ORDINAL', 'PRODUCT', 'CARDINAL', 'LOC', "FAC", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE", "TIME", "PERCENT", "MONEY", "QUANTITY"]
            self.pos_tag = ["NOUN", "NUM", "PROPN"]
            import spacy
            self.nlp = spacy.load('en_core_web_sm')

        if len(parameters_path):
            use_alignscore = "_AlignScore" if self.use_alignscore else ""
            self.full_path = f"{parameters_path}/tad_{self.regression_model_name}_{self.aggregation}{use_alignscore}"
            os.makedirs(self.full_path, exist_ok=True)

        self.base_scorer = alignscorer
        self.aggregated = aggregated
        if aggregated:
            self.base_scorer = AggregatedMetric(base_metric=self.base_scorer)
        
        if self.use_alignscore:
            self.scorer = alignscorer
            if aggregated:
                self.scorer = AggregatedMetric(base_metric=self.scorer)

        if self.use_accuracy:
            self.scorer = accuracy
            if aggregated:
                self.scorer = AggregatedMetric(base_metric=self.scorer) 
            self.base_scorer = self.scorer
            
        if self.use_comet:
            self.scorer = comet
            if aggregated:
                self.scorer = AggregatedMetric(base_metric=self.scorer) 
            
        if self.regression_model_name == "LinReg":
            self.regression_model = Ridge(alpha=1)
            self.model_init = lambda param: Ridge(alpha=param[0])
            self.params = {
                "alpha": [1e+1, 1, 1e-1, 1e-2, 1e-3, 1e-4],
            }
        elif self.regression_model_name == "CatBoost":            
            self.regression_model = CatBoostRegressor(iterations=200, 
                                                      loss_function="RMSE",
                                                      learning_rate=1e-2,
                                                      depth=5,
                                                      logging_level='Silent')
            self.model_init = lambda param: CatBoostRegressor(iterations=param[0], 
                                                      loss_function="RMSE",
                                                      learning_rate=param[1],
                                                      depth=param[2],
                                                      logging_level='Silent')
            self.params = {
                "iterations": [100, 200],
                "learning_rate": [1e-1, 1e-2],
                "depth": [3, 5]
            }
        elif self.regression_model_name == "MLP":
            self.regression_model = MLP()
            self.model_init = lambda param: MLP(n_layers=param[0], 
                                                n_epochs=param[1],
                                                lr=param[2],
                                                dropout=param[3],
                                                batch_size=param[4],
                                                n_features=param[5])
            self.params = {
                "n_layers": [2, 4],
                "n_epochs": [10, 20, 30],
                "lr": [1e-5, 3e-5, 5e-5],
                "dropout": [0],
                "batch_size": [64, 128],
                "n_features": [1603]
            }
        self.is_fitted = False

    def _postinit_catboost(self, X):
        if (self.regression_model_name == "CatBoost") and (X.shape[0] > 10_000):
            self.regression_model = CatBoostRegressor(iterations=200, 
                                                      loss_function="RMSE",
                                                      learning_rate=1e-2,
                                                      depth=5,
                                                      logging_level='Silent',
                                                      task_type="GPU",
                                                      devices='0')
            self.model_init = lambda param: CatBoostRegressor(iterations=param[0], 
                                                      loss_function="RMSE",
                                                      learning_rate=param[1],
                                                      depth=param[2],
                                                      logging_level='Silent',
                                                      task_type="GPU",
                                                      devices='0')

    def _get_kw_mask(self, greedy_text, greedy_tokens, tokenizer):
        sentence = self.nlp(greedy_text)
        decodings = tokenizer.batch_decode(greedy_tokens, skip_special_tokens=True)
        span_index = 0
        kw_mask = np.zeros_like(greedy_tokens, dtype=bool)
        try:
            for token_index, token in enumerate(decodings):
                while (token.strip() not in sentence[span_index].text) and (sentence[span_index].text not in token.strip()):
                    span_index += 1
                span = sentence[span_index]
                if span.text not in self.NER_type and (span.ent_type_ in self.NER_type or span.pos_ in self.pos_tag):
                    kw_mask[token_index] = True
        except:
            pass
        return kw_mask

    def __str__(self):
        use_alignscore = ", +AlignScore" if self.use_alignscore else ""
        use_accuracy = ", +Accuracy" if self.use_accuracy else ""
        use_comet = ", +Comet" if self.use_comet else ""
        cross_val = ", +cross_val" if self.cross_val else ""
        target_metric = ", +mse" if self.target_metric == "mse" else ""
        idf = ", +idf" if self.use_idf else ""
        kw = ", +kw" if self.use_keywords else ""
        attention_only = ", attention_only" if self.attention_only else ""
        if self.ignore_special_tokens:
            return f"TAD ({self.regression_model_name}, {self.aggregation}, {self.clip_y}{use_alignscore}{use_accuracy}{use_comet}{cross_val}{target_metric}{idf}{kw}{attention_only})"
        return f"TAD ({self.regression_model_name}, all, {self.aggregation}, {self.clip_y}{use_alignscore}{use_accuracy}{use_comet}{cross_val}{target_metric}{idf}{kw}{attention_only})"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        # take the embeddings
        attention_features = stats["attention_features"]
        greedy_log_likelihoods = stats["greedy_log_likelihoods"]
        greedy_texts = stats["greedy_texts"]
        tokenizer = stats["tokenizer"]

        # compute centroids if not given
        if not self.is_fitted:
            train_attention_features = stats[f"train_attention_features"]
            train_greedy_log_likelihoods = stats[f"train_greedy_log_likelihoods"]
            train_greedy_tokens = stats[f"train_greedy_tokens"]
            train_greedy_texts = stats[f"train_greedy_texts"]
            train_target_texts = stats[f"train_target_texts"]
            train_input_texts = stats[f"train_input_texts"]
            train_greedy_log_probs = stats[f"train_greedy_log_probs"]

            if self.regression_model_name == "MLP":
                self.regression_model = MLP(n_features=train_attention_features.shape[-1] + 3)
                self.params["n_features"] = [train_attention_features.shape[-1] + 3]
            
            X, y = [], []
            kw_masks = []
            new_ll = []
            train_metrics = []
            k = 0
            for idx in range(len(train_target_texts)):
                if isinstance(train_target_texts[idx], list):
                    target_tokens = ''.join([t.lower() for t in tokenizer.tokenize(train_target_texts[idx][0])])
                else:
                    target_tokens = ''.join([t.lower() for t in tokenizer.tokenize(train_target_texts[idx])])
                greedy_tokens = np.array([t.lower() for t in tokenizer.tokenize(tokenizer.decode(train_greedy_tokens[idx]))])
            
                if len(greedy_tokens) != len(train_greedy_tokens[idx]):
                    greedy_tokens = []
                    for i, t in enumerate(train_greedy_tokens[idx]):
                        if i:
                            greedy_tokens.append(tokenizer.decode(t))
                        else:
                            greedy_tokens.append('_'+tokenizer.decode(t))
                    greedy_tokens = np.array(greedy_tokens)

                if isinstance(train_target_texts[idx], list) and (not self.aggregated):
                    train_target_texts_idx = train_target_texts[idx][0]
                elif isinstance(train_target_texts[idx], str) and self.aggregated:
                    train_target_texts_idx = [train_target_texts[idx]]
                else:
                    train_target_texts_idx = train_target_texts[idx]

                score = self.base_scorer({"greedy_texts": [tokenizer.decode(train_greedy_tokens[idx])], "target_texts": [train_target_texts_idx]}, [train_target_texts_idx], [train_target_texts_idx])[0]
                train_metrics.append(score)
                
                if self.use_alignscore:
                    score = self.scorer({"greedy_texts": [tokenizer.decode(train_greedy_tokens[idx])], "target_texts": [train_target_texts_idx]}, [train_target_texts_idx], [train_target_texts_idx])[0]
                    p_i = [np.clip(((greedy_tokens[0] in target_tokens) * 1.0 + score)/2, 0, 1)]
                elif self.use_comet:
                    score = self.scorer({"greedy_texts": [tokenizer.decode(train_greedy_tokens[idx])], 
                                         "target_texts": [train_target_texts[idx]],
                                         "input_texts": [train_input_texts[idx]]}, 
                                        [train_target_texts[idx]], [train_target_texts[idx]])[0]
                    p_i = [np.clip(((greedy_tokens[0] in target_tokens) * 1.0 + score)/2, 0, 1)]
                elif self.use_accuracy:
                    score = self.scorer({"greedy_texts": [tokenizer.decode(train_greedy_tokens[idx])], "target_texts": [train_target_texts[idx]]}, [train_target_texts[idx]], [train_target_texts[idx]])[0]
                    p_i = [np.clip(((greedy_tokens[0] in target_tokens) * 1.0 + score)/2, 0, 1)]
                else:
                    p_i = [np.clip((greedy_tokens[0] in target_tokens) * 1.0 + 1e-3, 0, 1)]

                if self.use_idf:
                    prob = np.exp(train_greedy_log_probs[idx])
                    mask = prob < 0.01
                    prob[mask] = 0
                    if prob.shape[-1] > len(self.token_idf):
                        prob[:, :len(self.token_idf)] = prob[:, :len(self.token_idf)] * self.token_idf
                    else:
                        prob = prob * self.token_idf
                    prob = prob / np.sum(prob, axis=-1, keepdims=True)
                    ll = np.log(np.array([prob[j, train_greedy_tokens[idx][j]] for j in range(len(prob))]))
                    new_ll.append(ll)
                
                X_i, y_i = [], []
                for j in range(1, len(greedy_tokens)):
                    if self.use_alignscore or self.use_accuracy or self.use_comet:
                        pt_j = np.clip(((greedy_tokens[j] in target_tokens) * 1.0 + score) / 2, 0, 1)
                    else:
                        pt_j = np.clip((greedy_tokens[j] in target_tokens) * 1.0 + 1e-3, 0, 1)
                    pt_jm1 = np.clip(p_i[-1], 0, 1)
                    if self.use_idf:
                        ptt_j = np.exp(ll[j])
                    else:
                        ptt_j = np.exp(train_greedy_log_likelihoods[idx][j])
                    y_value = (pt_j - ptt_j*pt_jm1) / max((1 - pt_jm1), 1e-1)
                    y_value = np.clip(y_value, -self.clip_y, self.clip_y)
                    p_i_ = ptt_j*pt_jm1 + y_value * (1 - pt_jm1)

                    p_i.append(p_i_)
                    y_i.append(y_value)
                    if self.use_idf:
                        if self.attention_only:
                            X_i.append(train_attention_features[k].tolist())
                        else:
                            X_i.append(train_attention_features[k].tolist() + [ptt_j, np.exp(ll[j-1]), pt_jm1])
                    else:
                        if self.attention_only:
                            X_i.append(train_attention_features[k].tolist())
                        else:
                            X_i.append(train_attention_features[k].tolist() + [ptt_j, np.exp(train_greedy_log_likelihoods[idx][j-1]), pt_jm1])
                    k+=1
                    
                if self.use_keywords:
                    kw_masks.append(self._get_kw_mask(train_greedy_texts[idx], train_greedy_tokens[idx], tokenizer))

                if len(X_i):
                    y.append(y_i)
                    X.append(X_i)
                
            X = np.concatenate(X)
            y = np.concatenate(y)

            self._postinit_catboost(X)
            if not self.cross_val:
                self.regression_model.fit(X, y)
            else:
                if self.use_idf:
                    best_params = cross_val_hp(X, y, self.model_init, self.params, train_attention_features, new_ll, train_metrics, self.aggregation, kw_masks, self.attention_only, target_metric=self.target_metric)
                else:
                    best_params = cross_val_hp(X, y, self.model_init, self.params, train_attention_features, train_greedy_log_likelihoods, train_metrics, self.aggregation, kw_masks, self.attention_only, target_metric=self.target_metric)
                self.regression_model = self.model_init(best_params)
                self.regression_model.fit(X, y)   
                with open(f'{self.full_path}/best_params.json', 'w') as fp:
                    json.dump(best_params, fp)
            self.is_fitted = True
            
        greedy_tokens = stats["greedy_tokens"]
        greedy_log_probs = stats["greedy_log_probs"]
        
        tad_scores = []
        k = 0
        print(len(greedy_log_likelihoods))
        print(len(attention_features))
        for idx in range(len(greedy_log_likelihoods)):
            
            if self.use_idf:
                prob = np.exp(greedy_log_probs[idx])
                mask = prob < 0.01
                prob[mask] = 0
                if prob.shape[-1] > len(self.token_idf):
                    prob[:, :len(self.token_idf)] = prob[:, :len(self.token_idf)] * self.token_idf
                else:
                    prob = prob * self.token_idf
                prob = prob / np.sum(prob, axis=-1, keepdims=True)
                ll = np.log(np.array([prob[j, greedy_tokens[idx][j]] for j in range(len(prob))]))
                ll[np.isnan(ll)] = 0
                p_i = [np.exp(ll[0])]
            else:
                p_i = [np.exp(greedy_log_likelihoods[idx][0])]
            for j in range(1, len(greedy_log_likelihoods[idx])):                
                pt_jm1 = p_i[-1]
                
                if self.use_idf:
                    ptt_j = np.exp(ll[j])
                else:
                    ptt_j = np.exp(greedy_log_likelihoods[idx][j])

                if self.use_idf:
                    if self.attention_only:
                        X_test_i = [attention_features[k].tolist()]
                    else:
                        X_test_i = [attention_features[k].tolist() + [ptt_j, np.exp(ll[j-1]), pt_jm1]]
                else:
                    if self.attention_only:
                        X_test_i = [attention_features[k].tolist()]
                    else:
                        X_test_i = [attention_features[k].tolist() + [ptt_j, np.exp(greedy_log_likelihoods[idx][j-1]), pt_jm1]]
                y_pred_i = self.regression_model.predict(X_test_i)[0]  

                p_i_ = ptt_j*pt_jm1 + y_pred_i * (1 - pt_jm1)
                p_i_ = np.clip(p_i_, 0, 1)
                p_i.append(p_i_)
                k += 1
            
            tokens = np.array(greedy_tokens[idx])
            ignore_idx = ((tokens == tokenizer.eos_token_id) | (tokens == tokenizer('.')['input_ids'][-1]))
            if self.ignore_special_tokens:
                p_i = np.array(p_i)[~ignore_idx]
            else:
                p_i = np.array(p_i)
                
            if self.use_keywords:
                kw_mask = self._get_kw_mask(greedy_texts[idx], greedy_tokens[idx], tokenizer)
                if kw_mask.sum():
                    p_i = p_i[kw_mask]
                
            if self.aggregation == "mean":
                tad_scores.append(-p_i.mean())
            elif self.aggregation  == "sum(log(p_i))":
                tad_scores.append(-np.log(p_i + 1e-5).sum())
            elif self.aggregation  == "mean(log(p_i))":
                tad_scores.append(-np.log(p_i + 1e-5).mean())

        return np.array(tad_scores)


def cross_val_hp_claim(X, y, model_init, params, attention_features, greedy_log_likelihoods, claims, targets, factcheck):
    best_mse = -np.inf
    best_params = None
    print("Cross Val Stats")
    for param in tqdm(itertools.product(*params.values())):
        model = model_init(param)
        mse_cv = []
        lens = np.array([0]+[len(ll)-1 for ll in greedy_log_likelihoods])
        tokens_before = np.cumsum(lens)         
        for i, (train, val) in enumerate(KFold(n_splits=5, random_state=1, shuffle=True).split(list(range(len(greedy_log_likelihoods))))):

            train_index = np.concatenate([np.arange(tokens_before[i], tokens_before[i+1]) for i in train])
            val_index = np.concatenate([np.arange(tokens_before[i], tokens_before[i+1]) for i in val])
            
            X_train = X[train_index]
            X_val = X[val_index]
        
            y_train = y[train_index]
            y_val = y[val_index]
        
            model.fit(X_train, y_train)            

            tad_scores = []
            k = 0
            for idx in val:
                p_i = [np.exp(greedy_log_likelihoods[idx][0])]
                for j in range(1, len(greedy_log_likelihoods[idx])):                
                    pt_jm1 = p_i[-1]
                    ptt_j = np.exp(greedy_log_likelihoods[idx][j])
                    X_test_i = [attention_features[val_index[k]].tolist() + [ptt_j, np.exp(greedy_log_likelihoods[idx][j-1]), pt_jm1]]
                    y_pred_i = model.predict(X_test_i)[0]   
    
                    p_i_ = ptt_j*pt_jm1 + y_pred_i * (1 - pt_jm1)
                    p_i_ = np.clip(p_i_, 0, 1)
                    p_i.append(p_i_)
                    k += 1
                
                p_i = np.array(p_i)
                    
                tad_scores.append([])
                for claim in claims[idx]:
                    tokens = np.array(claim.aligned_token_ids)
                    claim_p_i = p_i[tokens]
                    tad_scores[-1].append(-claim_p_i.mean())

            factcheck_scores = np.concatenate([factcheck[i] for i in val])
            tad = np.concatenate(tad_scores)

            tad = tad[~np.isnan(factcheck_scores)]
            factcheck_scores = factcheck_scores[~np.isnan(factcheck_scores)]
            mse_cv.append(roc_auc_score(factcheck_scores, tad))
    
        mse = np.mean(mse_cv)
        print(mse)
        if mse > best_mse:
            best_mse = mse
            best_params = param
    print("BEST:", best_params)
    return best_params


class TADClaim(Estimator):
    def __init__(
        self,
        regression_model: str = "LinReg",
        ignore_special_tokens: bool = False,
        aggregation: str = "mean",
        clip_y: int = 1,
        use_alignscore: bool = False,
        aggregated: bool = False,
        use_accuracy: bool = False,
        accuracy = None,
        use_comet: bool = False,
        comet = None,
        cross_val: bool = False,
        parameters_path: str = "",
        use_ccp: bool = False,
        ccp_context: str = "",
    ):

        self.use_ccp = use_ccp
        dependencies = ["attention_features", "greedy_log_likelihoods", "train_attention_features", "train_greedy_log_likelihoods", "train_greedy_tokens", "train_target_texts", 
                        "claims", "train_claims", "train_input_texts"]
        if self.use_ccp:
            dependencies += ["greedy_tokens", "greedy_tokens_alternatives", "greedy_tokens_alternatives_nli", "greedy_tokens_alternatives_fact_pref_nli", 
                             "train_greedy_tokens_alternatives", "train_greedy_tokens_alternatives_nli", "train_greedy_tokens_alternatives_fact_pref_nli"]
            
        super().__init__(dependencies, "claim")
        
        self.ignore_special_tokens = ignore_special_tokens
        self.aggregation = aggregation
        self.clip_y = clip_y
        self.regression_model_name = regression_model
        self.use_alignscore = use_alignscore
        self.use_accuracy = use_accuracy
        self.use_comet = use_comet
        self.cross_val = cross_val
        self.factcheck = OpenAIFactCheck(openai_model="gpt-4o")
        
        if self.use_ccp:
            self.ccp_context = ccp_context
            if len(ccp_context):
                self.CCP = CCPClaim_token(nli_context=ccp_context, is_tad=True)
            else:
                self.CCP = CCP_token()

        if len(parameters_path):
            ccp = f"+{str(self.CCP)}" if self.use_ccp else ""
            self.full_path = f"{parameters_path}/tad_{ccp}_{self.regression_model_name}_{self.aggregation}"
            os.makedirs(self.full_path, exist_ok=True)
        
        if self.use_alignscore:
            self.scorer = alignscorer
            if aggregated:
                self.scorer = AggregatedMetric(base_metric=self.scorer)

        if self.use_accuracy:
            self.scorer = accuracy
            if aggregated:
                self.scorer = AggregatedMetric(base_metric=self.scorer) 
        if self.use_comet:
            self.scorer = comet
            if aggregated:
                self.scorer = AggregatedMetric(base_metric=self.scorer) 
            
        if self.regression_model_name == "LinReg":
            self.regression_model = Ridge(alpha=1)
            self.model_init = lambda param: Ridge(alpha=param[0])
            self.params = {
                "alpha": [1e+3, 1e+2, 1e+1, 1, 1e-1, 1e-2, 1e-3, 1e-4],
            }
        elif self.regression_model_name == "CatBoost":            
            self.regression_model = CatBoostRegressor(iterations=200, 
                                                      loss_function="RMSE",
                                                      learning_rate=1e-2,
                                                      depth=5,
                                                      logging_level='Silent')
            self.model_init = lambda param: CatBoostRegressor(iterations=param[0], 
                                                      loss_function="RMSE",
                                                      learning_rate=param[1],
                                                      depth=param[2],
                                                      logging_level='Silent')
            self.params = {
                "iterations": [100, 200],
                "learning_rate": [1e-1, 1e-2],
                "depth": [3, 5]
            }
        elif self.regression_model_name == "MLP":
            self.regression_model = MLP()
            self.model_init = lambda param: MLP(n_layers=param[0], 
                                                n_epochs=param[1],
                                                lr=param[2],
                                                dropout=param[3],
                                                batch_size=param[4],
                                                n_features=param[5])
            self.params = {
                "n_layers": [2, 4],
                "n_epochs": [10, 20, 30],
                "lr": [1e-5, 3e-5, 5e-5],
                "dropout": [0],
                "batch_size": [64, 128],
                "n_features": [1603]
            }

        self.is_fitted = False

    def __str__(self):
        use_alignscore = ", +AlignScore" if self.use_alignscore else ""
        use_accuracy = ", +Accuracy" if self.use_accuracy else ""
        use_comet = ", +Comet" if self.use_comet else ""
        cross_val = ", +cross_val" if self.cross_val else ""
        ccp = f"+{str(self.CCP)}" if self.use_ccp else ""
        if self.ignore_special_tokens:
            return f"TADClaim{ccp} ({self.regression_model_name}, {self.aggregation}, {self.clip_y}{use_alignscore}{use_accuracy}{use_comet}{cross_val})"
        return f"TADClaim{ccp} ({self.regression_model_name}, all, {self.aggregation}, {self.clip_y}{use_alignscore}{use_accuracy}{use_comet}{cross_val})"


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

    def __call__(self, stats: Dict[str, np.ndarray], key="greedy_log_likelihoods") -> np.ndarray:
        # take the embeddings
        attention_features = stats["attention_features"]
        greedy_log_likelihoods = stats[key]

        # compute centroids if not given
        if not self.is_fitted:
            train_attention_features = stats[f"train_attention_features"]
            train_greedy_log_likelihoods = stats[f"train_{key}"]
            train_greedy_tokens = stats[f"train_greedy_tokens"]
            train_input_texts = stats[f"train_input_texts"]
            train_claims = stats[f"train_claims"]

            train_stats = {"claims": train_claims, "input_texts": train_input_texts}
            
            factcheck = self.factcheck(train_stats, None, None)
            targets = self._get_targets(train_greedy_tokens, train_claims, factcheck)

            if self.use_ccp:
                if self.ccp_context == "fact_pref":
                    train_greedy_tokens_alternatives = stats[f"train_greedy_tokens_alternatives"]
                    train_greedy_tokens_alternatives_fact_pref_nli = stats[f"train_greedy_tokens_alternatives_fact_pref_nli"]
    
                    train_stats = {"greedy_tokens": train_greedy_tokens, 
                                   "greedy_tokens_alternatives": train_greedy_tokens_alternatives,
                                   "greedy_tokens_alternatives_fact_pref_nli": train_greedy_tokens_alternatives_fact_pref_nli,
                                   "claims": train_claims,}

                else:
                    train_greedy_tokens_alternatives = stats[f"train_greedy_tokens_alternatives"]
                    train_greedy_tokens_alternatives_nli = stats[f"train_greedy_tokens_alternatives_nli"]
    
                    train_stats = {"greedy_tokens": train_greedy_tokens, 
                                   "greedy_tokens_alternatives": train_greedy_tokens_alternatives,
                                   "greedy_tokens_alternatives_nli": train_greedy_tokens_alternatives_nli,
                                   "claims": train_claims,}
                
                train_greedy_log_likelihoods = self.CCP(train_stats, return_probs=True)

            if self.regression_model_name == "MLP":
                self.regression_model = MLP(n_features=train_attention_features.shape[-1] + 3)
                self.params["n_features"] = [train_attention_features.shape[-1] + 3]
            
            X, y = [], []
            k = 0
            for idx in range(len(train_input_texts)):
                p_i = [np.clip(targets[idx][0] + 1e-3, 0, 1)]
                X_i, y_i = [], []
                for j in range(1, len(train_greedy_log_likelihoods[idx])):
                    pt_j = np.clip(targets[idx][j] + 1e-3, 0, 1)
                    pt_jm1 = np.clip(p_i[-1], 0, 1)
                    ptt_j = np.exp(train_greedy_log_likelihoods[idx][j])
                    y_value = (pt_j - ptt_j*pt_jm1) / max((1 - pt_jm1), 1e-1)
                    y_value = np.clip(y_value, -self.clip_y, self.clip_y)
                    p_i_ = ptt_j*pt_jm1 + y_value * (1 - pt_jm1)

                    p_i.append(p_i_)
                    y_i.append(y_value)
                    X_i.append(train_attention_features[k].tolist() + [ptt_j, np.exp(train_greedy_log_likelihoods[idx][j-1]), pt_jm1])
                    k+=1

                if len(X_i):
                    y.append(y_i)
                    X.append(X_i)
                
            X = np.concatenate(X)
            y = np.concatenate(y)
            if not self.cross_val:
                self.regression_model.fit(X, y)
            else:
                best_params = cross_val_hp_claim(X, y, self.model_init, self.params, train_attention_features, train_greedy_log_likelihoods, stats["train_claims"], targets, factcheck)
                self.regression_model = self.model_init(best_params)
                self.regression_model.fit(X, y) 
                with open(f'{self.full_path}/best_params.json', 'w') as fp:
                    json.dump(best_params, fp)
            self.is_fitted = True
            
        greedy_tokens = stats["greedy_tokens"]
        claims = stats["claims"]

        if self.use_ccp:
            if self.ccp_context == "fact_pref":
                greedy_tokens_alternatives = stats[f"greedy_tokens_alternatives"]
                greedy_tokens_alternatives_fact_pref_nli = stats[f"greedy_tokens_alternatives_fact_pref_nli"]

                ccp_stats = {"greedy_tokens": greedy_tokens, 
                             "greedy_tokens_alternatives": greedy_tokens_alternatives,
                             "greedy_tokens_alternatives_fact_pref_nli": greedy_tokens_alternatives_fact_pref_nli,
                             "claims": claims,}
            else:
                greedy_tokens_alternatives = stats[f"greedy_tokens_alternatives"]
                greedy_tokens_alternatives_nli = stats[f"greedy_tokens_alternatives_nli"]
            
                ccp_stats = {"greedy_tokens": greedy_tokens, 
                             "greedy_tokens_alternatives": greedy_tokens_alternatives,
                             "greedy_tokens_alternatives_nli": greedy_tokens_alternatives_nli,
                             "claims": claims,}
            greedy_log_likelihoods = self.CCP(ccp_stats, return_probs=True)
        
        tad_scores = []
        k = 0
        for idx in range(len(greedy_log_likelihoods)):
            p_i = [np.exp(greedy_log_likelihoods[idx][0])]
            for j in range(1, len(greedy_log_likelihoods[idx])):                
                pt_jm1 = p_i[-1]
                ptt_j = np.exp(greedy_log_likelihoods[idx][j])
                X_test_i = [attention_features[k].tolist() + [ptt_j, np.exp(greedy_log_likelihoods[idx][j-1]), pt_jm1]]
                y_pred_i = self.regression_model.predict(X_test_i)[0]   

                p_i_ = ptt_j*pt_jm1 + y_pred_i * (1 - pt_jm1)
                p_i_ = np.clip(p_i_, 0, 1)
                p_i.append(p_i_)
                k += 1
            
            tokens = np.array(greedy_tokens[idx])
            if self.ignore_special_tokens:
                p_i = np.array(p_i)[~ignore_idx]
            else:
                p_i = np.array(p_i)
                
            tad_scores.append([])
            for claim in claims[idx]:
                tokens = np.array(claim.aligned_token_ids)
                claim_p_i = p_i[tokens]
                
                if self.aggregation == "mean":
                    tad_scores[-1].append(-claim_p_i.mean())
                elif self.aggregation  == "sum(log(p_i))":
                    tad_scores[-1].append(-np.log(claim_p_i + 1e-5).sum())
                elif self.aggregation  == "mean(log(p_i))":
                    tad_scores[-1].append(-np.log(claim_p_i + 1e-5).mean())

        return tad_scores