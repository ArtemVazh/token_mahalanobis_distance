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
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from optuna.samplers import TPESampler
from copy import deepcopy

NAMING_MAP = {"bert-base-uncased": "bert_base", 
              "bert-large-uncased": "bert_large", 
              "google/electra-small-discriminator": "electra_base", 
              "roberta-base": "roberta_base", 
              "roberta-large": "roberta_large",
              "meta-llama/Llama-3.2-1B": "llama1b", 
              "meta-llama/Llama-3.2-3B": "llama3b", 
              "meta-llama/Llama-3.1-8B": "llama8b"}

def preprocess_function(
    sentence1_key, sentence2_key, tokenizer, max_seq_length, examples
):
    # Tokenize the texts
    args = (
        (examples[sentence1_key],)
        if sentence2_key is None
        else (examples[sentence1_key], examples[sentence2_key])
    )
    result = tokenizer(
        *args, padding="max_length", max_length=max_seq_length, truncation=True
    )

    # Map labels to IDs (not necessary for GLUE tasks)
    if "label" in examples:
        result["label"] = [
            l for l in examples["label"]
        ]
    return result

def hp_space_discrete(trial):
    return {
        "learning_rate": trial.suggest_categorical(
            "learning_rate",
            [5e-6, 6e-6, 7e-6, 9e-6, 1e-5, 2e-5, 3e-5, 5e-5, 7e-5, 1e-4],
        ),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 15),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", [8, 16, 32, 64]
        ),
        "weight_decay": trial.suggest_categorical(
            "weight_decay", [0, 0.01, 0.1]
        ), 
    }


def get_optimal_hyperparameters(
    trainer,
    model_init,
):
    # To avoid overriding
    trainer_hyp_opt = trainer
    trainer_hyp_opt.model_init = model_init

    def compute_objective(metrics):
        return metrics["eval_rmse"]

    seed = trainer.args.seed
    sampler = TPESampler(seed=seed)
    hyp_opt_result = trainer_hyp_opt.hyperparameter_search(
        direction="minimize",
        hp_space=hp_space_discrete,
        backend="optuna",
        compute_objective=compute_objective,
        n_trials=10,
        sampler=sampler,
    )
    print(f"Optimal hyperparameters: {hyp_opt_result.hyperparameters}")
    print(f"Optimal metric value: {hyp_opt_result.objective}")

    result = hyp_opt_result.hyperparameters
    result.update({"objective": hyp_opt_result.objective})

    return result


class TransformerUQ(Estimator):
    def __init__(
        self,
        metric = None,
        metric_name: str = "",
        aggregated: bool = False,
        device: str = "cuda",
        model_name: str = "roberta-base",
        max_seq_length: int = 512,
    ):
        self.model_name = NAMING_MAP[model_name] 
        self.orig_model_name = model_name
        super().__init__(["train_greedy_texts", "train_target_texts", "greedy_texts"], "sequence")
        
        self.metric_name = metric_name
        self.device = device
        self.aggregated = aggregated
        self.max_seq_length = max_seq_length
        self.is_fitted = False
        if metric is not None:
            self.metric = metric
            if aggregated:
                self.metric = AggregatedMetric(base_metric=self.metric)


    def __str__(self):
        return f"TransformerUQ_{self.model_name} ({self.metric_name})"

    def __call__(self, stats: Dict[str, np.ndarray], save_data: bool = True) -> np.ndarray:
        # compute centroids if not given
        if not self.is_fitted:
            train_input_texts = stats[f"train_input_texts"]
            train_greedy_texts = stats[f"train_greedy_texts"]
            train_target_texts = stats[f"train_target_texts"]
            
            metric_key = f"train_{self.metric_name}_{len(train_greedy_texts)}"
            if metric_key in stats.keys():
                self.train_seq_metrics = stats[metric_key]
            else:
                metrics = []
                for x, y in zip(train_greedy_texts, train_target_texts):
                    if isinstance(y, list) and (not self.aggregated):
                        y_ = y[0]
                    elif isinstance(y, str) and (self.aggregated):
                        y_ = [y]
                    else:
                        y_ = y
                    metrics.append([self.metric({"greedy_texts": [x], "target_texts": [y_]}, [y_])[0]])
                    
                self.train_seq_metrics = np.concatenate(metrics).astype(float)
                stats[metric_key] = self.train_seq_metrics

            self.tokenizer = AutoTokenizer.from_pretrained(self.orig_model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.orig_model_name, num_labels=1)
            model_copy = deepcopy(self.model)
            model_init = lambda x: deepcopy(model_copy)
    
            self.model.to(self.device)

            df = pd.DataFrame({"text":[inp+out for inp,out in zip(train_input_texts, train_greedy_texts)], 
                               "labels":self.train_seq_metrics})

            train_idx, dev_idx = train_test_split(list(range(len(train_greedy_texts))), test_size=0.3, shuffle=True, random_state=42)
            train_dataset = Dataset.from_pandas(df.iloc[train_idx])
            eval_dataset = Dataset.from_pandas(df.iloc[dev_idx])
            dataset = Dataset.from_pandas(df)
                
            f_preprocess = lambda examples: preprocess_function(
                "text", None, self.tokenizer, self.max_seq_length, examples
            )

            train_dataset = train_dataset.map(
                f_preprocess,
                batched=True,
                load_from_cache_file=False,
            )
            eval_dataset = eval_dataset.map(
                f_preprocess,
                batched=True,
                load_from_cache_file=False,
            )
            dataset = dataset.map(
                f_preprocess,
                batched=True,
                load_from_cache_file=False,
            )
        
            def compute_metrics(eval_pred):
                predictions, labels = eval_pred
                rmse = root_mean_squared_error(labels, predictions)
                return {"rmse": rmse}

            training_args = TrainingArguments(output_dir="test_trainer",
                                              logging_strategy="epoch",
                                              eval_strategy="epoch",
                                              seed=42,
                                              learning_rate=5e-5,
                                              weight_decay=1e-1,
                                              num_train_epochs=5,
                                              gradient_accumulation_steps=1,
                                              per_device_train_batch_size=64,
                                              per_device_eval_batch_size=64,
                                              label_names=["labels"],
                                              report_to="none",)
            
            training_args.warmup_steps = int(
                0.1
                * len(train_dataset)
                * training_args.num_train_epochs
                / training_args.per_device_train_batch_size
            )
            trainer = Trainer(
                args = training_args,
                model = model_copy,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=compute_metrics,
            )

            results = get_optimal_hyperparameters(
                trainer,
                model_init,
            )
            training_args = TrainingArguments(output_dir="test_trainer",
                                              logging_strategy="epoch",
                                              seed=42,
                                              learning_rate=results['learning_rate'],
                                              weight_decay=results['weight_decay'],
                                              num_train_epochs=results['num_train_epochs'],
                                              gradient_accumulation_steps=1,
                                              per_device_train_batch_size=results['per_device_train_batch_size'],
                                              per_device_eval_batch_size=64,
                                              label_names=["labels"],
                                              report_to="none",)
            trainer = Trainer(
                args = training_args,
                model = self.model,
                train_dataset=dataset,
                eval_dataset=None,
                compute_metrics=None,
            )
            trainer.train()
            self.is_fitted = True

        input_texts = stats[f"input_texts"]
        greedy_texts = stats[f"greedy_texts"]
        texts = [inp+out for inp,out in zip(input_texts, input_texts)]
        batch = self.tokenizer(texts, padding="max_length", max_length=self.max_seq_length, truncation=True, return_tensors='pt')
        batch = {k: v.to(self.model.device) for k, v in batch.items()}
        uq = -self.model(**batch).logits.cpu().detach().numpy().flatten()
        
        return uq