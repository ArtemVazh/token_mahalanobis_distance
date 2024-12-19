#!/usr/bin/env python3

import hydra
import importlib
import os
import torch
import transformers
import argparse
from pathlib import Path
import json
import copy
import codecs
import numpy as np
import logging

log = logging.getLogger()

from lm_polygraph.utils.manager import UEManager
from dataset import Dataset
from lm_polygraph.utils.model import WhiteboxModel, create_ensemble
from lm_polygraph.utils.processor import Logger
from lm_polygraph.generation_metrics.accuracy import AccuracyMetric
from lm_polygraph.generation_metrics.bart_score import BartScoreSeqMetric
from lm_polygraph.generation_metrics.rouge import RougeMetric
from lm_polygraph.generation_metrics.bert_score import BertScoreMetric
from lm_polygraph.generation_metrics.sbert import SbertMetric
from lm_polygraph.generation_metrics.aggregated_metric import AggregatedMetric
from lm_polygraph.generation_metrics.alignscore import AlignScore
from lm_polygraph.generation_metrics.comet import Comet
from lm_polygraph.utils.openai_chat import OpenAIChat
from lm_polygraph.generation_metrics.openai_fact_check import OpenAIFactCheck
from lm_polygraph.estimators import *
from lm_polygraph.estimators.ensemble_token_measures import all_token_estimators
from lm_polygraph.estimators.ensemble_sequence_measures import all_ep_estimators, all_pe_estimators
from lm_polygraph.estimators.ensemble_token_measures import *
from lm_polygraph.ue_metrics import *

from token_mahalanobis_distance import TokenMahalanobisDistance, TokenMahalanobisDistanceClaim
from token_knn import TokenKNN
from average_token_mahalanobis_distance import LinRegTokenMahalanobisDistance, LinRegTokenMahalanobisDistance_Claim
from average_token_mahalanobis_distance_hybrid import LinRegTokenMahalanobisDistance_Hybrid, LinRegTokenMahalanobisDistance_Hybrid_Claim
from relative_token_mahalanobis_distance import RelativeTokenMahalanobisDistance, RelativeTokenMahalanobisDistanceClaim
from saplma import SAPLMA, SAPLMAClaim, SAPLMA_truefalse
from saplma_meta import SAPLMA_meta
from factoscope import LLMFactoscope, LLMFactoscopeAll
from eigenscore import EigenScore
from huq_msp_lrtmd import HUQ_LRTMD, HUQ_LRTMD_Claim
from SATRMD import StableTokenMahalanobisDistance

from alignscore import AlignScore as AlignScoreNew


hydra_config = Path(os.environ["HYDRA_CONFIG"])


@hydra.main(
    version_base=None,
    config_path=str(hydra_config.parent),
    config_name=str(hydra_config.name),
)
def main(args):
    save_path = os.getcwd()
    log.info(f"Main directory: {save_path}")
    os.chdir(hydra.utils.get_original_cwd())

    save_path = args.save_path if "save_path" in args else save_path

    if args.seed is None or len(args.seed) == 0:
        args.seed = [1]

    model_kwargs = get_model_kwargs(args)

    cache_kwargs = {}
    if os.environ.get('HF_DATASETS_OFFLINE', '').strip() == '1':
        cache_kwargs = {'cache_dir': args.cache_path}

    for seed in args.seed:
        log.info("=" * 100)
        log.info(f"SEED: {seed}")

        log.info(f"Loading model {args.model.path}...")
        transformers.set_seed(seed)
        
        model = WhiteboxModel.from_pretrained(
            args.model.path,
            getattr(args, "generation_params", {}),
            device_map=args.model.device_map,
            add_bos_token=getattr(args.model, "add_bos_token", True),
            **cache_kwargs,
            **model_kwargs,
        )
        
        if args.model.ensemble:
            # Only MC-ensembles for now
            log.info(f"Creating ensemble...")
            ensemble_model = create_ensemble(model_paths=[args.model.path],
                                             mc=True,
                                             seed=args.seed[0],
                                             ensembling_mode=args.model.ensembling_mode,
                                             mc_seeds=args.model.mc_seeds,
                                             dropout_rate=float(args.model.dropout_rate),
                                             **cache_kwargs,
                                             **model_kwargs
                                             )
        else:
            ensemble_model = None

        log.info("Done with loading model.")

        log.info(f"Loading dataset {args.dataset}...")
        dataset = Dataset.load(
            args.dataset,
            args.text_column,
            args.label_column,
            batch_size=args.batch_size,
            prompt=args.prompt,
            description=getattr(args, "description", ""),
            mmlu_max_subject_size=getattr(args, "mmlu_max_subject_size", 100),
            n_shot=getattr(args, "n_shot", 5),
            few_shot_split=getattr(args, "few_shot_split", "train"),
            split=args.eval_split,
            load_from_disk=args.load_from_disk,
            max_new_tokens=getattr(args, f"max_new_tokens", 100),
            **cache_kwargs
        )

        lens = np.array([len(y) for y in dataset.y])
        print("empty", (lens==0).sum())
        if getattr(args, "eval_dataset_1", False):
            if args.subsample_eval_dataset != -1:
                dataset.subsample(args.subsample_eval_dataset, seed=seed)
            k_ds = 1
            while getattr(args, f"eval_dataset_{k_ds}", False):
                if getattr(args, f"eval_dataset_{k_ds}") == getattr(args, f"dataset"):
                    k_ds += 1
                    continue
                eval_dataset_k = Dataset.load(
                    getattr(args, f"eval_dataset_{k_ds}"),
                    getattr(args, f"eval_text_column_{k_ds}"),
                    getattr(args, f"eval_label_column_{k_ds}"),
                    batch_size=args.batch_size,
                    prompt=codecs.decode(getattr(args, f"eval_prompt_{k_ds}"), 'unicode_escape'),
                    description=codecs.decode(getattr(args, f"eval_description_{k_ds}", ""), 'unicode_escape'),
                    mmlu_max_subject_size=getattr(args, "mmlu_max_subject_size", 100),
                    n_shot=getattr(args, f"eval_n_shot_{k_ds}", 5),
                    few_shot_split=getattr(args, f"eval_few_shot_split_{k_ds}", "train"),
                    split=getattr(args, f"eval_split_{k_ds}", "train"),
                    max_new_tokens=getattr(args, f"eval_max_new_tokens_{k_ds}", 100),
                    size=10_000,
                    load_from_disk=args.load_from_disk,
                    **cache_kwargs
                )
                if args.subsample_eval_dataset != -1:
                    eval_dataset_k.subsample(args.subsample_eval_dataset, seed=seed)

                lens = np.array([len(y) for y in eval_dataset_k.y])
                print("empty1", (lens==0).sum())
                
                if getattr(args, "multiref", False):
                    if isinstance(eval_dataset_k.y[0], list):
                        dataset.concat(eval_dataset_k.x, eval_dataset_k.y, eval_dataset_k.max_new_tokens)
                    else:
                        dataset.concat(eval_dataset_k.x, [[y] for y in eval_dataset_k.y], eval_dataset_k.max_new_tokens)
                else:
                    if isinstance(eval_dataset_k.y[0], list):
                        dataset.concat(eval_dataset_k.x, [y[0] for y in eval_dataset_k.y], eval_dataset_k.max_new_tokens)
                    else:
                        dataset.concat(eval_dataset_k.x, eval_dataset_k.y, eval_dataset_k.max_new_tokens)
                k_ds += 1
                lens = np.array([len(y) for y in dataset.y])
                print("empty2", (lens==0).sum())
                
        estimators = []
        estimators += get_ue_methods(args, model)
        density_based_ue_methods = get_density_based_ue_methods(args, model.model_type)
        estimators += density_based_ue_methods

        train_dataset = None
        background_train_dataset = None
        if any([not getattr(method, "is_fitted", True) for method in estimators]) and (not getattr(args, "kfolds", False)):
            if (args.train_dataset is not None) and (
                    args.train_dataset != args.dataset
            ):
                train_dataset = Dataset.load(
                    args.train_dataset,
                    args.text_column,
                    args.label_column,
                    batch_size=args.batch_size,
                    prompt=args.prompt,
                    description=getattr(args, "description", ""),
                    mmlu_max_subject_size=getattr(args, "mmlu_max_subject_size", 100),
                    n_shot=getattr(args, "n_shot", 5),
                    few_shot_split=getattr(args, "few_shot_split", "train"),
                    split=args.train_split,
                    size=10_000,
                    load_from_disk=args.load_from_disk,
                    max_new_tokens=getattr(args, f"max_new_tokens", 100),
                    **cache_kwargs
                )
            elif args.train_test_split:
                X_train, X_test, y_train, y_test, max_new_tokens_train, max_new_tokens_test = dataset.train_test_split(
                    test_size=args.test_split_size, seed=seed, split=args.eval_split
                )
                train_dataset = Dataset(
                    x=X_train, y=y_train, max_new_tokens=getattr(args, "max_new_tokens", 100), batch_size=args.batch_size
                )
            else:
                train_dataset = Dataset.load(
                    args.dataset,
                    args.text_column,
                    args.label_column,
                    batch_size=args.batch_size,
                    prompt=args.prompt,
                    description=getattr(args, "description", ""),
                    mmlu_max_subject_size=getattr(args, "mmlu_max_subject_size", 100),
                    n_shot=getattr(args, "n_shot", 5),
                    few_shot_split=getattr(args, "few_shot_split", "train"),
                    split=args.train_split,
                    size=10_000,
                    load_from_disk=args.load_from_disk,
                    max_new_tokens=getattr(args, f"max_new_tokens", 100),
                    **cache_kwargs
                )
            if args.subsample_train_dataset != -1:
                train_dataset.subsample(args.subsample_train_dataset, seed=seed)
            
            if getattr(args, "train_dataset_1", False):

                k_ds = 1
                train_dataset = None
                while getattr(args, f"train_dataset_{k_ds}", False):
                    train_dataset_k = Dataset.load(
                        getattr(args, f"train_dataset_{k_ds}"),
                        getattr(args, f"train_text_column_{k_ds}"),
                        getattr(args, f"train_label_column_{k_ds}"),
                        batch_size=args.batch_size,
                        prompt=codecs.decode(getattr(args, f"train_prompt_{k_ds}"), 'unicode_escape'),
                        description=codecs.decode(getattr(args, f"train_description_{k_ds}", ""), 'unicode_escape'),
                        mmlu_max_subject_size=getattr(args, "mmlu_max_subject_size", 100),
                        n_shot=getattr(args, f"train_n_shot_{k_ds}", 5),
                        few_shot_split=getattr(args, f"few_shot_split_{k_ds}", "train"),
                        split=getattr(args, f"train_split_{k_ds}", "train"),
                        max_new_tokens=getattr(args, f"max_new_tokens_{k_ds}", 100),
                        size=10_000,
                        load_from_disk=args.load_from_disk,
                        **cache_kwargs
                    )
                    k_ds += 1
                    if args.subsample_train_dataset != -1:
                        train_dataset_k.subsample(args.subsample_train_dataset, seed=seed)
                    
                    if train_dataset is None:
                        train_dataset = train_dataset_k
                    else:
                        train_dataset.concat(train_dataset_k.x, train_dataset_k.y, train_dataset_k.max_new_tokens)

        if any([not getattr(method, "is_fitted", False) for method in estimators]):
            background_train_dataset = Dataset.load(
                args.background_train_dataset,
                args.background_train_dataset_text_column,
                args.background_train_dataset_label_column,
                batch_size=args.batch_size,
                data_files=args.background_train_dataset_data_files,
                split="train",
                size=100_000,
                load_from_disk=args.background_load_from_disk,
                **cache_kwargs
            )
            if args.subsample_background_train_dataset != -1:
                background_train_dataset.subsample(
                    args.subsample_background_train_dataset, seed=seed
                )            

        if not getattr(args, "eval_dataset_1", False):
            if args.subsample_eval_dataset != -1:
                dataset.subsample(args.subsample_eval_dataset, seed=seed)

        log.info("Done with loading data.")
        generation_metrics = get_generation_metrics(args)
        ue_metrics = get_ue_metrics(args)
        
        if getattr(args, "cherrypick", False):
            x = []
            y = []
            for x_i, y_i in zip(dataset.x, dataset.y):
                if "that is not an official language of the U.S." in x_i:
                    x.append(x_i)
                    y.append(y_i)
            dataset.x = x
            dataset.y = y

        if getattr(args, "crossval_claim_ue", False):
            
            train_idx, test_idx = dataset.kfolds(seed=seed, n_folds=getattr(args, "kfolds", 2))
            for f_idx in range(len(train_idx)):
                train_ds = copy.deepcopy(dataset).select(train_idx[f_idx])
                eval_ds = copy.deepcopy(dataset).select(test_idx[f_idx])

                estimators = []
                estimators += get_ue_methods(args, model)
                density_based_ue_methods = get_density_based_ue_methods(args, model.model_type)
                estimators += density_based_ue_methods
        
                man = UEManager(
                    eval_ds,
                    model,
                    estimators,
                    generation_metrics,
                    ue_metrics,
                    [
                        Logger(),
                    ],
                    deberta_batch_size=getattr(args, 'deberta_batch_size', 10),
                    train_data=train_ds,
                    ignore_exceptions=args.ignore_exceptions,
                    background_train_data=background_train_dataset,
                    max_new_tokens=args.max_new_tokens,
                    ensemble_model=ensemble_model
                )
                man()
                man.save(save_path + f"/ue_manager_seed{seed}_fold{f_idx}")
        else:
            man = UEManager(
                dataset,
                model,
                estimators,
                generation_metrics,
                ue_metrics,
                [
                    Logger(),
                ],
                deberta_batch_size=getattr(args, 'deberta_batch_size', 10),
                train_data=train_dataset,
                ignore_exceptions=args.ignore_exceptions,
                background_train_data=background_train_dataset,
                max_new_tokens=args.max_new_tokens,
                ensemble_model=ensemble_model
            )
    
            man()
    
            man.save(save_path + f"/ue_manager_seed{seed}")

def get_ue_metrics(args):
    ue_metrics = [
        #ReversedPairsProportion(),
        PredictionRejectionArea(),
        #RiskCoverageCurveAUC(),
    ]
    if getattr(args, "use_claim_ue", False) or getattr(args, "train_claim_pi", False):
        ue_metrics += [
            ROCAUC(),
            PRAUC(),
        ]
    return ue_metrics


def get_density_based_ue_methods(args, model_type):
    estimators = []
    if args.use_density_based_ue:
        accuracy = AccuracyMetric(
                target_ignore_regex = getattr(args, "target_ignore_regex", None),
                output_ignore_regex = getattr(args, "output_ignore_regex", None),
                normalize = getattr(args, "normalize", False),
            )
        rougel = RougeMetric("rougeL")
        if getattr(args, "run_proposed_methods", False):
            if (args.task == "qa") and (args.dataset not in ["keivalya/MedQuad-MedicalQnADataset", "bigbio/pubmed_qa", ['truthful_qa', 'generation']]):
                if getattr(args, "is_ood", False):
                    alignscorer = AlignScoreNew(return_mean=True, batch_size=1)
                    metrics = [alignscorer]
                    metrics_names = ["AlignScore"]
                else:
                    metrics = [accuracy]
                    metrics_names = ["Accuracy"]
            else:
                alignscorer = AlignScoreNew(return_mean=True, batch_size=1)
                metrics = [alignscorer]
                metrics_names = ["AlignScore"]
    
        layers = getattr(args, "layers", [0, -1])
        metric_thrs = getattr(args, "metric_thrs", [0.3])            
    
        if getattr(args, 'parameters_path', False):
            parameters_path = args.parameters_path
        else:
            dataset_name = args.dataset if isinstance(args.dataset, str) else '_'.join(args.dataset)
            dataset_name = dataset_name.split("/")[-1].split(".")[0]
            model_name = args.model.path.split("/")[-1]
            parameters_path = f"{args.cache_path}/density_stats/{dataset_name}/{model_name}"
        
        if model_type == "Seq2SeqLM":
            estimators += [
                MahalanobisDistanceSeq("encoder", parameters_path=parameters_path),
                MahalanobisDistanceSeq("decoder", parameters_path=parameters_path),
                RelativeMahalanobisDistanceSeq(
                    "encoder", parameters_path=parameters_path
                ),
                RelativeMahalanobisDistanceSeq(
                    "decoder", parameters_path=parameters_path
                ),
                RDESeq("encoder", parameters_path=parameters_path),
                RDESeq("decoder", parameters_path=parameters_path),
                PPLMDSeq("encoder", md_type="MD", parameters_path=parameters_path),
                PPLMDSeq("encoder", md_type="RMD", parameters_path=parameters_path),
                PPLMDSeq("decoder", md_type="MD", parameters_path=parameters_path),
                PPLMDSeq("decoder", md_type="RMD", parameters_path=parameters_path),
            ]
        else:
            if getattr(args, "run_baselines", False):
                estimators += [
                    PPLMDSeq("decoder", md_type="MD", parameters_path=parameters_path, storage_device=getattr(args, "md_device", "cpu")),
                    PPLMDSeq("decoder", md_type="RMD", parameters_path=parameters_path, storage_device=getattr(args, "md_device", "cpu")),
                ]

                if getattr(args, "use_truefalse_dataset", False):
                    for layer in layers:
                        estimators += [
                                SAPLMA_truefalse("decoder", parameters_path=None, aggregated=False, hidden_layer=layer, device=getattr(args, "md_device", "cpu")),
                            ]

            if getattr(args, "run_baselines", False) or getattr(args, "run_layerwise_methods", False):
                #layer-wise methods
                for layer in layers:
                    estimators += [
                        MahalanobisDistanceSeq("decoder", parameters_path=None, hidden_layer=layer, storage_device=getattr(args, "md_device", "cpu")),
                        RelativeMahalanobisDistanceSeq("decoder", parameters_path=None, hidden_layer=layer, storage_device=getattr(args, "md_device", "cpu")),
                        
                        TokenMahalanobisDistance("decoder", parameters_path=None, hidden_layer=layer, storage_device=getattr(args, "md_device", "cpu")),
                        RelativeTokenMahalanobisDistance("decoder", parameters_path=None, hidden_layer=layer, storage_device=getattr(args, "md_device", "cpu")),
                    ]
                    if getattr(args, "run_rde", True):
                        estimators += [
                            RDESeq("decoder", parameters_path=None, hidden_layer=layer),
                        ]
                        
                    if getattr(args, "run_eigenscore", True):
                        estimators += [
                            # EigenScore(hidden_layer=layer),
                            EigenScore("sample_embeddings_last_token", hidden_layer=layer),
                        ]
                    
            if getattr(args, "run_proposed_methods", False):
                # #layer-wise methods
                # for layer in layers:
                #     for metric, metric_name in zip(metrics, metrics_names):
                #         # estimators += [SAPLMA("decoder", parameters_path=None, metric=metric, metric_name=metric_name, aggregated=getattr(args, "multiref", False), hidden_layer=layer, cv_hp=True)]
                #         for thr in metric_thrs:
                #             estimators += [TokenMahalanobisDistance("decoder", parameters_path=None, metric=metric, metric_name=metric_name, aggregated=getattr(args, "multiref", False), hidden_layer=layer, metric_thr=thr, storage_device=getattr(args, "clean_md_device", "cpu")),
                #                            RelativeTokenMahalanobisDistance("decoder", parameters_path=None, metric=metric, metric_name=metric_name, aggregated=getattr(args, "multiref", False), hidden_layer=layer, metric_thr=thr, storage_device=getattr(args, "clean_md_device", "cpu"))]
            
                # meta methods
                for metric, metric_name in zip(metrics, metrics_names):
                    # estimators += [SAPLMA_meta("decoder", parameters_path=None, metric=metric, metric_name=metric_name, aggregated=getattr(args, "multiref", False), hidden_layer=layers, device="cuda", cv_hp=True)]
                    for thr in metric_thrs:
                        estimators += [
                                    LLMFactoscopeAll(metric=metric, metric_name=metric_name, metric_thr=thr, hidden_layers=layers, return_dist=True, return_new_dist=True),
                                    
                                    # ####################### satrmd
                                    # LinRegTokenMahalanobisDistance("decoder", parameters_path=None, 
                                    #                                 metric=metric, metric_name=metric_name, metric_md=metric, metric_md_name=metric_name,
                                    #                                 aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=thr,
                                    #                                 ue="TokenMahalanobis", positive=False, meta_model="LinReg", norm="orig", remove_corr=True, remove_alg=3, storage_device=getattr(args, "clean_md_device", "cpu")),
                                    
                                    # LinRegTokenMahalanobisDistance("decoder", parameters_path=None, 
                                    #                                 metric=metric, metric_name=metric_name, metric_md=metric, metric_md_name=metric_name,
                                    #                                 aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=thr,
                                    #                                 ue="RelativeTokenMahalanobis", positive=False, meta_model="LinReg", norm="orig", remove_corr=True, remove_alg=3, storage_device=getattr(args, "clean_md_device", "cpu")),
                                    
                                    # ####################### satrmd
                                    # # LinRegTokenMahalanobisDistance("decoder", parameters_path=None, 
                                    # #                                 metric=metric, metric_name=metric_name, metric_md=metric, metric_md_name=metric_name,
                                    # #                                 aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=thr,
                                    # #                                 ue="TokenMahalanobis", positive=False, meta_model="Lasso", norm="orig", remove_corr=True, remove_alg=3, storage_device=getattr(args, "clean_md_device", "cpu")),
                                    
                                    # # LinRegTokenMahalanobisDistance("decoder", parameters_path=None, 
                                    # #                                 metric=metric, metric_name=metric_name, metric_md=metric, metric_md_name=metric_name,
                                    # #                                 aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=thr,
                                    # #                                 ue="RelativeTokenMahalanobis", positive=False, meta_model="Lasso", norm="orig", remove_corr=True, remove_alg=3, storage_device=getattr(args, "clean_md_device", "cpu")),


                                    # ####################### weighted satrmd
                                    # # LinRegTokenMahalanobisDistance("decoder", parameters_path=None, 
                                    # #                                 metric=metric, metric_name=metric_name, metric_md=metric, metric_md_name=metric_name,
                                    # #                                 aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=thr,
                                    # #                                 ue="TokenMahalanobis", positive=False, meta_model="weights", norm="scaler", remove_corr=False, remove_alg=3, storage_device=getattr(args, "clean_md_device", "cpu")),
                                    
                                    # # LinRegTokenMahalanobisDistance("decoder", parameters_path=None, 
                                    # #                                 metric=metric, metric_name=metric_name, metric_md=metric, metric_md_name=metric_name,
                                    # #                                 aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=thr,
                                    # #                                 ue="RelativeTokenMahalanobis", positive=False, meta_model="weights", norm="scaler", remove_corr=False, remove_alg=3, storage_device=getattr(args, "clean_md_device", "cpu")),
                                    
                                    
                                    
                                    # ####################### nonzero stable satrmd by prr
                                    # # StableTokenMahalanobisDistance("decoder", parameters_path=None, 
                                    # #                                 metric=metric, metric_name=metric_name, metric_md=metric, metric_md_name=metric_name,
                                    # #                                 aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=thr,
                                    # #                                 ue="TokenMahalanobis", positive=True, meta_model="weights", norm="orig", remove_corr=False, remove_alg=3, storage_device=getattr(args, "clean_md_device", "cpu")),
                                    
                                    # # StableTokenMahalanobisDistance("decoder", parameters_path=None, 
                                    # #                                 metric=metric, metric_name=metric_name, metric_md=metric, metric_md_name=metric_name,
                                    # #                                 aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=thr,
                                    # #                                 ue="RelativeTokenMahalanobis", positive=True, meta_model="weights", norm="orig", remove_corr=False, remove_alg=3, storage_device=getattr(args, "clean_md_device", "cpu")),
                                    
                                    # ####################### stable satrmd by prr
                                    # StableTokenMahalanobisDistance("decoder", parameters_path=None, 
                                    #                                 metric=metric, metric_name=metric_name, metric_md=metric, metric_md_name=metric_name,
                                    #                                 aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=thr,
                                    #                                 ue="TokenMahalanobis", positive=False, meta_model="weights", norm="orig", remove_corr=False, remove_alg=3, storage_device=getattr(args, "clean_md_device", "cpu")),
                                    
                                    # StableTokenMahalanobisDistance("decoder", parameters_path=None, 
                                    #                                 metric=metric, metric_name=metric_name, metric_md=metric, metric_md_name=metric_name,
                                    #                                 aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=thr,
                                    #                                 ue="RelativeTokenMahalanobis", positive=False, meta_model="weights", norm="orig", remove_corr=False, remove_alg=3, storage_device=getattr(args, "clean_md_device", "cpu")),
                                    
                                    # ####################### nonzero stable satrmd+msp by prr
                                    # # StableTokenMahalanobisDistance("decoder", parameters_path=None, 
                                    # #                                 metric=metric, metric_name=metric_name, metric_md=metric, metric_md_name=metric_name,
                                    # #                                 aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=thr,
                                    # #                                 ue="TokenMahalanobis", positive=True, meta_model="weights", norm="orig", remove_corr=False, remove_alg=3, storage_device=getattr(args, "clean_md_device", "cpu"), add_msp=True),
                                    
                                    # # StableTokenMahalanobisDistance("decoder", parameters_path=None, 
                                    # #                                 metric=metric, metric_name=metric_name, metric_md=metric, metric_md_name=metric_name,
                                    # #                                 aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=thr,
                                    # #                                 ue="RelativeTokenMahalanobis", positive=True, meta_model="weights", norm="orig", remove_corr=False, remove_alg=3, storage_device=getattr(args, "clean_md_device", "cpu"), add_msp=True),
                                    
                                    # ####################### stable satrmd+msp by prr
                                    # StableTokenMahalanobisDistance("decoder", parameters_path=None, 
                                    #                                 metric=metric, metric_name=metric_name, metric_md=metric, metric_md_name=metric_name,
                                    #                                 aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=thr,
                                    #                                 ue="TokenMahalanobis", positive=False, meta_model="weights", norm="orig", remove_corr=False, remove_alg=3, storage_device=getattr(args, "clean_md_device", "cpu"), add_msp=True),
                                    
                                    # StableTokenMahalanobisDistance("decoder", parameters_path=None, 
                                    #                                 metric=metric, metric_name=metric_name, metric_md=metric, metric_md_name=metric_name,
                                    #                                 aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=thr,
                                    #                                 ue="RelativeTokenMahalanobis", positive=False, meta_model="weights", norm="orig", remove_corr=False, remove_alg=3, storage_device=getattr(args, "clean_md_device", "cpu"), add_msp=True),

                                    # ####################### linregstable satrmd by prr
                                    # # LinRegTokenMahalanobisDistance("decoder", parameters_path=None, 
                                    # #                                 metric=metric, metric_name=metric_name, metric_md=metric, metric_md_name=metric_name,
                                    # #                                 aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=thr,
                                    # #                                 ue="TokenMahalanobis", positive=False, meta_model="StableLinReg", norm="orig", remove_corr=True, remove_alg=3, storage_device=getattr(args, "clean_md_device", "cpu")),
                                    
                                    # # LinRegTokenMahalanobisDistance("decoder", parameters_path=None, 
                                    # #                                 metric=metric, metric_name=metric_name, metric_md=metric, metric_md_name=metric_name,
                                    # #                                 aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=thr,
                                    # #                                 ue="RelativeTokenMahalanobis", positive=False, meta_model="StableLinReg", norm="orig", remove_corr=True, remove_alg=3, storage_device=getattr(args, "clean_md_device", "cpu")),

                                    # ####################### weighted linregstable satrmd by prr
                                    # LinRegTokenMahalanobisDistance("decoder", parameters_path=None, 
                                    #                                 metric=metric, metric_name=metric_name, metric_md=metric, metric_md_name=metric_name,
                                    #                                 aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=thr,
                                    #                                 ue="TokenMahalanobis", positive=False, meta_model="WeightedStableLinReg", norm="orig", remove_corr=True, remove_alg=3, storage_device=getattr(args, "clean_md_device", "cpu")),
                                    
                                    # LinRegTokenMahalanobisDistance("decoder", parameters_path=None, 
                                    #                                 metric=metric, metric_name=metric_name, metric_md=metric, metric_md_name=metric_name,
                                    #                                 aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=thr,
                                    #                                 ue="RelativeTokenMahalanobis", positive=False, meta_model="WeightedStableLinReg", norm="orig", remove_corr=True, remove_alg=3, storage_device=getattr(args, "clean_md_device", "cpu")),



                                    # ####################### linregs satrmd+msp by prr
                                    # LinRegTokenMahalanobisDistance_Hybrid("decoder", parameters_path=None, 
                                    #                                         metric=metric, metric_name=metric_name, metric_md=metric, metric_md_name=metric_name,
                                    #                                         aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=thr,
                                    #                                         ue="TokenMahalanobis", positive=False, meta_model="LinReg", norm="orig", remove_corr=True, remove_alg=3, storage_device=getattr(args, "clean_md_device", "cpu")),
                                    
                                    # LinRegTokenMahalanobisDistance_Hybrid("decoder", parameters_path=None, 
                                    #                                         metric=metric, metric_name=metric_name, metric_md=metric, metric_md_name=metric_name,
                                    #                                         aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=thr,
                                    #                                         ue="RelativeTokenMahalanobis", positive=False, meta_model="LinReg", norm="orig", remove_corr=True, remove_alg=3, storage_device=getattr(args, "clean_md_device", "cpu")),
                                    # ####################### linregs satrmd+msp by prr
                                    # # LinRegTokenMahalanobisDistance_Hybrid("decoder", parameters_path=None, 
                                    # #                                         metric=metric, metric_name=metric_name, metric_md=metric, metric_md_name=metric_name,
                                    # #                                         aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=thr,
                                    # #                                         ue="TokenMahalanobis", positive=False, meta_model="Lasso", norm="orig", remove_corr=True, remove_alg=3, storage_device=getattr(args, "clean_md_device", "cpu")),
                                    
                                    # # LinRegTokenMahalanobisDistance_Hybrid("decoder", parameters_path=None, 
                                    # #                                         metric=metric, metric_name=metric_name, metric_md=metric, metric_md_name=metric_name,
                                    # #                                         aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=thr,
                                    # #                                         ue="RelativeTokenMahalanobis", positive=False, meta_model="Lasso", norm="orig", remove_corr=True, remove_alg=3, storage_device=getattr(args, "clean_md_device", "cpu")),

                                    # ####################### linregstable satrmd+msp by prr
                                    # # LinRegTokenMahalanobisDistance_Hybrid("decoder", parameters_path=None, 
                                    # #                                         metric=metric, metric_name=metric_name, metric_md=metric, metric_md_name=metric_name,
                                    # #                                         aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=thr,
                                    # #                                         ue="TokenMahalanobis", positive=False, meta_model="StableLinReg", norm="orig", remove_corr=True, remove_alg=3, storage_device=getattr(args, "clean_md_device", "cpu")),
                                    
                                    # # LinRegTokenMahalanobisDistance_Hybrid("decoder", parameters_path=None, 
                                    # #                                         metric=metric, metric_name=metric_name, metric_md=metric, metric_md_name=metric_name,
                                    # #                                         aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=thr,
                                    # #                                         ue="RelativeTokenMahalanobis", positive=False, meta_model="StableLinReg", norm="orig", remove_corr=True, remove_alg=3, storage_device=getattr(args, "clean_md_device", "cpu")),


                                    # ####################### weighted linregstable satrmd+msp by prr
                                    # LinRegTokenMahalanobisDistance_Hybrid("decoder", parameters_path=None, 
                                    #                                         metric=metric, metric_name=metric_name, metric_md=metric, metric_md_name=metric_name,
                                    #                                         aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=thr,
                                    #                                         ue="TokenMahalanobis", positive=False, meta_model="WeightedStableLinReg", norm="orig", remove_corr=True, remove_alg=3, storage_device=getattr(args, "clean_md_device", "cpu")),
                                    
                                    # LinRegTokenMahalanobisDistance_Hybrid("decoder", parameters_path=None, 
                                    #                                         metric=metric, metric_name=metric_name, metric_md=metric, metric_md_name=metric_name,
                                    #                                         aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=thr,
                                    #                                         ue="RelativeTokenMahalanobis", positive=False, meta_model="WeightedStableLinReg", norm="orig", remove_corr=True, remove_alg=3, storage_device=getattr(args, "clean_md_device", "cpu")),
                                    
                                    
                                    # HUQ_LRTMD("decoder", parameters_path=None, 
                                    #             metric=metric, metric_name=metric_name, metric_md=metric, metric_md_name=metric_name, 
                                    #             aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=thr,
                                    #             ue="TokenMahalanobis", positive=False, meta_model="LinReg", norm="orig", remove_corr=True, remove_alg=3, storage_device=getattr(args, "clean_md_device", "cpu")),

                                    # HUQ_LRTMD("decoder", parameters_path=None, 
                                    #             metric=metric, metric_name=metric_name, metric_md=metric, metric_md_name=metric_name, 
                                    #             aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=thr,
                                    #             ue="RelativeTokenMahalanobis", positive=False, meta_model="LinReg", norm="orig", remove_corr=True, remove_alg=3, storage_device=getattr(args, "clean_md_device", "cpu")),
                                    
                        ]
        
    return estimators


def get_ue_methods(args, model):
    estimators = []
    if args.task == "nmt":
        comet = Comet(source_ignore_regex = getattr(args, "source_ignore_regex", None))
        use_comet = True
        acc = None
        use_accuracy = False
    elif args.task == "qa":
        comet = None
        use_comet = False
        acc = AccuracyMetric(
                target_ignore_regex = getattr(args, "target_ignore_regex", None),
                output_ignore_regex = getattr(args, "output_ignore_regex", None),
                normalize = getattr(args, "normalize", False),
            )
        use_accuracy = True

    dataset_name = args.dataset if isinstance(args.dataset, str) else '_'.join(args.dataset)
    dataset_name = dataset_name.split("/")[-1].split(".")[0]
    model_name = args.model.path.split("/")[-1]
    parameters_path = f"{args.cache_path}/tad_stats/{dataset_name}/{model_name}"
                      
    if args.use_seq_ue:
        estimators += [
            MaximumSequenceProbability(),
            Perplexity(),
            MeanTokenEntropy(),
        ]
        if getattr(args, "run_baselines", False):
            estimators += [
                # MeanPointwiseMutualInformation(),
                # MeanConditionalPointwiseMutualInformation(),
                # ClaimConditionedProbability(),
                # PTrue(),
                # PTrueSampling(),
                MonteCarloSequenceEntropy(),
                MonteCarloNormalizedSequenceEntropy(),
                # LexicalSimilarity(metric="rouge1"),
                # LexicalSimilarity(metric="rouge2"),
                LexicalSimilarity(metric="rougeL"),
                # LexicalSimilarity(metric="BLEU"),
                NumSemSets(),
                EigValLaplacian(similarity_score="NLI_score", affinity="entail"),
                # EigValLaplacian(similarity_score="NLI_score", affinity="contra"),
                # EigValLaplacian(similarity_score="Jaccard_score"),
                DegMat(similarity_score="NLI_score", affinity="entail"),
                # DegMat(similarity_score="NLI_score", affinity="contra"),
                # DegMat(similarity_score="Jaccard_score"),
                Eccentricity(similarity_score="NLI_score", affinity="entail"),
                # Eccentricity(similarity_score="NLI_score", affinity="contra"),
                # Eccentricity(similarity_score="Jaccard_score"),
                SemanticEntropy(),
                SAR(),
                # TokenSAR(),
                SentenceSAR(),
                # RenyiNeg(),
                # FisherRao(),
            ]

    if args.use_ens_ue:
        if not (model.model_type == "Seq2SeqLM"):
            raise NotImplementedError('Only Encoder-Decoder models can be ensembled at this time')

        token_measures = all_token_estimators()
        if args.model.ensembling_mode == 'pe':
            sequence_measures = all_pe_estimators()
        elif args.model.ensembling_mode == 'ep':
            sequence_measures = all_ep_estimators()
        else:
            raise ValueError(f'Ensemble type should be one of: "pe", "ep", but is {args.ens_type} instead')
        estimators += (token_measures + sequence_measures)

    if args.use_tok_ue:
        estimators += [
            MaximumTokenProbability(),
            TokenEntropy(),
            PointwiseMutualInformation(),
            ConditionalPointwiseMutualInformation(),
            SemanticEntropyToken(model.model_path, args.cache_path),
        ]

    if getattr(args, "use_claim_ue", False):
        estimators += [
            MaximumClaimProbability(),
            # PerplexityClaim(),
            # MaxTokenEntropyClaim(),
            # PointwiseMutualInformationClaim(),
            PTrueClaim(),
            ClaimConditionedProbabilityClaim(nli_context="no_context"),
            ClaimConditionedProbabilityClaim(nli_context="fact_pref"),
        ]
        layers = getattr(args, "layers", [0, -1])
        metric_thrs = getattr(args, "metric_thrs", [0.0, 0.5])
        for layer in layers:
            estimators += [SAPLMAClaim("decoder", hidden_layer=layer)]
            for thr in metric_thrs:
                estimators += [TokenMahalanobisDistanceClaim("decoder", hidden_layer=layer, metric_thr=thr),
                               RelativeTokenMahalanobisDistanceClaim("decoder", hidden_layer=layer, metric_thr=thr)]  

        #########################################################################
        # LinReg
        estimators += [LinRegTokenMahalanobisDistance_Claim("decoder", parameters_path=parameters_path, 
                                                      aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=metric_thrs[-1], aggregation="mean",
                                                      ue="TokenMahalanobis", positive=False, meta_model="LinReg", norm="orig", remove_corr=True, remove_alg=3)]
       
        estimators += [LinRegTokenMahalanobisDistance_Claim("decoder", parameters_path=parameters_path, 
                                                      aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=metric_thrs[-1], aggregation="mean",
                                                      ue="RelativeTokenMahalanobis", positive=False, meta_model="LinReg", norm="orig", remove_corr=True, remove_alg=3)]

        estimators += [LinRegTokenMahalanobisDistance_Claim("decoder", parameters_path=parameters_path, 
                                                      aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=metric_thrs[-1], aggregation="mean",
                                                      ue="TokenMahalanobis", positive=False, meta_model="LinReg", norm="orig", remove_corr=False, remove_alg=3)]
        estimators += [LinRegTokenMahalanobisDistance_Claim("decoder", parameters_path=parameters_path, 
                                                      aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=metric_thrs[-1], aggregation="mean",
                                                      ue="RelativeTokenMahalanobis", positive=False, meta_model="LinReg", norm="orig", remove_corr=False, remove_alg=3)]

        #########################################################################
        # LinReg + MSP
        estimators += [LinRegTokenMahalanobisDistance_Hybrid_Claim("decoder", parameters_path=parameters_path, 
                                                      aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=metric_thrs[-1], aggregation="mean",
                                                      ue="TokenMahalanobis", positive=False, meta_model="LinReg", norm="orig", remove_corr=True, remove_alg=3)]
        estimators += [LinRegTokenMahalanobisDistance_Hybrid_Claim("decoder", parameters_path=parameters_path, 
                                                      aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=metric_thrs[-1], aggregation="mean",
                                                      ue="RelativeTokenMahalanobis", positive=False, meta_model="LinReg", norm="orig", remove_corr=True, remove_alg=3)]

        estimators += [LinRegTokenMahalanobisDistance_Hybrid_Claim("decoder", parameters_path=parameters_path, 
                                                      aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=metric_thrs[-1], aggregation="mean",
                                                      ue="TokenMahalanobis", positive=False, meta_model="LinReg", norm="orig", remove_corr=False, remove_alg=3)]
        estimators += [LinRegTokenMahalanobisDistance_Hybrid_Claim("decoder", parameters_path=parameters_path, 
                                                      aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=metric_thrs[-1], aggregation="mean",
                                                      ue="RelativeTokenMahalanobis", positive=False, meta_model="LinReg", norm="orig", remove_corr=False, remove_alg=3)]

        
        #########################################################################
        # HUQ
        estimators += [HUQ_LRTMD_Claim("decoder", parameters_path=parameters_path, 
                                                      aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=metric_thrs[-1], aggregation="mean",
                                                      ue="TokenMahalanobis", positive=False, meta_model="LinReg", norm="orig", remove_corr=True, remove_alg=3)]
        estimators += [HUQ_LRTMD_Claim("decoder", parameters_path=parameters_path, 
                                                      aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=metric_thrs[-1], aggregation="mean",
                                                      ue="RelativeTokenMahalanobis", positive=False, meta_model="LinReg", norm="orig", remove_corr=True, remove_alg=3)]

        estimators += [HUQ_LRTMD_Claim("decoder", parameters_path=parameters_path, 
                                                      aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=metric_thrs[-1], aggregation="mean",
                                                      ue="TokenMahalanobis", positive=False, meta_model="LinReg", norm="orig", remove_corr=False, remove_alg=3)]
        estimators += [HUQ_LRTMD_Claim("decoder", parameters_path=parameters_path, 
                                                      aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=metric_thrs[-1], aggregation="mean",
                                                      ue="RelativeTokenMahalanobis", positive=False, meta_model="LinReg", norm="orig", remove_corr=False, remove_alg=3)]

        #########################################################################
        # LinReg + MSP + CCP
        estimators += [LinRegTokenMahalanobisDistance_Hybrid_Claim("decoder", parameters_path=parameters_path, 
                                                      aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=metric_thrs[-1], aggregation="mean",
                                                      ue="TokenMahalanobis", positive=False, meta_model="LinReg", norm="orig", remove_corr=True, remove_alg=3, use_ccp=True)]
        estimators += [LinRegTokenMahalanobisDistance_Hybrid_Claim("decoder", parameters_path=parameters_path, 
                                                      aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=metric_thrs[-1], aggregation="mean",
                                                      ue="RelativeTokenMahalanobis", positive=False, meta_model="LinReg", norm="orig", remove_corr=True, remove_alg=3, use_ccp=True)]

        estimators += [LinRegTokenMahalanobisDistance_Hybrid_Claim("decoder", parameters_path=parameters_path, 
                                                      aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=metric_thrs[-1], aggregation="mean",
                                                      ue="TokenMahalanobis", positive=False, meta_model="LinReg", norm="orig", remove_corr=False, remove_alg=3, use_ccp=True)]
        estimators += [LinRegTokenMahalanobisDistance_Hybrid_Claim("decoder", parameters_path=parameters_path, 
                                                      aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=metric_thrs[-1], aggregation="mean",
                                                      ue="RelativeTokenMahalanobis", positive=False, meta_model="LinReg", norm="orig", remove_corr=False, remove_alg=3, use_ccp=True)]

        
        #########################################################################
        # HUQ + CCP
        estimators += [HUQ_LRTMD_Claim("decoder", parameters_path=parameters_path, 
                                                      aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=metric_thrs[-1], aggregation="mean",
                                                      ue="TokenMahalanobis", positive=False, meta_model="LinReg", norm="orig", remove_corr=True, remove_alg=3, use_ccp=True)]
        estimators += [HUQ_LRTMD_Claim("decoder", parameters_path=parameters_path, 
                                                      aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=metric_thrs[-1], aggregation="mean",
                                                      ue="RelativeTokenMahalanobis", positive=False, meta_model="LinReg", norm="orig", remove_corr=True, remove_alg=3, use_ccp=True)]

        estimators += [HUQ_LRTMD_Claim("decoder", parameters_path=parameters_path, 
                                                      aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=metric_thrs[-1], aggregation="mean",
                                                      ue="TokenMahalanobis", positive=False, meta_model="LinReg", norm="orig", remove_corr=False, remove_alg=3, use_ccp=True)]
        estimators += [HUQ_LRTMD_Claim("decoder", parameters_path=parameters_path, 
                                                      aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=metric_thrs[-1], aggregation="mean",
                                                      ue="RelativeTokenMahalanobis", positive=False, meta_model="LinReg", norm="orig", remove_corr=False, remove_alg=3, use_ccp=True)]

        #########################################################################
        # LinReg + MSP + CCP_FP
        estimators += [LinRegTokenMahalanobisDistance_Hybrid_Claim("decoder", parameters_path=parameters_path, 
                                                      aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=metric_thrs[-1], aggregation="mean",
                                                      ue="TokenMahalanobis", positive=False, meta_model="LinReg", norm="orig", remove_corr=True, remove_alg=3, use_ccp=True, ccp_context="fact_pref")]
        estimators += [LinRegTokenMahalanobisDistance_Hybrid_Claim("decoder", parameters_path=parameters_path, 
                                                      aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=metric_thrs[-1], aggregation="mean",
                                                      ue="RelativeTokenMahalanobis", positive=False, meta_model="LinReg", norm="orig", remove_corr=True, remove_alg=3, use_ccp=True, ccp_context="fact_pref")]

        estimators += [LinRegTokenMahalanobisDistance_Hybrid_Claim("decoder", parameters_path=parameters_path, 
                                                      aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=metric_thrs[-1], aggregation="mean",
                                                      ue="TokenMahalanobis", positive=False, meta_model="LinReg", norm="orig", remove_corr=False, remove_alg=3, use_ccp=True, ccp_context="fact_pref")]
        estimators += [LinRegTokenMahalanobisDistance_Hybrid_Claim("decoder", parameters_path=parameters_path, 
                                                      aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=metric_thrs[-1], aggregation="mean",
                                                      ue="RelativeTokenMahalanobis", positive=False, meta_model="LinReg", norm="orig", remove_corr=False, remove_alg=3, use_ccp=True, ccp_context="fact_pref")]

        
        #########################################################################
        # HUQ + CCP_FP
        estimators += [HUQ_LRTMD_Claim("decoder", parameters_path=parameters_path, 
                                                      aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=metric_thrs[-1], aggregation="mean",
                                                      ue="TokenMahalanobis", positive=False, meta_model="LinReg", norm="orig", remove_corr=True, remove_alg=3, use_ccp=True, ccp_context="fact_pref")]
        estimators += [HUQ_LRTMD_Claim("decoder", parameters_path=parameters_path, 
                                                      aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=metric_thrs[-1], aggregation="mean",
                                                      ue="RelativeTokenMahalanobis", positive=False, meta_model="LinReg", norm="orig", remove_corr=True, remove_alg=3, use_ccp=True, ccp_context="fact_pref")]

        estimators += [HUQ_LRTMD_Claim("decoder", parameters_path=parameters_path, 
                                                      aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=metric_thrs[-1], aggregation="mean",
                                                      ue="TokenMahalanobis", positive=False, meta_model="LinReg", norm="orig", remove_corr=False, remove_alg=3, use_ccp=True, ccp_context="fact_pref")]
        estimators += [HUQ_LRTMD_Claim("decoder", parameters_path=parameters_path, 
                                                      aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=metric_thrs[-1], aggregation="mean",
                                                      ue="RelativeTokenMahalanobis", positive=False, meta_model="LinReg", norm="orig", remove_corr=False, remove_alg=3, use_ccp=True, ccp_context="fact_pref")]

            

    additional_estimators = getattr(args, "additional_estimators", {})
    additional_estimators_kwargs = getattr(args, "additional_estimators_kwargs", {})

    for i, (module_name, estimator_classes) in enumerate(additional_estimators.items()):
        module = importlib.import_module(module_name)
        for j, estimator_class in enumerate(estimator_classes):
            try:
                estimator_kwargs = additional_estimators_kwargs[estimator_class]
            except KeyError:
                raise TypeError(f'Arguments for {estimator} were not passed')

            estimators.append(getattr(module, estimator_class)(**estimator_kwargs))

    return estimators


def get_generation_metrics(args):
    generation_metrics = getattr(args, "generation_metrics", None)
    if not generation_metrics:
        
        if (args.task == "qa") and  (args.dataset not in ["keivalya/MedQuad-MedicalQnADataset", "bigbio/pubmed_qa", ['truthful_qa', 'generation']]):
            alignscorer = AlignScore(batch_size=1)
        else:
            alignscorer = AlignScoreNew(return_mean=True, batch_size=1)
            
        result = [
            #RougeMetric("rouge1"),
            #RougeMetric("rouge2"),
            RougeMetric("rougeL"),
            #BertScoreMetric('rh'),
            #SbertMetric(),
            AccuracyMetric(
                target_ignore_regex = getattr(args, "target_ignore_regex", None),
                output_ignore_regex = getattr(args, "output_ignore_regex", None),
                normalize = getattr(args, "normalize", False),
            ),
            alignscorer
        ]
        if args.task == "nmt":
            ignore_regex = getattr(args, "source_ignore_regex", None)
            result += [Comet(source_ignore_regex = ignore_regex)]
        if not getattr(args, "multiref", False):
            pass
            # Currently, BartScoreSeqMetric does not support multiref
            # result.append(BartScoreSeqMetric('rh'))
        else:
            # Wrap each metric in AggregatedMetric
            result = [AggregatedMetric(base_metric=metric) for metric in result]
    else:
        result = []
        for metric in generation_metrics:
            metric_name = metric["name"]
            if getattr(args, "multiref", False) and metric_name == "BartScoreSeqMetric":
                raise ValueError("BartScoreSeqMetric does not support multiref")
            metric_class = globals()[metric_name]
            result.append(metric_class(*metric.get("args", [])))
    return result


def get_model_kwargs(args):
    model_kwargs = {}
    # if getattr(args.model, 'device_map', None):
    #     model_kwargs['device_map'] = args.model.device_map
    if getattr(args.model, 'attn_implementation', None):
        model_kwargs['attn_implementation'] = args.model.attn_implementation

    return model_kwargs


if __name__ == "__main__":
    main()
