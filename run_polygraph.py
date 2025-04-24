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

import logging

log = logging.getLogger()

from lm_polygraph.utils.manager import UEManager
from utils.dataset import Dataset
from utils.alignscore import AlignScore as AlignScoreNew
from lm_polygraph.utils.model import WhiteboxModel, BlackboxModel, create_ensemble
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
from lm_polygraph.generation_metrics import *
from lm_polygraph.ue_metrics import *
from transformers import AutoConfig
from lm_polygraph.utils.common import load_external_module

from token_mahalanobis_distance import TokenMahalanobisDistance, TokenMahalanobisDistanceClaim
from average_token_mahalanobis_distance import LinRegTokenMahalanobisDistance, LinRegTokenMahalanobisDistance_Claim
from average_token_mahalanobis_distance_hybrid import LinRegTokenMahalanobisDistance_Hybrid, LinRegTokenMahalanobisDistance_Hybrid_Claim
from relative_token_mahalanobis_distance import RelativeTokenMahalanobisDistance, RelativeTokenMahalanobisDistanceClaim
from supervised_baselines.saplma import SAPLMA, SAPLMAClaim, SAPLMA_truefalse
from supervised_baselines.factoscope import LLMFactoscope, LLMFactoscopeAll
from unsupervised_baselines.eigenscore import EigenScore
from huq_msp_lrtmd import HUQ_LRTMD, HUQ_LRTMD_Claim

import nltk
nltk.download('punkt_tab')

hydra_config = Path(os.environ["HYDRA_CONFIG"])

def get_model(args, cache_kwargs={}):
    if getattr(args.model, "type", "Whitebox") == "Blackbox":
        return get_blackbox_model(args)
    else:
        return get_whitebox_model(args, cache_kwargs)


def get_blackbox_model(args):
    provider = getattr(args.model, "provider", "") 
    if provider is None or provider.strip() == "":
        raise ValueError("Blackbox model provider cannot be empty or None. Please specify a valid provider.")

    if provider == "openai":
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if openai_api_key is None:
            raise ValueError("OpenAI API key is not set in the environment variables.")
        return BlackboxModel.from_openai(
            openai_api_key=openai_api_key,
            model_path=args.model.path
        )
    elif provider == "huggingface":
        hf_api_key = os.environ.get("HUGGINGFACE_API_KEY")
        if hf_api_key is None:
            raise ValueError("HuggingFace API key is not set in the environment variables.")
        return BlackboxModel.from_huggingface(
            hf_api_token=hf_api_key,
            hf_model_id=args.model.path
        )
    else:
        raise ValueError(f"Unsupported black-box model provider: {provider}")


def get_whitebox_model(args, cache_kwargs={}):
    if not "path_to_load_script" in args.model:
        log.warning(
            "Loading model by directly passing the path to the model is deprecated and will be removed in the next release. Please use loading script instead."
        )
        model_kwargs = get_model_kwargs(args)
        
        return WhiteboxModel.from_pretrained(
            args.model.path,
            getattr(args, "generation_params", {}),
            device_map=args.model.device_map,
            add_bos_token=getattr(args.model, "add_bos_token", True),
            **cache_kwargs,
            **model_kwargs,
        )

    path_to_load_script = get_abs_path_from_hydra_config(
            args.model.path_to_load_script
        )
    load_module = load_external_module(path_to_load_script)

    load_model_args = {'model_path': args.model.path}
    load_model_args.update(args.model.load_model_args)
    base_model = load_module.load_model(**load_model_args)

    load_tok_args = {'model_path': args.model.path}
    load_tok_args.update(args.model.load_tokenizer_args)
    tokenizer = load_module.load_tokenizer(**load_tok_args)

    generation_params = GenerationParameters(**getattr(args, "generation_params", {}))

    model = WhiteboxModel(base_model,
                          tokenizer,
                          args.model.path,
                          args.model.type,
                          generation_params)

    return model


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

    # model_kwargs = get_model_kwargs(args)

    cache_kwargs = {}
    if os.environ.get('HF_DATASETS_OFFLINE', '').strip() == '1':
        cache_kwargs = {'cache_dir': args.cache_path}

    for seed in args.seed:
        log.info("=" * 100)
        log.info(f"SEED: {seed}")

        log.info(f"Loading model {args.model.path}...")
        transformers.set_seed(seed)

        model = get_model(args, cache_kwargs)
        
        # model = WhiteboxModel.from_pretrained(
        #     args.model.path,
        #     getattr(args, "generation_params", {}),
        #     device_map=args.model.device_map,
        #     add_bos_token=getattr(args.model, "add_bos_token", True),
        #     **cache_kwargs,
        #     **model_kwargs,
        # )
        
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
            instruct=getattr(args, "instruct", False),
            few_shot_prompt=getattr(args, "few_shot_prompt", None),
            load_from_disk=args.load_from_disk,
            max_new_tokens=getattr(args, f"max_new_tokens", 100),
            **cache_kwargs
        )
        
        if args.subsample_eval_dataset != -1:
            dataset.subsample(args.subsample_eval_dataset, seed=seed)

        if getattr(args, "eval_dataset_1", False):
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
                    instruct=getattr(args, "instruct", False),
                    few_shot_prompt=getattr(args, "few_shot_prompt", None),
                    load_from_disk=args.load_from_disk,
                    **cache_kwargs
                )
                if args.subsample_eval_dataset != -1:
                    eval_dataset_k.subsample(args.subsample_eval_dataset, seed=seed)

                if getattr(args, "multiref", False):
                    if isinstance(eval_dataset_k.y[0], list):
                        dataset.concat(eval_dataset_k.x, eval_dataset_k.y, eval_dataset_k.max_new_tokens)
                    else:
                        dataset.concat(eval_dataset_k.x, [[y] for y in eval_dataset_k.y], eval_dataset_k.max_new_tokens)
                else:
                    if isinstance(eval_dataset_k.y[0], list):
                        dataset.concat(eval_dataset_k.x, eval_dataset_k.y[0], eval_dataset_k.max_new_tokens)
                    else:
                        dataset.concat(eval_dataset_k.x, eval_dataset_k.y, eval_dataset_k.max_new_tokens)
                k_ds += 1
                

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
                    instruct=getattr(args, "instruct", False),
                    few_shot_prompt=getattr(args, "few_shot_prompt", None),
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
                    instruct=getattr(args, "instruct", False),
                    few_shot_prompt=getattr(args, "few_shot_prompt", None),
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
                        instruct=getattr(args, "instruct", False),
                        few_shot_prompt=getattr(args, "few_shot_prompt", None),
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
                instruct=getattr(args, "instruct", False),
                few_shot_prompt=getattr(args, "few_shot_prompt", None),
                load_from_disk=args.background_load_from_disk,
                **cache_kwargs
            )
            if args.subsample_background_train_dataset != -1:
                background_train_dataset.subsample(
                    args.subsample_background_train_dataset, seed=seed
                )            

        log.info("Done with loading data.")
        generation_metrics = get_generation_metrics(args)
        ue_metrics = get_ue_metrics(args)

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
        if getattr(args, "run_proposed_methods", False) or getattr(args, "run_ablation", False):
            if (args.task == "qa") and (args.dataset not in ["keivalya/MedQuad-MedicalQnADataset", "bigbio/pubmed_qa", ['truthful_qa', 'generation']]):
                if getattr(args, "is_ood", False):
                    if getattr(args, "mean_als", True):
                        alignscorer = AlignScoreNew(return_mean=True, batch_size=1)
                    else:
                        alignscorer = AlignScoreNew(return_mean=False, batch_size=1)
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
        n_components_all = getattr(args, "n_components", [10])    

        process_output_fn = getattr(args, "process_output_fn", None)
        process_target_fn = getattr(args, "process_target_fn", None)
        if process_target_fn or process_output_fn:
            if (getattr(args, "target_ignore_regex", None) or 
                getattr(args, "output_ignore_regex", None) or
                getattr(args, "normalize", False)):
                raise ValueError("Specifying ignore_regex or normalize simultaneously with process scripts is not allowed.")
    
            def load_process_fn(fn_config):
                if not fn_config:
                    return None
                path = get_abs_path_from_hydra_config(fn_config.path)
                module = load_external_module(path)
                return getattr(module, fn_config.fn_name)
    
            process_output_fn = load_process_fn(process_output_fn)
            process_target_fn = load_process_fn(process_target_fn)
    
            metrics = [PreprocessOutputTarget(metric, process_output_fn, process_target_fn) for metric in metrics]
    
        if getattr(args, 'parameters_path', False):
            parameters_path = args.parameters_path
        else:
            dataset_name = args.dataset if isinstance(args.dataset, str) else '_'.join(args.dataset)
            dataset_name = dataset_name.split("/")[-1].split(".")[0]
            model_name = args.model.path.split("/")[-1]
            parameters_path = f"{args.cache_path}/density_stats/{dataset_name}/{model_name}"
        
        if (getattr(args.model, "type", "Whitebox") != "Blackbox") and (model_type == "Seq2SeqLM"):
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
        elif getattr(args.model, "type", "Whitebox") != "Blackbox":
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
                #layer-wise methods
                for layer in layers:
                    for metric, metric_name in zip(metrics, metrics_names):
                        estimators += [SAPLMA("decoder", parameters_path=None, metric=metric, metric_name=metric_name, aggregated=getattr(args, "multiref", False), hidden_layer=layer, cv_hp=True)]
                        for thr in metric_thrs:
                            estimators += [ 
                                TokenMahalanobisDistance("decoder", parameters_path=None, metric=metric, metric_name=metric_name, aggregated=getattr(args, "multiref", False), hidden_layer=layer, metric_thr=thr, storage_device=getattr(args, "clean_md_device", "cpu")),
                                RelativeTokenMahalanobisDistance("decoder", parameters_path=None, metric=metric, metric_name=metric_name, aggregated=getattr(args, "multiref", False), hidden_layer=layer, metric_thr=thr, storage_device=getattr(args, "clean_md_device", "cpu"))
                            ]
                            
                # meta methods
                for metric, metric_name in zip(metrics, metrics_names):
                    for thr in metric_thrs:
                        estimators += [
                                    LLMFactoscopeAll(metric=metric, metric_name=metric_name, metric_thr=thr, hidden_layers=layers, return_dist=True, return_new_dist=True),
                                    
                                    ####################### SATRMD
                                    LinRegTokenMahalanobisDistance("decoder", parameters_path=None, 
                                                                    metric=metric, metric_name=metric_name, metric_md=metric, metric_md_name=metric_name,
                                                                    aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=thr,
                                                                    ue="TokenMahalanobis", positive=False, meta_model="LinReg", norm="orig", remove_corr=True, remove_alg=3, storage_device=getattr(args, "clean_md_device", "cpu")),
                                    
                                    LinRegTokenMahalanobisDistance("decoder", parameters_path=None, 
                                                                    metric=metric, metric_name=metric_name, metric_md=metric, metric_md_name=metric_name,
                                                                    aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=thr,
                                                                    ue="RelativeTokenMahalanobis", positive=False, meta_model="LinReg", norm="orig", remove_corr=True, remove_alg=3, storage_device=getattr(args, "clean_md_device", "cpu")),

                                    ####################### SATRMD+MSP
                                    LinRegTokenMahalanobisDistance_Hybrid("decoder", parameters_path=None, 
                                                                            metric=metric, metric_name=metric_name, metric_md=metric, metric_md_name=metric_name,
                                                                            aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=thr,
                                                                            ue="TokenMahalanobis", positive=False, meta_model="LinReg", norm="orig", remove_corr=True, remove_alg=3, storage_device=getattr(args, "clean_md_device", "cpu"), sim_pca=False),
                                    
                                    LinRegTokenMahalanobisDistance_Hybrid("decoder", parameters_path=None, 
                                                                            metric=metric, metric_name=metric_name, metric_md=metric, metric_md_name=metric_name,
                                                                            aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=thr,
                                                                            ue="RelativeTokenMahalanobis", positive=False, meta_model="LinReg", norm="orig", remove_corr=True, remove_alg=3, storage_device=getattr(args, "clean_md_device", "cpu"), sim_pca=False),
                                    ####################### HUQ-SATRMD
                                    HUQ_LRTMD("decoder", parameters_path=None, 
                                                metric=metric, metric_name=metric_name, metric_md=metric, metric_md_name=metric_name, 
                                                aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=thr,
                                                ue="TokenMahalanobis", positive=False, meta_model="LinReg", norm="orig", remove_corr=True, remove_alg=3, storage_device=getattr(args, "clean_md_device", "cpu")),

                                    HUQ_LRTMD("decoder", parameters_path=None, 
                                                metric=metric, metric_name=metric_name, metric_md=metric, metric_md_name=metric_name, 
                                                aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=thr,
                                                ue="RelativeTokenMahalanobis", positive=False, meta_model="LinReg", norm="orig", remove_corr=True, remove_alg=3, storage_device=getattr(args, "clean_md_device", "cpu")),
                        ]
                        
            if getattr(args, "run_ablation", False):
                # meta methods
                for metric, metric_name in zip(metrics, metrics_names):
                    for n_components in n_components_all:
                        for thr in metric_thrs:
                            estimators += [
                                        
                                        LinRegTokenMahalanobisDistance_Hybrid("decoder", parameters_path=None, 
                                                                                metric=metric, metric_name=metric_name, metric_md=metric, metric_md_name=metric_name,
                                                                                aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=thr, n_components=n_components,
                                                                                ue="RelativeTokenMahalanobis", positive=False, meta_model="LinReg", norm="orig", remove_corr=True, remove_alg=3, storage_device=getattr(args, "clean_md_device", "cpu"), sim_pca=False),
                                        HUQ_LRTMD("decoder", parameters_path=None, 
                                                    metric=metric, metric_name=metric_name, metric_md=metric, metric_md_name=metric_name, 
                                                    aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=thr, n_components=n_components,
                                                    ue="RelativeTokenMahalanobis", positive=False, meta_model="LinReg", norm="orig", remove_corr=True, remove_alg=3, storage_device=getattr(args, "clean_md_device", "cpu")),                                    
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

    if getattr(args.model, "type", "Whitebox") == "Blackbox":
        if getattr(args, "use_seq_ue", False):
            estimators += [
                LexicalSimilarity(metric="rouge1"),
                LexicalSimilarity(metric="rouge2"),
                LexicalSimilarity(metric="rougeL"),
                LexicalSimilarity(metric="BLEU"),
                NumSemSets(),
                EigValLaplacian(similarity_score="NLI_score", affinity="entail"),
                EigValLaplacian(similarity_score="NLI_score", affinity="contra"),
                EigValLaplacian(similarity_score="Jaccard_score"),
                DegMat(similarity_score="NLI_score", affinity="entail"),
                DegMat(similarity_score="NLI_score", affinity="contra"),
                DegMat(similarity_score="Jaccard_score"),
                Eccentricity(similarity_score="NLI_score", affinity="entail"),
                Eccentricity(similarity_score="NLI_score", affinity="contra"),
                Eccentricity(similarity_score="Jaccard_score"),
            ]
            
    elif args.use_seq_ue:
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
        
        #########################################################################
        # LinReg + MSP + CCP_FP
        estimators += [LinRegTokenMahalanobisDistance_Hybrid_Claim("decoder", parameters_path=parameters_path, 
                                                      aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=metric_thrs[-1], aggregation="mean",
                                                      ue="TokenMahalanobis", positive=False, meta_model="LinReg", norm="orig", remove_corr=True, remove_alg=3, use_ccp=True, ccp_context="fact_pref")]
        estimators += [LinRegTokenMahalanobisDistance_Hybrid_Claim("decoder", parameters_path=parameters_path, 
                                                      aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=metric_thrs[-1], aggregation="mean",
                                                      ue="RelativeTokenMahalanobis", positive=False, meta_model="LinReg", norm="orig", remove_corr=True, remove_alg=3, use_ccp=True, ccp_context="fact_pref")]

        
        #########################################################################
        # HUQ + CCP_FP
        estimators += [HUQ_LRTMD_Claim("decoder", parameters_path=parameters_path, 
                                                      aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=metric_thrs[-1], aggregation="mean",
                                                      ue="TokenMahalanobis", positive=False, meta_model="LinReg", norm="orig", remove_corr=True, remove_alg=3, use_ccp=True, ccp_context="fact_pref")]
        estimators += [HUQ_LRTMD_Claim("decoder", parameters_path=parameters_path, 
                                                      aggregated=getattr(args, "multiref", False), hidden_layers=layers, metric_thr=metric_thrs[-1], aggregation="mean",
                                                      ue="RelativeTokenMahalanobis", positive=False, meta_model="LinReg", norm="orig", remove_corr=True, remove_alg=3, use_ccp=True, ccp_context="fact_pref")]
            

    additional_estimators = getattr(args, "additional_estimators", {})

    for estimator_args in additional_estimators:
        module = importlib.import_module(estimator_args.module)
        estimator_class = getattr(module, estimator_args.class_name)
        estimators.append(estimator_class(**estimator_args.kwargs))

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

        process_output_fn = getattr(args, "process_output_fn", None)
        process_target_fn = getattr(args, "process_target_fn", None)
        if process_target_fn or process_output_fn:
            if (getattr(args, "target_ignore_regex", None) or 
                getattr(args, "output_ignore_regex", None) or
                getattr(args, "normalize", False)):
                raise ValueError("Specifying ignore_regex or normalize simultaneously with process scripts is not allowed.")
    
            def load_process_fn(fn_config):
                if not fn_config:
                    return None
                path = get_abs_path_from_hydra_config(fn_config.path)
                module = load_external_module(path)
                return getattr(module, fn_config.fn_name)
    
            process_output_fn = load_process_fn(process_output_fn)
            process_target_fn = load_process_fn(process_target_fn)
    
            result = [PreprocessOutputTarget(metric, process_output_fn, process_target_fn) for metric in result]
        
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

def get_abs_path_from_hydra_config(path: str) -> Path:
    path = Path(path)
    if not os.path.isabs(path):
        path = hydra_config.parent / path

    return path


if __name__ == "__main__":
    main()
