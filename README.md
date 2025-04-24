# Token-Level Density-Based Uncertainty Quantification Methods for Eliciting Truthfulness of Large Language Models

This repository contains the implementation of token-level density-based methods for Uncertainty Quantification (UQ) for selective text generation tasks using LLMs.

Namely, the repository contains code for reproducing experiments from the paper ["Token-Level Density-Based Uncertainty Quantification Methods for Eliciting Truthfulness of Large Language Models"](https://arxiv.org/pdf/2502.14427) at the NAACL-2025 conference.

## What the paper is about?

Uncertainty quantification (UQ) is a prominent approach for eliciting truthful answers from large language models (LLMs). To date, information-based and consistency-based UQ have been the dominant UQ methods for text generation via LLMs. Density-based methods, despite being very effective for UQ in text classification with encoder-based models, have not been very successful with generative LLMs. In this work, we adapt Mahalanobis Distance (MD) - a well-established UQ technique in classification tasks - for text generation and introduce a new supervised UQ method. Our method extracts token embeddings from multiple layers of LLMs, computes MD scores for each token, and uses linear regression trained on these features to provide robust uncertainty scores. Through extensive experiments on eleven datasets, we demonstrate that our approach substantially improves over existing UQ methods, providing accurate and computationally efficient uncertainty scores for both sequence-level selective generation and claim-level fact-checking tasks. Our method also exhibits strong generalization to out-of-domain data, making it suitable for a wide range of LLM-based applications.

## Reproducing Experiments

### Installation

```shell
conda create -n token_md python=3.11
source activate token_md
git clone https://github.com/ArtemVazh/token_mahalanobis_distance.git
cd token_mahalanobis_distance
pip install -r requirements.txt 
```

### Run Experiment

```
HYDRA_CONFIG=./configs/polygraph_eval_sciq.yaml python run_polygraph.py \
ignore_exceptions=False use_density_based_ue=True batch_size=1 \
subsample_train_dataset=5000 subsample_background_train_dataset=2000 subsample_eval_dataset=2000 \
model.path=meta-llama/Meta-Llama-3.1-8B +model.attn_implementation=eager \
cache_path=./workdir/output_layers_internal_final \
+metric_thrs="[0.3]" +layers="[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,-1]" \
+generation_params.samples_n=5 +run_baselines=True +md_device=cuda +run_proposed_methods=True +clean_md_device=cuda
```


### Reproduce Paper Experiments

To reproduce the experiments from the paper, run the following commands.

1. For the SciQ, CoQA, TriviaQA, MMLU, TruthfulQA, and SamSum datasets:
```shell
cd scripts
bash run_tmd_exps_final_1.sh
```

2. For the GSM8k, MedQUAD, and XSum datasets:
```shell
cd scripts
bash run_tmd_exps_final_2.sh
```

3. For the CNN and PubMedQA datasets:
```shell
cd scripts
bash run_tmd_exps_final_3.sh
```

4. For the fact-cheking task:
```shell
cd scripts
bash fact_checking.sh
```


### Methods

## Citation
```
@article{vazhentsev2025token,
  title={Token-Level Density-Based Uncertainty Quantification Methods for Eliciting Truthfulness of Large Language Models},
  author={Vazhentsev, Artem and Rvanova, Lyudmila and Lazichny, Ivan and Panchenko, Alexander and Panov, Maxim and Baldwin, Timothy and Shelmanov, Artem},
  journal={arXiv preprint arXiv:2502.14427},
  year={2025}
}
```