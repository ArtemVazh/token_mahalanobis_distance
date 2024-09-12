import numpy as np
from lm_polygraph.generation_metrics.alignscore_utils import AlignScorer

import torch
from typing import List, Dict
from lm_polygraph.generation_metrics.generation_metric import GenerationMetric


class AlignScore(GenerationMetric):
    """
    Calculates AlignScore metric (https://aclanthology.org/2023.acl-long.634/)
    between model-generated texts and ground truth texts.
    """

    def __init__(
        self,
        lang="en",
        ckpt_path="https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-large.ckpt",
        batch_size=16,
        return_mean=False,
        return_inverse=False,
    ):
        super().__init__(["greedy_texts", "input_texts"], "sequence")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scorer = AlignScorer(
            model="roberta-large",
            batch_size=batch_size,
            device=device,
            ckpt_path=ckpt_path,
            evaluation_mode="nli_sp",
            verbose=False,
        )
        self.return_mean = return_mean
        self.return_inverse = return_inverse

    def __str__(self):
        if self.return_mean:
            return "AlignScoreMean"
        if self.return_inverse:
            return "AlignScoreInv"
        return "AlignScore"

    def __call__(
        self,
        stats: Dict[str, np.ndarray],
        target_texts: List[str],
        target_tokens: List[List[int]],
    ) -> np.ndarray:
        """
        Calculates AlignScore (https://aclanthology.org/2023.acl-long.634/) between
        stats['greedy_texts'], and target_texts.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * model-generated texts in 'greedy_texts'
            target_texts (List[str]): ground-truth texts
            target_tokens (List[List[int]]): corresponding token splits for each target text
        Returns:
            np.ndarray: list of AlignScore Scores for each sample in input.
        """
        scores = np.array(
            self.scorer.score(
                claims=target_texts,
                contexts=stats["greedy_texts"],
            )
        )
        if self.return_mean or self.return_inverse:
            scores_ = np.array(
                self.scorer.score(
                    claims=[x if len(x.strip()) else "-" for x in stats["greedy_texts"]],
                    contexts=target_texts,
                )
            )
            
        if self.return_mean:
            return (scores + scores_) / 2
        if self.return_inverse:
            return scores_
        return score