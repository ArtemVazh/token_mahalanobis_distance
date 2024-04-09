import torch
import numpy as np

from typing import Dict, List

from .stat_calculator import StatCalculator
from lm_polygraph.utils.model import WhiteboxModel
from lm_polygraph.generation_metrics.alignscore import AlignScore

def get_embeddings_from_output(
    output,
    batch,
    model_type,
    hidden_state: List[str] = ["encoder", "decoder"],
    ignore_padding: bool = True,
    use_averaging: bool = True,
    all_layers: bool = False,
    aggregation_method: str = "mean",
    level: str = "sequence",
):
    batch_embeddings = None
    batch_embeddings_decoder = None
    batch_size = len(batch["input_ids"])

    if model_type == "CausalLM":
        if not all_layers:
            hidden_layer = -1
            input_tokens_hs = output.hidden_states[0][hidden_layer].cpu().detach()
            if len(output.hidden_states) > 1:
                generated_tokens_hs = torch.cat(
                    [h[hidden_layer].cpu().detach() for h in output.hidden_states[1:]],
                    dim=1,
                )
        else:
            input_tokens_hs = output.hidden_states[0].mean(axis=0).cpu().detach()
            if len(output.hidden_states) > 1:
                generated_tokens_hs = torch.cat(
                    [h.mean(axis=0).cpu().detach() for h in output.hidden_states[1:]],
                    dim=1,
                )
        if len(output.hidden_states) > 1:
            if level == "sequence":
                batch_embeddings_decoder = (
                    torch.cat([input_tokens_hs, generated_tokens_hs], dim=1)
                    .mean(axis=1)
                    .cpu()
                    .detach()
                )
            elif level == "token":
                batch_embeddings_decoder = (
                    torch.cat([input_tokens_hs[:, -1:], generated_tokens_hs], dim=1)
                    .cpu()
                    .detach()
                )
        else:
            batch_embeddings_decoder = input_tokens_hs.mean(axis=1).cpu().detach()
        batch_embeddings = None
    elif model_type == "Seq2SeqLM":
        if use_averaging:
            if "decoder" in hidden_state:
                try:
                    decoder_hidden_states = torch.stack(
                        [torch.stack(hidden) for hidden in output.decoder_hidden_states]
                    )
                    if all_layers:
                        agg_decoder_hidden_states = decoder_hidden_states[
                            :, :, :, 0
                        ].mean(axis=1)
                    else:
                        agg_decoder_hidden_states = decoder_hidden_states[:, -1, :, 0]

                    batch_embeddings_decoder = aggregate(
                        agg_decoder_hidden_states, aggregation_method, axis=0
                    )
                    batch_embeddings_decoder = (
                        batch_embeddings_decoder.cpu()
                        .detach()
                        .reshape(batch_size, -1, agg_decoder_hidden_states.shape[-1])[
                            :, 0
                        ]
                    )
                except TypeError:
                    if all_layers:
                        agg_decoder_hidden_states = torch.stack(
                            output.decoder_hidden_states
                        ).mean(axis=0)
                    else:
                        agg_decoder_hidden_states = torch.stack(
                            output.decoder_hidden_states
                        )[-1]

                    batch_embeddings_decoder = aggregate(
                        agg_decoder_hidden_states, aggregation_method, axis=1
                    )
                    batch_embeddings_decoder = (
                        batch_embeddings_decoder.cpu()
                        .detach()
                        .reshape(-1, agg_decoder_hidden_states.shape[-1])
                    )

            if "encoder" in hidden_state:
                mask = batch["attention_mask"][:, :, None].cpu().detach()
                seq_lens = batch["attention_mask"].sum(-1)[:, None].cpu().detach()
                if all_layers:
                    encoder_embeddings = (
                        aggregate(
                            torch.stack(output.encoder_hidden_states), "mean", axis=0
                        )
                        .cpu()
                        .detach()
                        * mask
                    )
                else:
                    encoder_embeddings = (
                        output.encoder_hidden_states[-1].cpu().detach() * mask
                    )

                if ignore_padding:
                    if aggregation_method == "mean":
                        batch_embeddings = (encoder_embeddings).sum(
                            1
                        ).cpu().detach() / seq_lens
                    else:
                        batch_embeddings = (
                            aggregate(encoder_embeddings, aggregation_method, axis=1)
                            .cpu()
                            .detach()
                        )
                else:
                    batch_embeddings = (
                        aggregate(encoder_embeddings, aggregation_method, axis=1)
                        .cpu()
                        .detach()
                    )
            if not ("encoder" in hidden_state) and not ("decoder" in hidden_state):
                raise NotImplementedError
        else:
            if "decoder" in hidden_state:
                decoder_hidden_states = torch.stack(
                    [torch.stack(hidden) for hidden in output.decoder_hidden_states]
                )
                last_decoder_hidden_states = decoder_hidden_states[-1, -1, :, 0]
                batch_embeddings_decoder = (
                    last_decoder_hidden_states.reshape(
                        batch_size, -1, last_decoder_hidden_states.shape[-1]
                    )[:, 0]
                    .cpu()
                    .detach()
                )
            if "encoder" in hidden_state:
                batch_embeddings = output.encoder_hidden_states[-1][:, 0].cpu().detach()
            if not ("encoder" in hidden_state) and not ("decoder" in hidden_state):
                raise NotImplementedError
    else:
        raise NotImplementedError

    return batch_embeddings, batch_embeddings_decoder


def aggregate(x, aggregation_method, axis):
    if aggregation_method == "max":
        return x.max(axis=axis).values
    elif aggregation_method == "mean":
        return x.mean(axis=axis)
    elif aggregation_method == "sum":
        return x.sum(axis=axis)


class EmbeddingsCalculator(StatCalculator):
    def __init__(self):
        super().__init__(["train_embeddings", "background_train_embeddings", "train_token_embeddings", "background_train_token_embeddings", "train_token_metrics"], [])
        self.hidden_layer = -1
        self.alignscore = AlignScore()

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        batch: Dict[str, torch.Tensor] = model.tokenize(texts)
        batch = {k: v.to(model.device()) for k, v in batch.items()}
        with torch.no_grad():
            out = model.generate(
                **batch,
                output_scores=True,
                return_dict_in_generate=True,
                max_new_tokens=max_new_tokens,
                min_new_tokens=1,
                output_attentions=False,
                output_hidden_states=True,
                num_beams=1,
                suppress_tokens=(
                    []
                    if model.parameters.allow_newlines
                    else [
                        t
                        for t in range(len(model.tokenizer))
                        if "\n" in model.tokenizer.decode([t])
                    ]
                ),
            )
            sequences = out.sequences
            cut_texts = []
            cut_sequences = []
            for i in range(len(texts)):
                if model.model_type == "CausalLM":
                    idx = batch["input_ids"].shape[1]
                    seq = sequences[i, idx:].cpu()
                else:
                    seq = sequences[i, 1:].cpu()
                length, text_length = len(seq), len(seq)
                for j in range(len(seq)):
                    if seq[j] == model.tokenizer.eos_token_id:
                        length = j + 1
                        text_length = j
                        break
                cut_texts.append(model.tokenizer.decode(seq[:text_length]))
                cut_sequences.append(seq[:length].tolist())
            
            stats = {"greedy_texts": cut_texts, "target_texts": dependencies["target_texts"]}
            scores = self.alignscore(stats, None, None)
            tokenwise_scores = [[score]*len(cut_sequences[i]) for i, score in enumerate(scores)]
            
            embeddings_encoder, embeddings_decoder = get_embeddings_from_output(
                out, batch, model.model_type
            )
            token_embeddings_encoder, token_embeddings_decoder = get_embeddings_from_output(
                out, batch, model.model_type, level="token"
            )
            token_embeddings_decoder = token_embeddings_decoder.reshape(-1, model.model.config.hidden_size)

        if model.model_type == "CausalLM":
            return {
                "embeddings_decoder": embeddings_decoder.cpu().detach().numpy(),
                "token_embeddings_decoder": token_embeddings_decoder.cpu().detach().numpy(),
                "token_metrics": tokenwise_scores,
            }
        elif model.model_type == "Seq2SeqLM":
            return {
                "embeddings_encoder": embeddings_encoder.cpu().detach().numpy(),
                "embeddings_decoder": embeddings_decoder.cpu().detach().numpy(),
            }
        else:
            raise NotImplementedError
