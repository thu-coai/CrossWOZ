# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from math import log10
from typing import Dict, Optional, List, Any

import allennlp.nn.util as util
import numpy as np
import torch
from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import Vocabulary
from allennlp.data.dataset_readers.dataset_utils.span_utils import bio_tags_to_spans
from allennlp.models.model import Model
from allennlp.modules import Attention, ConditionalRandomField, FeedForward
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.modules.attention import LegacyAttention
from allennlp.modules.conditional_random_field import allowed_transitions
from allennlp.modules.similarity_functions import SimilarityFunction
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import sequence_cross_entropy_with_logits
from allennlp.training.metrics import SpanBasedF1Measure
from overrides import overrides
from torch.nn.modules.linear import Linear

from convlab2.nlu.milu.dai_f1_measure import DialogActItemF1Measure
from convlab2.nlu.milu.multilabel_f1_measure import MultiLabelF1Measure


@Model.register("milu")
class MILU(Model):
    """
    The ``MILU`` encodes a sequence of text with a ``Seq2SeqEncoder``,
    then performs multi-label classification for closed-class dialog act items and 
    sequence labeling to predict a tag for each token in the sequence.

    Parameters
    ----------
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 intent_encoder: Seq2SeqEncoder = None,
                 tag_encoder: Seq2SeqEncoder = None,
                 attention: Attention = None,
                 attention_function: SimilarityFunction = None,
                 context_for_intent: bool = True,
                 context_for_tag: bool = True,
                 attention_for_intent: bool = True,
                 attention_for_tag: bool = True,
                 sequence_label_namespace: str = "labels",
                 intent_label_namespace: str = "intent_labels",
                 feedforward: Optional[FeedForward] = None,
                 label_encoding: Optional[str] = None,
                 include_start_end_transitions: bool = True,
                 crf_decoding: bool = False,
                 constrain_crf_decoding: bool = None,
                 focal_loss_gamma: float = None,
                 nongeneral_intent_weight: float = 5.,
                 num_train_examples: float = None,
                 calculate_span_f1: bool = None,
                 dropout: Optional[float] = None,
                 verbose_metrics: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self.context_for_intent = context_for_intent
        self.context_for_tag = context_for_tag
        self.attention_for_intent = attention_for_intent
        self.attention_for_tag = attention_for_tag
        self.sequence_label_namespace = sequence_label_namespace
        self.intent_label_namespace = intent_label_namespace
        self.text_field_embedder = text_field_embedder
        self.num_tags = self.vocab.get_vocab_size(sequence_label_namespace)
        self.num_intents = self.vocab.get_vocab_size(intent_label_namespace)
        self.encoder = encoder
        self.intent_encoder = intent_encoder
        self.tag_encoder = intent_encoder
        self._feedforward = feedforward
        self._verbose_metrics = verbose_metrics
        self.rl = False 
 
        if attention:
            if attention_function:
                raise ConfigurationError("You can only specify an attention module or an "
                                         "attention function, but not both.")
            self.attention = attention
        elif attention_function:
            self.attention = LegacyAttention(attention_function)

        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None

        projection_input_dim = feedforward.get_output_dim() if self._feedforward else self.encoder.get_output_dim()
        if self.context_for_intent:
            projection_input_dim += self.encoder.get_output_dim()
        if self.attention_for_intent:
            projection_input_dim += self.encoder.get_output_dim()
        self.intent_projection_layer = Linear(projection_input_dim, self.num_intents)

        if num_train_examples:
            try:
                pos_weight = torch.tensor([log10((num_train_examples - self.vocab._retained_counter[intent_label_namespace][t]) / 
                                self.vocab._retained_counter[intent_label_namespace][t]) for i, t in 
                                self.vocab.get_index_to_token_vocabulary(intent_label_namespace).items()])
            except:
                pos_weight = torch.tensor([1. for i, t in 
                                self.vocab.get_index_to_token_vocabulary(intent_label_namespace).items()])
        else:
            # pos_weight = torch.tensor([(lambda t: 1. if "general" in t else nongeneral_intent_weight)(t) for i, t in 
            pos_weight = torch.tensor([(lambda t: nongeneral_intent_weight if "Request" in t else 1.)(t) for i, t in 
                            self.vocab.get_index_to_token_vocabulary(intent_label_namespace).items()])
        self.intent_loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

        tag_projection_input_dim = feedforward.get_output_dim() if self._feedforward else self.encoder.get_output_dim()
        if self.context_for_tag:
            tag_projection_input_dim += self.encoder.get_output_dim()
        if self.attention_for_tag:
            tag_projection_input_dim += self.encoder.get_output_dim()
        self.tag_projection_layer = TimeDistributed(Linear(tag_projection_input_dim,
                                                           self.num_tags))

        # if  constrain_crf_decoding and calculate_span_f1 are not
        # provided, (i.e., they're None), set them to True
        # if label_encoding is provided and False if it isn't.
        if constrain_crf_decoding is None:
            constrain_crf_decoding = label_encoding is not None
        if calculate_span_f1 is None:
            calculate_span_f1 = label_encoding is not None

        self.label_encoding = label_encoding
        if constrain_crf_decoding:
            if not label_encoding:
                raise ConfigurationError("constrain_crf_decoding is True, but "
                                         "no label_encoding was specified.")
            labels = self.vocab.get_index_to_token_vocabulary(sequence_label_namespace)
            constraints = allowed_transitions(label_encoding, labels)
        else:
            constraints = None

        self.include_start_end_transitions = include_start_end_transitions
        if crf_decoding:
            self.crf = ConditionalRandomField(
                    self.num_tags, constraints,
                    include_start_end_transitions=include_start_end_transitions
            )
        else:
            self.crf = None

        self._intent_f1_metric = MultiLabelF1Measure(vocab,
                                                namespace=intent_label_namespace)
        self.calculate_span_f1 = calculate_span_f1
        if calculate_span_f1:
            if not label_encoding:
                raise ConfigurationError("calculate_span_f1 is True, but "
                                          "no label_encoding was specified.")
            self._f1_metric = SpanBasedF1Measure(vocab,
                                                 tag_namespace=sequence_label_namespace,
                                                 label_encoding=label_encoding)
        self._dai_f1_metric = DialogActItemF1Measure()

        check_dimensions_match(text_field_embedder.get_output_dim(), encoder.get_input_dim(),
                               "text field embedding dim", "encoder input dim")
        if feedforward is not None:
            check_dimensions_match(encoder.get_output_dim(), feedforward.get_input_dim(),
                                   "encoder output dim", "feedforward input dim")
        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                context_tokens: Dict[str, torch.LongTensor],
                tokens: Dict[str, torch.LongTensor],
                tags: torch.LongTensor = None,
                intents: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None,
                # pylint: disable=unused-argument
                **kwargs) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------

        Returns
        -------
        """
        if self.context_for_intent or self.context_for_tag or \
            self.attention_for_intent or self.attention_for_tag:
            embedded_context_input = self.text_field_embedder(context_tokens)

            if self.dropout:
                embedded_context_input = self.dropout(embedded_context_input)

            context_mask = util.get_text_field_mask(context_tokens)
            encoded_context = self.encoder(embedded_context_input, context_mask)

            if self.dropout:
                encoded_context = self.dropout(encoded_context)

            encoded_context_summary = util.get_final_encoder_states(
                encoded_context,
                context_mask,
                self.encoder.is_bidirectional())

        embedded_text_input = self.text_field_embedder(tokens)
        mask = util.get_text_field_mask(tokens)

        if self.dropout:
            embedded_text_input = self.dropout(embedded_text_input)

        encoded_text = self.encoder(embedded_text_input, mask)

        if self.dropout:
            encoded_text = self.dropout(encoded_text)

        intent_encoded_text = self.intent_encoder(encoded_text, mask) if self.intent_encoder else encoded_text

        if self.dropout and self.intent_encoder:
            intent_encoded_text = self.dropout(intent_encoded_text)

        is_bidirectional = self.intent_encoder.is_bidirectional() if self.intent_encoder else self.encoder.is_bidirectional()
        if self._feedforward is not None:
            encoded_summary = self._feedforward(util.get_final_encoder_states(
                intent_encoded_text,
                mask,
                is_bidirectional))
        else:
            encoded_summary = util.get_final_encoder_states(
                intent_encoded_text,
                mask,
                is_bidirectional)
        
        tag_encoded_text = self.tag_encoder(encoded_text, mask) if self.tag_encoder else encoded_text

        if self.dropout and self.tag_encoder:
            tag_encoded_text = self.dropout(tag_encoded_text)

        if self.attention_for_intent or self.attention_for_tag:
            attention_weights = self.attention(encoded_summary, encoded_context, context_mask.float())
            attended_context = util.weighted_sum(encoded_context, attention_weights)

        if self.context_for_intent:
            encoded_summary = torch.cat([encoded_summary, encoded_context_summary], dim=-1)
        
        if self.attention_for_intent:
            encoded_summary = torch.cat([encoded_summary, attended_context], dim=-1)

        if self.context_for_tag:
            tag_encoded_text = torch.cat([tag_encoded_text, 
                encoded_context_summary.unsqueeze(dim=1).expand(
                    encoded_context_summary.size(0),
                    tag_encoded_text.size(1),
                    encoded_context_summary.size(1))], dim=-1)

        if self.attention_for_tag:
            tag_encoded_text = torch.cat([tag_encoded_text, 
                attended_context.unsqueeze(dim=1).expand(
                    attended_context.size(0),
                    tag_encoded_text.size(1),
                    attended_context.size(1))], dim=-1)

        intent_logits = self.intent_projection_layer(encoded_summary)
        intent_probs = torch.sigmoid(intent_logits)
        predicted_intents = (intent_probs > 0.5).long()

        sequence_logits = self.tag_projection_layer(tag_encoded_text)
        if self.crf is not None:
            best_paths = self.crf.viterbi_tags(sequence_logits, mask)
            # Just get the tags and ignore the score.
            predicted_tags = [x for x, y in best_paths]
        else:
            predicted_tags = self.get_predicted_tags(sequence_logits)

        output = {"sequence_logits": sequence_logits, "mask": mask, "tags": predicted_tags,
        "intent_logits": intent_logits, "intent_probs": intent_probs, "intents": predicted_intents}

        if tags is not None:
            if self.crf is not None:
                # Add negative log-likelihood as loss
                log_likelihood = self.crf(sequence_logits, tags, mask)
                output["loss"] = -log_likelihood

                # Represent viterbi tags as "class probabilities" that we can
                # feed into the metrics
                class_probabilities = sequence_logits * 0.
                for i, instance_tags in enumerate(predicted_tags):
                    for j, tag_id in enumerate(instance_tags):
                        class_probabilities[i, j, tag_id] = 1
            else:
                loss = sequence_cross_entropy_with_logits(sequence_logits, tags, mask)
                class_probabilities = sequence_logits
                output["loss"] = loss

            if self.calculate_span_f1:
                self._f1_metric(class_probabilities, tags, mask.float())
        
        if metadata is not None:
            output["words"] = [x["words"] for x in metadata]

        if tags is not None and metadata:
            self.decode(output)
            self._dai_f1_metric(output["dialog_act"], [x["dialog_act"] for x in metadata])
            rewards = self.get_rewards(output["dialog_act"], [x["dialog_act"] for x in metadata]) if self.rl else None

        if intents is not None:
            output["loss"] += torch.mean(self.intent_loss(intent_logits, intents.float()))
            self._intent_f1_metric(predicted_intents, intents)

        return output


    def get_predicted_tags(self, sequence_logits: torch.Tensor) -> torch.Tensor:
        """
        Does a simple position-wise argmax over each token, converts indices to string labels, and
        adds a ``"tags"`` key to the dictionary with the result.
        """
        all_predictions = sequence_logits
        all_predictions = all_predictions.detach().cpu().numpy()
        if all_predictions.ndim == 3:
            predictions_list = [all_predictions[i] for i in range(all_predictions.shape[0])]
        else:
            predictions_list = [all_predictions]
        all_tags = []
        for predictions in predictions_list:
            tags = np.argmax(predictions, axis=-1)
            all_tags.append(tags)
        return all_tags
 

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Converts the tag ids to the actual tags.
        ``output_dict["tags"]`` is a list of lists of tag_ids,
        so we use an ugly nested list comprehension.
        """
        output_dict["tags"] = [
                [self.vocab.get_token_from_index(tag, namespace=self.sequence_label_namespace)
                 for tag in instance_tags]
                for instance_tags in output_dict["tags"]
        ]
        output_dict["intents"] = [
                [self.vocab.get_token_from_index(intent[0], namespace=self.intent_label_namespace) 
            for intent in instance_intents.nonzero().tolist()] 
            for instance_intents in output_dict["intents"]
        ]

        output_dict["dialog_act"] = []
        for i, tags in enumerate(output_dict["tags"]): 
            seq_len = len(output_dict["words"][i])
            spans = bio_tags_to_spans(tags[:seq_len])
            dialog_act = {}
            for span in spans:
                domain_act = span[0].split("+")[0]
                slot = span[0].split("+")[1]
                value = " ".join(output_dict["words"][i][span[1][0]:span[1][1]+1])
                if domain_act not in dialog_act:
                    dialog_act[domain_act] = [[slot, value]]
                else:
                    dialog_act[domain_act].append([slot, value])
            for intent in output_dict["intents"][i]:
                if "+" in intent: 
                    if "*" in intent: 
                        intent, value = intent.split("*", 1) 
                    else:
                        value = "?"
                    domain_act = intent.split("+")[0] 
                    if domain_act not in dialog_act:
                        dialog_act[domain_act] = [[intent.split("+")[1], value]]
                    else:
                        dialog_act[domain_act].append([intent.split("+")[1], value])
                else:
                    dialog_act[intent] = [["none", "none"]]
            output_dict["dialog_act"].append(dialog_act)

        return output_dict


    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_to_return = {}
        intent_f1_dict = self._intent_f1_metric.get_metric(reset=reset)
        metrics_to_return.update({"int_"+x[:1]: y for x, y in intent_f1_dict.items() if "overall" in x})
        if self.calculate_span_f1:
            f1_dict = self._f1_metric.get_metric(reset=reset)
            metrics_to_return.update({"tag_"+x[:1]: y for x, y in f1_dict.items() if "overall" in x})
        metrics_to_return.update(self._dai_f1_metric.get_metric(reset=reset))
        return metrics_to_return
