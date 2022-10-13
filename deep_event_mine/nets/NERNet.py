# -*- coding: utf-8 -*-
import numpy as np
import random
import torch
import torch.nn as nn
from torch.nn import BCELoss

from deep_event_mine.bert.modeling import BertModel, BertPreTrainedModel


class NestedNERModel(BertPreTrainedModel):
    def __init__(self, config, params):
        super(NestedNERModel, self).__init__(config)

        self.params = params
        self.config = config

        # Consider only the top ner_label_limit predictions for NER labeling,
        # which are above the ner_threshold
        self.ner_label_limit = params["ner_label_limit"]
        self.thresholds = params["ner_threshold"]

        # Params used to set the shape of the classification linear layers
        self.num_entities = params["mappings"]["nn_mapping"]["num_entities"]
        self.num_triggers = params["mappings"]["nn_mapping"]["num_triggers"]

        # Max span width for entity/trigger prediction in terms of number of tokens
        self.max_span_width = params["max_span_width"]

        self.bert = BertModel(config).to(params['device'])

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Classification layers on top of BERT
        self.entity_classifier = nn.Linear(config.hidden_size * 3, self.num_entities)
        self.trigger_classifier = nn.Linear(config.hidden_size * 3, self.num_triggers)

        # Create a persistent buffer label_ids within the module
        # Multilabel binarizer as cache --> converts an iterable into a
        # binary matrix indicating the presence of a class label
        self.register_buffer(
            "label_ids",
            torch.tensor(
                params["mappings"]["nn_mapping"]["mlb"].classes_, dtype=torch.uint8
            ),
        )

        self.apply(self.init_bert_weights)
        self.params = params
        self.bce = BCELoss()

    def forward(
            self,
            all_tokens,
            all_ids,
            all_token_masks,
            all_attention_masks,
            all_entity_masks,
            all_trigger_masks,
            all_span_labels=None,
            balance_data=False
    ):

        # - all_token_masks (BERT mask)
        #   0s for special tokens and padding
        #   E.g. tensor([[0, 1, 1, ..., 0, 0, 0],
        #                ...
        #                [0, 1, 1, ..., 0, 0, 0]], dtype=torch.uint8)
        #   shape: (number of batches, batch_size), e.g. (16, 66)

        device = all_ids.device
        max_span_width = self.max_span_width

        embeddings, sentence_embedding = self.bert(
            all_ids, attention_mask=all_attention_masks, output_all_encoded_layers=False
        )  # (B, S, H)

        # - embeddings.shape (number of batches, size of each batch, bert_dim)
        #   E.g., (16, 66, 768)
        # - sentence_embedding (number of batches, bert_dim)
        #   E.g., (16, 768) --> one embedding for each sentence

        flattened_token_masks = all_token_masks.flatten()  # (B * S, )
        # E.g., (16x66,) = (1056,)

        flattened_embedding_indices = torch.arange(
            flattened_token_masks.size(0), device=device  # Size(0) = first and unique dimension = e.g. 1056
        ).masked_select(
            flattened_token_masks.bool()
        )  # (all_actual_tokens, )
        # (497) on 1056, i.e. effective textual tokens

        # We get the embeddings for the above indices
        flattened_embeddings = torch.index_select(
            embeddings.view(-1, embeddings.size(-1)), 0, flattened_embedding_indices
        )  # (all_actual_tokens, H)
        # (497, 768)

        # Build the util matrix to test each span combination
        # torch.arange give us a single row with many columns, from 0 to N-1
        # view(-1, 1) give us a flat structure with one column and many rows (i.e., T)
        # E.g. [0, 1, 2] --> [[0],
        #					  [1],
        #                     [2]]
        # we repeat the column max_span_width (e.g., 14) times
        # [[0, 0, ..., 0],
        #  [1, 1, ..., 1],
        #  ...
        #  [496, 496, ..., 496]]
        # (497, 14)
        span_starts = (
            torch.arange(flattened_embeddings.size(0), device=device)
                .view(-1, 1)
                .repeat(1, max_span_width)
        )  # (all_actual_tokens, max_span_width)

        flattened_span_starts = (
            span_starts.flatten()
        )  # (all_actual_tokens * max_span_width, )
        # E.g. [0...496] repeat max_span_width=14 times in a single row

        # Add a shift to each column
        span_ends = span_starts + torch.arange(max_span_width, device=device).view(
            1, -1
        )  # (all_actual_tokens, max_span_width)
        # [[0, 1, ..., 13],
        #  [1, 2, ..., 14],
        #  ...
        #  [496, 497, ..., 509]]

        flattened_span_ends = (
            span_ends.flatten()
        )  # (all_actual_tokens * max_span_width, )

        sentence_indices = (
            torch.arange(embeddings.size(0), device=device)
                .view(-1, 1)
                .repeat(1, embeddings.size(1))
        )  # (B, S)
        # [[0, 0, ..., 0],
        #  [1, 1, ..., 1],
        #  ...
        #  [15, 15, ..., 15]]

        # For each token, keep track of the sentence id
        flattened_sentence_indices = sentence_indices.flatten().masked_select(
            flattened_token_masks.bool()
        )  # (all_actual_tokens, )

        span_start_sentence_indices = torch.index_select(
            flattened_sentence_indices, 0, flattened_span_starts
        )  # (all_actual_tokens * max_span_width, )
        # E.g., [0, 0, 0, ..., 15, 15, 15]
        # (6958, )

        span_end_sentence_indices = torch.index_select(
            flattened_sentence_indices,
            0,
            torch.min(
                flattened_span_ends,
                torch.ones(
                    flattened_span_ends.size(),
                    dtype=flattened_span_ends.dtype,
                    device=device,
                )
                * (span_ends.size(0) - 1),
            ),
        )  # (all_actual_tokens * max_span_width, )

        # Build a mask with 1 for valid spans
        # - Start and end in the same sentence
        # - Delete invalid spans created after shifting
        #   E.g., last token id = 498 where max is 496
        candidate_mask = torch.eq(
            span_start_sentence_indices,
            span_end_sentence_indices,  # Checking both indices is in the same sentence
        ) & torch.lt(
            flattened_span_ends, span_ends.size(0)
        )  # (all_actual_tokens * max_span_width, )

        flattened_span_starts = flattened_span_starts.masked_select(
            candidate_mask
        )  # (all_valid_spans, )

        flattened_span_ends = flattened_span_ends.masked_select(
            candidate_mask
        )  # (all_valid_spans, )

        span_start_embeddings = torch.index_select(
            flattened_embeddings, 0, flattened_span_starts
        )  # (all_valid_spans, H)

        span_end_embeddings = torch.index_select(
            flattened_embeddings, 0, flattened_span_ends
        )  # (all_valid_spans, H)

        # For computing embedding mean
        mean_indices = flattened_span_starts.view(-1, 1) + torch.arange(
            max_span_width, device=device
        ).view(
            1, -1
        )  # (all_valid_spans, max_span_width)

        mean_indices_criteria = torch.gt(
            mean_indices, flattened_span_ends.view(-1, 1).repeat(1, max_span_width)
        )  # (all_valid_spans, max_span_width)

        mean_indices = torch.min(
            mean_indices, flattened_span_ends.view(-1, 1).repeat(1, max_span_width)
        )  # (all_valid_spans, max_span_width)

        span_mean_embeddings = torch.index_select(
            flattened_embeddings, 0, mean_indices.flatten()
        ).view(
            *mean_indices.size(), -1
        )  # (all_valid_spans, max_span_width, H)

        coeffs = torch.ones(
            mean_indices.size(), dtype=embeddings.dtype, device=device
        )  # (all_valid_spans, max_span_width)

        coeffs[mean_indices_criteria] = 0

        span_mean_embeddings = span_mean_embeddings * coeffs.unsqueeze(
            -1
        )  # (all_valid_spans, max_span_width, H)

        span_mean_embeddings = torch.sum(span_mean_embeddings, dim=1) / torch.sum(
            coeffs, dim=-1
        ).view(
            -1, 1
        )  # (all_valid_spans, H)

        combined_embeddings = torch.cat(
            (
                span_start_embeddings,
                span_mean_embeddings,
                span_end_embeddings,
            ),
            dim=1,
        )  # (all_valid_spans, H * 3 + distance_dim)

        all_span_masks = (all_entity_masks > -1) | (
                all_trigger_masks > -1
        )  # (B, max_spans)

        all_entity_masks = all_entity_masks[all_span_masks] > 0  # (all_valid_spans, )

        all_trigger_masks = all_trigger_masks[all_span_masks] > 0  # (all_valid_spans, )

        sentence_sections = all_span_masks.sum(dim=-1).cumsum(dim=-1)  # (B, )

        # The number of possible spans is all_valid_spans = K * (2 * N - K + 1) / 2
        # K: max_span_width
        # N: number of tokens
        # in actual_span_labels we have vector of length num_entities + num_triggers
        # that indicates the label/labels corresponding to each span
        actual_span_labels = all_span_labels[
            all_span_masks
        ]  # (all_valid_spans, num_entities + num_triggers)

        if balance_data:
            actual_span_labels, combined_embeddings, all_entity_masks, all_trigger_masks, balanced_ents = self.balance_dataset(
                actual_span_labels,
                combined_embeddings,
                all_entity_masks,
                all_trigger_masks)
        else:
            balanced_ents = None

        # ! REDUCE
        if self.params['ner_reduce']:
            combined_embeddings = self.reduce(combined_embeddings)

        entity_preds = self.entity_classifier(
            combined_embeddings
        )  # (all_valid_spans, num_entities)

        trigger_preds = self.trigger_classifier(
            combined_embeddings
        )  # (all_valid_spans, num_triggers)

        all_preds = torch.cat(
            (trigger_preds, entity_preds), dim=-1
        )  # (all_valid_spans, num_entities + num_triggers)

        # We could do this due to the independence between variables
        # we have num_entities + num_triggers dimension rather than num_entities + num_triggers +1
        # (one extra dimension for the non-entity label) because we will assign the 0 label
        # in the case where none of the sigmoid scores exceed self.thresholds e.g (0.5 from the paper)
        all_preds = torch.sigmoid(
            all_preds
        )  # (all_valid_spans, num_entities + num_triggers)

        # compute Trigger/Entity loss
        # The 'sum' reduction will be applied, the below code is equivalent to
        # bce = BCELoss(reduction='none')
        # loss_not_reduced = bce(all_preds, actual_span_labels)
        # sum over the feature dimension and then over the examples dimension
        # (loss_not_reduced.sum(1) returns all_valid_spans values
        # loss = loss_not_reduced.sum(1).sum()
        if self.params['compute_dem_loss']:
            loss = self.bce(all_preds, actual_span_labels)
        else:
            loss = None

        all_preds_out = all_preds.clone()

        # Clear values at invalid positions
        all_preds_out[~all_trigger_masks, : self.num_triggers] = 0
        all_preds_out[~all_entity_masks, self.num_triggers:] = 0

        # Support for random-noise adding trick
        entity_coeff = all_entity_masks.sum().float()
        trigger_coeff = all_trigger_masks.sum().float()
        denominator = entity_coeff + trigger_coeff

        entity_coeff /= denominator
        trigger_coeff /= denominator

        # Find top ner_label_limit prediction scores for each span
        _, all_preds_top_indices = torch.topk(all_preds_out, k=self.ner_label_limit, dim=-1)

        # Convert binary value to label ids, if inequality it's false the index takes value 0
        # label 0 is created in this way, if none of the sigmoid scores don't exceed self.thresholds
        # the span is labeled as 0
        all_preds_out = (all_preds_out > self.thresholds) * self.label_ids
        all_golds = (actual_span_labels > 0) * self.label_ids

        # Stupid trick
        all_golds, _ = torch.sort(all_golds, dim=-1, descending=True)
        all_golds = torch.narrow(all_golds, 1, 0, self.ner_label_limit)

        # Convert  all_preds_top_indices to entity/trigger indices
        all_preds_out = torch.gather(all_preds_out, dim=1, index=all_preds_top_indices)

        all_preds_out = all_preds_out.detach().cpu().numpy()
        all_golds = all_golds.detach().cpu().numpy()

        all_aligned_preds = []
        trigger_indices = []
        for idx, (preds, golds) in enumerate(zip(all_preds_out, all_golds)):
            # check trigger in preds
            for pred in preds:
                if pred in self.params['mappings']['nn_mapping']['trTypes_Ids']:
                    trigger_indices.append(idx)
                    break
            aligned_preds = []
            pred_set = set(preds) - {0}
            gold_set = set(golds) - {0}
            shared = pred_set & gold_set
            diff = pred_set - shared
            for gold in golds:
                if gold in shared:
                    aligned_preds.append(gold)
                else:
                    aligned_preds.append(diff.pop() if diff else 0)
            all_aligned_preds.append(aligned_preds)

        all_aligned_preds = np.array(all_aligned_preds)

        if self.training:
            if self.params['compute_metrics']:
                self.compute_metrics(all_aligned_preds, all_golds)

        return (
            all_aligned_preds,
            all_golds,
            sentence_sections,
            all_span_masks,
            combined_embeddings,
            sentence_embedding,
            trigger_indices,
            loss,
            balanced_ents
        )

    def balance_dataset(self, actual_span_labels, combined_embeddings, all_entity_masks, all_trigger_masks):

        balance_mult = self.params['ent_balance_mult']

        positive_ents = []
        negative_ents = []
        for idx, ent in enumerate(actual_span_labels):
            if len(torch.nonzero(ent)) > 0:
                positive_ents.append(idx)
            else:
                negative_ents.append(idx)

        negative_ents = random.sample(negative_ents, balance_mult * len(positive_ents))
        balanced_ents = positive_ents + negative_ents
        balanced_ents.sort()

        actual_span_labels = actual_span_labels[balanced_ents]
        combined_embeddings = combined_embeddings[balanced_ents]
        all_entity_masks = all_entity_masks[balanced_ents]
        all_trigger_masks = all_trigger_masks[balanced_ents]

        return actual_span_labels, combined_embeddings, all_entity_masks, all_trigger_masks, balanced_ents

    def compute_metrics(self, all_aligned_preds, all_golds):

        e_preds_list = all_aligned_preds.flatten().tolist()
        e_golds_list = all_golds.flatten().tolist()

        pos_e_golds_list = [ent for ent in e_golds_list if ent > 0]
        tp = [i for i, ent in enumerate(e_golds_list) if e_golds_list[i] == e_preds_list[i] and ent > 0]
        tp_fp = [ent for ent in e_preds_list if ent > 0]

        if len(pos_e_golds_list) > 0:
            ner_recall = len(tp) / len(pos_e_golds_list)
            print('')
            print('')
            print(f'number of positive matching entities: {len(tp)} out of {len(pos_e_golds_list)} actual positive {ner_recall}')

        if len(tp_fp) > 0:
            ner_precision = len(tp) / len(tp_fp)
            print(
                f'number of positive matching entities: {len(tp)} out of {len(tp_fp)} predicted as positive {ner_precision}')
