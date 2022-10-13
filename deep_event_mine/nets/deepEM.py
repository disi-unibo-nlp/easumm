from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as f
from torch import nn

from deep_event_mine.eval.evalRE import get_entities_info_input, get_entities_info
from deep_event_mine.nets import EVNet
from deep_event_mine.nets import RELNet
from deep_event_mine.nets.NERNet import NestedNERModel
from deep_event_mine.utils import utils


class DeepEM(nn.Module):
    """
    Network architecture
    """

    def __init__(self, params):
        super(DeepEM, self).__init__()

        sizes = params['voc_sizes']
        device = params['device']

        self.NER_layer = NestedNERModel.from_pretrained(params['bert_model'], params=params)
        self.REL_layer = RELNet.RELModel(params, sizes)
        self.EV_layer = EVNet.EVModel(params, sizes)
        self.trigger_id = -1

        if params['train']:
            self.beta = 1
        else:
            self.beta = params['beta']

        if 'loss_bal' in params:
            if params["loss_bal"]:
                self.loss_weights = nn.Parameter(torch.tensor([5, 0.01, 0.01], device=device))
                self.softmax = nn.Softmax(dim=0)

        self.device = device
        self.params = params

    def is_tr(self, label):
        nn_tr_types_ids = self.params['mappings']['nn_mapping']['trTypes_Ids']
        return label in nn_tr_types_ids

    def generate_entity_pairs_4rel(self, bert_out, preds):

        lbls = preds

        # all spans that are an entity ar a trigger
        labeled_spans = (lbls > 0).nonzero(as_tuple=False).transpose(0, 1).long()

        ent_types = torch.full((lbls.shape[0], lbls.shape[1]), -1, dtype=torch.int64, device=self.device)

        e_ids = torch.zeros((lbls.shape[0], lbls.shape[1]), dtype=torch.long)
        tr_ids = torch.zeros((lbls.shape), dtype=torch.int64, device=self.device)

        batch_eids_list = defaultdict(list)
        trig_list = []

        # store only entity in each batch
        batch_ent_list = defaultdict(list)

        # loop over all sentences
        for idx, i in enumerate(labeled_spans[0]):
            j = labeled_spans[1][idx]
            type_a1 = self.params['mappings']['nn_mapping']['tag2type_map'][lbls[i][j].item()]
            ent_types[i][j] = torch.tensor(type_a1, device=self.device)
            e_ids[i][j] = 1

            # check if span is a trigger
            if type_a1 in self.params['trTypes_Ids']:
                tr_ids[i][j] = 1
                # append j-th span in the i-th sentence to list of all the spans labeled as trigger
                trig_list.append((i, j))
            else:
                batch_ent_list[i.item()].append(j)

            batch_eids_list[i.item()].append(j)

        ent_embeds = bert_out.clone()
        tr_embeds = bert_out.clone()
        # vector of zeros for those spans that aren't entities or triggers
        ent_embeds[e_ids == 0] = torch.zeros((bert_out.shape[2]), dtype=bert_out.dtype, device=self.device)
        # vector of zeros for those spans that aren't triggers
        tr_embeds[tr_ids == 0] = torch.zeros((bert_out.shape[2]), dtype=bert_out.dtype, device=self.device)

        pairs_idx = []

        # create pairs of trigger-entity and also trigger-trigger if params['enable_triggers_pair'] = True
        if len(trig_list):
            # loop over all triggers
            # -batch_id indicates the sentence we are referring to
            # -trig_id indicates the trigger id
            for batch_id, trig_id in trig_list:
                if len(batch_eids_list[batch_id.item()]) > 1:

                    # enable relation between triggers
                    if self.params['enable_triggers_pair']:
                        # get all entity ids in this batch
                        b_eids = batch_eids_list[batch_id.item()].copy()

                        # remove this trigger to avoid self relation
                        b_eids.remove(trig_id.clone().detach())

                    # or only between trigger and entity
                    else:
                        # pair with only entity
                        b_eids = batch_ent_list[batch_id.item()].copy()

                    # check empty
                    if len(b_eids) > 0:
                        # make pairs
                        batch_pair_idx = torch.tensor([[batch_id], [trig_id]]).repeat(1, len(b_eids))
                        batch_pair_idx = torch.cat(
                            (batch_pair_idx, torch.tensor(b_eids).view(1, len(b_eids))), dim=0)

                        # add to pairs
                        pairs_idx.append(batch_pair_idx)

            if len(pairs_idx) > 0:
                pairs_idx = torch.cat(pairs_idx, dim=-1)

        return ent_embeds, tr_embeds, ent_types, tr_ids, pairs_idx

    def predicted_span_terms(self, span_terms, e_preds):
        for items in span_terms:
            items.term2id.clear()
            items.id2term.clear()

        # Overwrite triggers
        if self.trigger_id == -1:
            self.trigger_id = utils.get_max_entity_id(span_terms) + 10000

        trigger_idx = self.trigger_id + 1
        for sentence_idx, span_preds in enumerate(e_preds):
            for pred_idx, label_id in enumerate(span_preds):
                if label_id > 0:
                    # if it starts with T it means that it's an entity
                    term = "T" + str(trigger_idx)

                    # check trigger
                    if label_id in self.params['mappings']['nn_mapping']['trTypes_Ids']:
                        # if it starts with TR it means that it's a trigger
                        term = "TR" + str(trigger_idx)

                    span_terms[sentence_idx].id2term[pred_idx] = term
                    span_terms[sentence_idx].term2id[term] = pred_idx
                    trigger_idx += 1

        self.trigger_id = trigger_idx

        return span_terms

    def calculate(self, batch_input):

        # for output
        ner_out = {}

        # input
        nn_tokens, nn_ids, nn_token_mask, nn_attention_mask, nn_span_indices, nn_span_labels, nn_span_labels_match_rel, nn_entity_masks, nn_trigger_masks, span_terms, \
        etypes, max_span_labels = batch_input

        # predict entity
        e_preds, e_golds, sentence_sections, span_masks, embeddings, sentence_emb, trigger_indices, _, _ = self.NER_layer(
            all_tokens=nn_tokens,
            all_ids=nn_ids,
            all_token_masks=nn_token_mask,
            all_attention_masks=nn_attention_mask,
            all_entity_masks=nn_entity_masks,
            all_trigger_masks=nn_trigger_masks,
            all_span_labels=nn_span_labels,
        )

        # run on CPU
        sentence_sections = sentence_sections.detach().cpu().numpy()[:-1]
        all_span_masks = span_masks.detach() > 0

        # Embedding of each span
        embeddings = torch.split(embeddings, torch.sum(all_span_masks, dim=-1).tolist())

        # Pred of each span
        e_preds = np.split(e_preds.astype(int), sentence_sections)
        e_preds = [pred.flatten() for pred in e_preds]
        ner_out['preds'] = e_preds

        e_golds = np.split(e_golds.astype(int), sentence_sections)
        e_golds = [gold.flatten() for gold in e_golds]

        # Note:
        # The gold entities refer to manually id-labeled text spans within the dataset
        # (e.g., cg). For each text span, we have a boolean array of X features where X
        # is the number of entities (i.e., params.full_labels):
        # - a span classified as entity i (within 0-(X-1)), has 1 in position i.
        # - span classification is multi-label: the same span can be labeled as a
        #   non-entity, or with 1-N entity ids.
        # Gold entity labels can be used:
        # (i) to train a NER layer for entity prediction (evaluation and loss)
        #     ner_predict_all = True;
        # (ii) to avoid the NER step and directly use labeled entities.
        #      ner_predict_all = False;
        # In the second case, the NER layer performs only trigger classification for the
        # gold entities.
        # If we apply the model on a raw text for which we do not have entity labels
        # (i.e., golds), we can only set ner_predict_all to true and take advantage of
        # the knowledge learned by a pre-trained DeepEventMine model on an event extraction
        # dataset (e.g., cg).

        # predict both entity and trigger
        if self.params["ner_predict_all"]:
            for items in span_terms:
                items.term2id.clear()
                items.id2term.clear()

            # Overwrite triggers
            if self.trigger_id == -1:
                self.trigger_id = utils.get_max_entity_id(span_terms) + 10000

            trigger_idx = self.trigger_id + 1
            for sentence_idx, span_preds in enumerate(e_preds):
                for pred_idx, label_id in enumerate(span_preds):
                    if label_id > 0:
                        # if it starts with T it means that it's an entity
                        term = "T" + str(trigger_idx)

                        # check trigger
                        if label_id in self.params['mappings']['nn_mapping']['trTypes_Ids']:
                            # if it starts with TR it means that it's a trigger
                            term = "TR" + str(trigger_idx)

                        span_terms[sentence_idx].id2term[pred_idx] = term
                        span_terms[sentence_idx].term2id[term] = pred_idx
                        trigger_idx += 1

            self.trigger_id = trigger_idx

        # given gold entity, predict trigger only
        else:
            # Overwrite triggers
            if self.trigger_id == -1:
                self.trigger_id = utils.get_max_entity_id(span_terms) + 10000

            trigger_idx = self.trigger_id + 1
            for sentence_idx, span_preds in enumerate(e_preds):

                # store gold entity index (a1)
                a1ent_set = set()

                for span_idx, span_term in span_terms[sentence_idx].id2term.items():

                    # replace for entity (using gold entity label)
                    if span_term != "O" and not span_term.startswith("TR") and span_preds[span_idx] != 255:

                        # but do not replace for entity in a2 files
                        span_label = span_terms[sentence_idx].id2label[span_idx]
                        if span_label not in self.params['a2_entities']:
                            span_preds[span_idx] = e_golds[sentence_idx][span_idx]

                            # save this index to ignore prediction
                            a1ent_set.add(span_idx)

                for pred_idx, label_id in enumerate(span_preds):
                    span_term = span_terms[sentence_idx].id2term.get(pred_idx, "O")

                    # if this entity in a1: skip this span
                    if pred_idx in a1ent_set:
                        continue

                    remove_span = False

                    # add prediction for trigger or entity a2
                    if label_id > 0:

                        term = ''

                        # is trigger
                        if self.is_tr(label_id):
                            term = "TR" + str(trigger_idx)

                        # is entity
                        else:
                            etype_label = self.params['mappings']['nn_mapping']['id_tag_mapping'][label_id]

                            # check this entity type in a2 or not
                            if etype_label in self.params['a2_entities']:
                                term = "T" + str(trigger_idx)
                            else:
                                remove_span = True

                        if len(term) > 0:
                            span_terms[sentence_idx].id2term[pred_idx] = term
                            span_terms[sentence_idx].term2id[term] = pred_idx
                            trigger_idx += 1

                    # null prediction
                    if label_id == 0 or remove_span:

                        # do not write anything
                        span_preds[pred_idx] = 0

                        # remove this span
                        if span_term.startswith("T"):
                            del span_terms[sentence_idx].id2term[pred_idx]
                            del span_terms[sentence_idx].term2id[span_term]

                span_preds[span_preds == 255] = 0
            self.trigger_id = trigger_idx

        # what is the maximum number of spans in a sentence
        # multiplied by two given the fact that we have 2 labels for each span
        # e.g 1694
        num_padding = max_span_labels * self.params["ner_label_limit"]

        # for each sentence we get a num_padding vector if the original
        # vectors are shorter we pad them with -1
        # (batch_size, num_padding) e.g (16, 1694)
        e_preds = [np.pad(pred, (0, num_padding - pred.shape[0]),
                          'constant', constant_values=-1) for pred in e_preds]
        e_golds = [np.pad(gold, (0, num_padding - gold.shape[0]),
                          'constant', constant_values=-1) for gold in e_golds]

        e_preds = torch.tensor(e_preds, device=self.device)

        # pad each embedding to max number of spans in a sentence in batch
        embeddings = [f.pad(embedding, (0, 0, 0, max_span_labels - embedding.shape[0]),
                            'constant', value=0) for embedding in embeddings]

        embeddings = torch.stack(embeddings)
        embeddings = embeddings.unsqueeze(dim=2).expand(-1, -1, self.params["ner_label_limit"], -1)

        # we have 2304-dimensional embedding for each span in a sentence. the number of spans is
        # max_number_of_spans in a sentence * 2 because each span can be labeled as 2 different entities
        # like "transformed" in the paper example
        # (batch_size, num_padding, embedding_size) e.g (16, 1694, 2304)
        embeddings = embeddings.reshape(embeddings.size(0), -1, embeddings.size(-1))

        # len(pairs_idx[1])  indicates the number of pairs
        # for each pair in pairs_idx we have 3 vectors:
        # -the first one indicates the sentence
        # -the second one indicates the first element of the pair that must be a trigger
        # -the third one indicates the second element of the pair can be a trigger or entity
        # in span_terms you can find all the id2term relations to learn e.g. that '8' corresponds to 'TR38'
        # the number of possible pairs is given by: sum(n_e_i) for i=1,...,N_t
        # N_t = number of triggers returned by the Trigger/Entity Layer
        # n_e_i = number of entities/triggers in sentence corresponding to the i-th trigger excluding the i-th trigger to avoid self-relation
        # ent_embeds is the vector of embeddings for each span, if the span is a non-entity it's going to have all zeros
        # tr_embeds is the vector of embeddings for each span, if the span is not a trigger it's going to have all zeros
        ent_embeds, tr_embeds, ent_types, tr_ids, pairs_idx = self.generate_entity_pairs_4rel(
            embeddings,
            preds=e_preds
        )

        ner_preds = {'preds': e_preds, 'golds': e_golds, 'embeddings': embeddings,
                     'ent_embeds': ent_embeds, 'tr_embeds': tr_embeds, 'tr_ids': tr_ids,
                     'ent_types': ent_types, 'pairs_idx': pairs_idx, 'e_types': etypes.long(),
                     'sentence_embeds': sentence_emb}

        # Role layer
        rel_preds = self.REL_layer(ner_preds)

        # if there are trigger-entity or trigger-trigger proceed
        if rel_preds['next']:

            ev_preds, empty_pred, _ = self.EV_layer(ner_preds, rel_preds)

            if empty_pred == True:
                ev_preds = None

        else:
            ev_preds = None

        ner_out['terms'] = span_terms
        ner_out['span_indices'] = nn_span_indices

        nner_preds = e_preds.detach().cpu().numpy()
        ner_out['nner_preds'] = nner_preds

        return ner_out, rel_preds, ev_preds

    def forward_train(self, eval_data, eval_data_ids, all_ann_info, balance_ent=False):
        # for output
        ner_out = {}

        # input
        batch_input = utils.get_tensors(eval_data_ids, eval_data, self.params)
        nn_tokens, nn_ids, nn_token_mask, nn_attention_mask, nn_span_indices, nn_span_labels, nn_span_labels_match_rel, nn_entity_masks, nn_trigger_masks, span_terms, \
        etypes, max_span_labels = batch_input

        # predict entity
        e_preds, e_golds, sentence_sections, span_masks, embeddings, sentence_emb, trigger_indices, ner_loss, balanced_ents = self.NER_layer(
            all_tokens=nn_tokens,
            all_ids=nn_ids,
            all_token_masks=nn_token_mask,
            all_attention_masks=nn_attention_mask,
            all_entity_masks=nn_entity_masks,
            all_trigger_masks=nn_trigger_masks,
            all_span_labels=nn_span_labels,
            balance_data=balance_ent
        )

        # run on CPU
        all_span_masks = span_masks.detach() > 0

        if balance_ent:
            sentence_split = sentence_sections

            emb_split_list = torch.sum(all_span_masks, dim=-1).tolist()
            curr_thr = 0
            new_thr_list = []
            for thr in emb_split_list:
                curr_thr += thr
                prev_thr = curr_thr - thr
                new_thr = len([el for el in balanced_ents if el in range(prev_thr, curr_thr)])
                new_thr_list.append(new_thr)

            # Embedding of each span
            embeddings = torch.split(embeddings, new_thr_list)
            sent_sec = 0
            sentence_sections = []
            for i, el in enumerate(new_thr_list):
                sent_sec += el
                sentence_sections.append(sent_sec)

            sentence_sections = np.asarray(sentence_sections[:-1])

            flattened_span_indices = torch.flatten(nn_span_indices, end_dim=1)
            pos_span_indices = flattened_span_indices[flattened_span_indices[:, 0] >= 0]
            balanced_ents1 = [2 * idx for idx in balanced_ents]
            balanced_ents2 = [idx + 1 if (idx % 2) == 0 else idx - 1 for idx in balanced_ents1]
            balanced_ents1.extend(balanced_ents2)
            balanced_ents1.sort()
            bl_span_indices = pos_span_indices[balanced_ents1]
            bl_span_indices = np.split(bl_span_indices, 2 * sentence_sections)
            balanced_ents_sent = np.split(np.asarray(balanced_ents), 2 * sentence_sections)

            for i in range(1, len(balanced_ents_sent)):
                balanced_ents_sent[i] = balanced_ents_sent[i] - sentence_split[i - 1].item()

        else:
            sentence_sections = sentence_sections.detach().cpu().numpy()[:-1]
            embeddings = torch.split(embeddings, torch.sum(all_span_masks, dim=-1).tolist())

        # Pred of each span
        e_preds = np.split(e_preds.astype(int), sentence_sections)
        e_preds = [pred.flatten() for pred in e_preds]
        ner_out['preds'] = e_preds

        e_golds = np.split(e_golds.astype(int), sentence_sections)
        e_golds = [gold.flatten() for gold in e_golds]

        # predict both entity and trigger

        # predict both entity and trigger
        self.params["ner_predict_all"] = True
        if self.params["ner_predict_all"]:
            for items in span_terms:
                items.term2id.clear()
                items.id2term.clear()

            # Overwrite triggers
            if self.trigger_id == -1:
                self.trigger_id = utils.get_max_entity_id(span_terms) + 10000

            trigger_idx = self.trigger_id + 1
            for sentence_idx, span_preds in enumerate(e_preds):
                for pred_idx, label_id in enumerate(span_preds):
                    if label_id > 0:
                        term = "T" + str(trigger_idx)

                        # check trigger
                        if label_id in self.params['mappings']['nn_mapping']['trTypes_Ids']:
                            term = "TR" + str(trigger_idx)

                        span_terms[sentence_idx].id2term[pred_idx] = term
                        span_terms[sentence_idx].term2id[term] = pred_idx
                        trigger_idx += 1

            self.trigger_id = trigger_idx

        # given gold entity, predict trigger only
        else:
            # Overwrite triggers
            if self.trigger_id == -1:
                self.trigger_id = utils.get_max_entity_id(span_terms) + 10000

            trigger_idx = self.trigger_id + 1
            for sentence_idx, span_preds in enumerate(e_preds):

                # store gold entity index (a1)
                a1ent_set = set()

                for span_idx, span_term in span_terms[sentence_idx].id2term.items():

                    # replace for entity (using gold entity label)
                    if span_term != "O" and not span_term.startswith("TR") and span_preds[span_idx] != 255:

                        # but do not replace for entity in a2 files
                        span_label = span_terms[sentence_idx].id2label[span_idx]
                        if span_label not in self.params['a2_entities']:
                            span_preds[span_idx] = e_golds[sentence_idx][span_idx]

                            # save this index to ignore prediction
                            a1ent_set.add(span_idx)

                for pred_idx, label_id in enumerate(span_preds):
                    span_term = span_terms[sentence_idx].id2term.get(pred_idx, "O")

                    # if this entity in a1: skip this span
                    if pred_idx in a1ent_set:
                        continue

                    remove_span = False

                    # add prediction for trigger or entity a2
                    if label_id > 0:

                        term = ''

                        # is trigger
                        if self.is_tr(label_id):
                            term = "TR" + str(trigger_idx)

                        # is entity
                        else:
                            etype_label = self.params['mappings']['nn_mapping']['id_tag_mapping'][label_id]

                            # check this entity type in a2 or not
                            if etype_label in self.params['a2_entities']:
                                term = "T" + str(trigger_idx)
                            else:
                                remove_span = True

                        if len(term) > 0:
                            span_terms[sentence_idx].id2term[pred_idx] = term
                            span_terms[sentence_idx].term2id[term] = pred_idx
                            trigger_idx += 1

                    # null prediction
                    if label_id == 0 or remove_span:

                        # do not write anything
                        span_preds[pred_idx] = 0

                        # remove this span
                        if span_term.startswith("T"):
                            del span_terms[sentence_idx].id2term[pred_idx]
                            del span_terms[sentence_idx].term2id[span_term]

                span_preds[span_preds == 255] = 0
            self.trigger_id = trigger_idx

        ner_out['terms'] = span_terms

        # create a dictionary for each entity where all its relevant information is stored
        ent_ann = get_entities_info_input(eval_data, eval_data_ids, ner_out, nn_span_indices)
        if balance_ent:
            ent_ann['span_indices'] = bl_span_indices
        entity_map = get_entities_info(ent_ann, self.params)

        # what is the maximum number of spans in a sentence
        # multiplied by two given the fact that we have 2 labels for each span
        num_padding = max_span_labels * self.params["ner_label_limit"]

        # for each sentence we get a num_padding vector if the original
        # vectors are shorter we pad them with -1
        e_preds = [np.pad(pred, (0, num_padding - pred.shape[0]),
                          'constant', constant_values=-1) for pred in e_preds]
        e_golds = [np.pad(gold, (0, num_padding - gold.shape[0]),
                          'constant', constant_values=-1) for gold in e_golds]

        e_preds = torch.tensor(e_preds, device=self.device)

        # pad each embedding to max number of spans in a sentence in batch
        embeddings = [f.pad(embedding, (0, 0, 0, max_span_labels - embedding.shape[0]),
                            'constant', value=0) for embedding in embeddings]

        embeddings = torch.stack(embeddings)
        embeddings = embeddings.unsqueeze(dim=2).expand(-1, -1, self.params["ner_label_limit"], -1)

        # we have 2304-dimensional embedding for each span in a sentence. the number of spans is
        # max_number_of_spans in a sentence * 2 because each span can be labeled as 2 different entities
        # like "transformed" in the paper example
        embeddings = embeddings.reshape(embeddings.size(0), -1, embeddings.size(-1))

        # len(pairs_idx[1])  indicates the number of pairs
        # for each pair in pairs_idx we have 3 vectors:
        # -the first one indicates the sentence
        # -the second one indicates the first element of the pair that must be a trigger
        # -the third one indicates the second element of the pair can be a trigger or entity
        # in span_terms you can find all the id2term relations to learn e.g. that '8' corresponds to 'TR38'
        # the number of possible pairs is given by: sum(n_e_i) for i=1,...,N_t
        # N_t = number of triggers returned by the Trigger/Entity Layer
        # n_e_i = number of entities/triggers in sentence corresponding to the i-th trigger excluding the i-th trigger to avoid self-relation
        # ent_embeds is the vector of embeddings for each span, if the span is a non-entity it's going to have all zeros
        # tr_embeds is the vector of embeddings for each span, if the span is not a trigger it's going to have all zeros
        ent_embeds, tr_embeds, ent_types, tr_ids, pairs_idx = self.generate_entity_pairs_4rel(
            embeddings,
            preds=e_preds
        )

        fids = [
            eval_data["fids"][data_id] for data_id in eval_data_ids[0].tolist()
        ]

        ner_preds = {'preds': e_preds, 'golds': e_golds, 'embeddings': embeddings,
                     'ent_embeds': ent_embeds, 'tr_embeds': tr_embeds, 'tr_ids': tr_ids,
                     'ent_types': ent_types, 'pairs_idx': pairs_idx, 'e_types': etypes.long(),
                     'sentence_embeds': sentence_emb, 'fids': fids, 'entity_map': entity_map,
                     'all_ann_info': all_ann_info, 'params': self.params}

        # Role layer
        rel_preds = self.REL_layer(ner_preds)
        rel_preds['entity_map'] = entity_map
        rel_preds['all_ann_info'] = all_ann_info
        rel_preds['fids'] = fids

        if rel_preds['next']:

            ev_preds, empty_pred, ev_loss = self.EV_layer(ner_preds, rel_preds)

            if ev_loss is not None:

                if self.params["loss_bal"]:
                    loss_weights = self.softmax(self.loss_weights)
                    loss = (loss_weights[0] * ner_loss + loss_weights[1] * rel_preds['rel_loss'] + loss_weights[
                        2] * ev_loss) / sum(loss_weights)
                    print(f" \t {loss_weights.data}, {loss_weights[0] * ner_loss}, {loss_weights[1] * rel_preds['rel_loss']}, {loss_weights[2] * ev_loss}")

                else:
                    loss = ner_loss + rel_preds['rel_loss'] + ev_loss


            else:
                loss = ner_loss + rel_preds['rel_loss']

            return loss

        else:

            loss = ner_loss

            return loss

    def warm_up(self, eval_data, eval_data_ids, all_ann_info, balance_ent=False):
        ner_out = {}

        # input
        batch_input = utils.get_tensors(eval_data_ids, eval_data, self.params)
        nn_tokens, nn_ids, nn_token_mask, nn_attention_mask, nn_span_indices, nn_span_labels, nn_span_labels_match_rel, nn_entity_masks, nn_trigger_masks, span_terms, \
        etypes, max_span_labels = batch_input

        # predict entity
        e_preds, e_golds, sentence_sections, span_masks, embeddings, sentence_emb, trigger_indices, ner_loss, balanced_ents = self.NER_layer(
            all_tokens=nn_tokens,
            all_ids=nn_ids,
            all_token_masks=nn_token_mask,
            all_attention_masks=nn_attention_mask,
            all_entity_masks=nn_entity_masks,
            all_trigger_masks=nn_trigger_masks,
            all_span_labels=nn_span_labels,
            balance_data=balance_ent
        )

        all_span_masks = span_masks.detach() > 0

        if balance_ent:
            sentence_split = sentence_sections

            emb_split_list = torch.sum(all_span_masks, dim=-1).tolist()
            curr_thr = 0
            new_thr_list = []
            for thr in emb_split_list:
                curr_thr += thr
                prev_thr = curr_thr - thr
                new_thr = len([el for el in balanced_ents if el in range(prev_thr, curr_thr)])
                new_thr_list.append(new_thr)

            # Embedding of each span
            embeddings = torch.split(embeddings, new_thr_list)
            sent_sec = 0
            sentence_sections = []
            for i, el in enumerate(new_thr_list):
                sent_sec += el
                sentence_sections.append(sent_sec)

            sentence_sections = np.asarray(sentence_sections[:-1])

            flattened_span_indices = torch.flatten(nn_span_indices, end_dim=1)
            pos_span_indices = flattened_span_indices[flattened_span_indices[:, 0] >= 0]
            balanced_ents1 = [2 * idx for idx in balanced_ents]
            balanced_ents2 = [idx + 1 if (idx % 2) == 0 else idx - 1 for idx in balanced_ents1]
            balanced_ents1.extend(balanced_ents2)
            balanced_ents1.sort()
            bl_span_indices = pos_span_indices[balanced_ents1]
            bl_span_indices = np.split(bl_span_indices, 2 * sentence_sections)
            balanced_ents_sent = np.split(np.asarray(balanced_ents), 2 * sentence_sections)

            for i in range(1, len(balanced_ents_sent)):
                balanced_ents_sent[i] = balanced_ents_sent[i] - sentence_split[i - 1].item()

        else:
            sentence_sections = sentence_sections.detach().cpu().numpy()[:-1]
            embeddings = torch.split(embeddings, torch.sum(all_span_masks, dim=-1).tolist())

        e_golds = np.split(e_golds.astype(int), sentence_sections)
        e_golds = [gold.flatten() for gold in e_golds]
        ner_out['preds'] = e_golds

        # use predictions for entities and triggers (DeepEventMine gold) or just triggers
        self.params["ner_predict_all"] = True
        if self.params["ner_predict_all"]:
            for items in span_terms:
                items.term2id.clear()
                items.id2term.clear()

            # Overwrite triggers
            if self.trigger_id == -1:
                self.trigger_id = utils.get_max_entity_id(span_terms) + 10000

            trigger_idx = self.trigger_id + 1
            for sentence_idx, span_preds in enumerate(e_golds):
                for pred_idx, label_id in enumerate(span_preds):
                    if label_id > 0:
                        # if it starts with T it means that it's an entity
                        term = "T" + str(trigger_idx)

                        # check trigger
                        if label_id in self.params['mappings']['nn_mapping']['trTypes_Ids']:
                            # if it starts with TR it means that it's a trigger
                            term = "TR" + str(trigger_idx)

                        span_terms[sentence_idx].id2term[pred_idx] = term
                        span_terms[sentence_idx].term2id[term] = pred_idx
                        trigger_idx += 1

            self.trigger_id = trigger_idx

            # given gold entity, predict trigger only
        else:
            # Overwrite triggers
            if self.trigger_id == -1:
                self.trigger_id = utils.get_max_entity_id(span_terms) + 10000

            trigger_idx = self.trigger_id + 1
            for sentence_idx, span_preds in enumerate(e_golds):

                # store gold entity index (a1)
                a1ent_set = set()

                for span_idx, span_term in span_terms[sentence_idx].id2term.items():

                    # replace for entity (using gold entity label)
                    if span_term != "O" and not span_term.startswith("TR") and span_preds[span_idx] != 255:

                        # but do not replace for entity in a2 files
                        span_label = span_terms[sentence_idx].id2label[span_idx]
                        if span_label not in self.params['a2_entities']:
                            span_preds[span_idx] = e_golds[sentence_idx][span_idx]

                            # save this index to ignore prediction
                            a1ent_set.add(span_idx)

                for pred_idx, label_id in enumerate(span_preds):
                    span_term = span_terms[sentence_idx].id2term.get(pred_idx, "O")

                    # if this entity in a1: skip this span
                    if pred_idx in a1ent_set:
                        continue

                    remove_span = False

                    # add prediction for trigger or entity a2
                    if label_id > 0:

                        term = ''

                        # is trigger
                        if self.is_tr(label_id):
                            term = "TR" + str(trigger_idx)

                        # is entity
                        else:
                            etype_label = self.params['mappings']['nn_mapping']['id_tag_mapping'][label_id]

                            # check this entity type in a2 or not
                            if etype_label in self.params['a2_entities']:
                                term = "T" + str(trigger_idx)
                            else:
                                remove_span = True

                        if len(term) > 0:
                            span_terms[sentence_idx].id2term[pred_idx] = term
                            span_terms[sentence_idx].term2id[term] = pred_idx
                            trigger_idx += 1

                    # null prediction
                    if label_id == 0 or remove_span:

                        # do not write anything
                        span_preds[pred_idx] = 0

                        # remove this span
                        if span_term.startswith("T"):
                            del span_terms[sentence_idx].id2term[pred_idx]
                            del span_terms[sentence_idx].term2id[span_term]

                span_preds[span_preds == 255] = 0
            self.trigger_id = trigger_idx

        ner_out['terms'] = span_terms

        # create a dictionary for each entity where all its relevant information is stored
        ent_ann = get_entities_info_input(eval_data, eval_data_ids, ner_out, nn_span_indices)
        if balance_ent:
            ent_ann['span_indices'] = bl_span_indices
        entity_map = get_entities_info(ent_ann, self.params)

        # for each sentence we get a num_padding vector if the original
        # vectors are shorter we pad them with -1
        num_padding = max_span_labels * self.params["ner_label_limit"]
        e_golds = [np.pad(gold, (0, num_padding - gold.shape[0]),
                          'constant', constant_values=-1) for gold in e_golds]

        e_golds = torch.tensor(e_golds, device=self.device)

        # pad each embedding to max number of spans in a sentence in batch
        embeddings = [f.pad(embedding, (0, 0, 0, max_span_labels - embedding.shape[0]),
                            'constant', value=0) for embedding in embeddings]

        embeddings = torch.stack(embeddings)
        embeddings = embeddings.unsqueeze(dim=2).expand(-1, -1, self.params["ner_label_limit"], -1)

        # we have 2304-dimensional embedding for each span in a sentence. the number of spans is
        # max_number_of_spans in a sentence * 2 because each span can be labeled as 2 different entities
        # like "transformed" in the paper example
        embeddings = embeddings.reshape(embeddings.size(0), -1, embeddings.size(-1))

        ent_embeds, tr_embeds, ent_types, tr_ids, pairs_idx = self.generate_entity_pairs_4rel(
            embeddings,
            preds=e_golds
        )

        fids = [
            eval_data["fids"][data_id] for data_id in eval_data_ids[0].tolist()
        ]

        ner_preds = {'preds': e_golds, 'embeddings': embeddings,
                     'ent_embeds': ent_embeds, 'tr_embeds': tr_embeds, 'tr_ids': tr_ids,
                     'ent_types': ent_types, 'pairs_idx': pairs_idx, 'e_types': etypes.long(),
                     'sentence_embeds': sentence_emb, 'fids': fids,
                     'all_ann_info': all_ann_info, 'params': self.params, 'entity_map': entity_map}

        # Role layer
        rel_preds = self.REL_layer(ner_preds, warm_up=True)

        rel_preds['entity_map'] = entity_map
        rel_preds['all_ann_info'] = all_ann_info
        rel_preds['fids'] = fids

        # avoid gradients to propagate to REL_layer
        # rel_preds['enttoks_type_embeds'] contains the concatenation of span embeddings and type embeddings obtaining
        # v_t = [m_t;s_t] for triggers and v_a = [m_a;s_a] for entities
        # rel_preds['rel_embeds'] contains the pair representations r_i
        rel_preds['enttoks_type_embeds'] = rel_preds['enttoks_type_embeds'].detach_()
        rel_preds['rel_embeds'] = rel_preds['rel_embeds'].detach_()

        if rel_preds['next']:
            ev_preds, empty_pred, ev_loss = self.EV_layer(ner_preds, rel_preds)

            loss = ner_loss + rel_preds['rel_loss'] + ev_loss

            return loss, ner_loss, rel_preds['rel_loss'], ev_loss

        else:
            loss = ner_loss + rel_preds['rel_loss']

            return loss, ner_loss, rel_preds['rel_loss'], None

    def count_labels(self, eval_data, eval_data_ids, all_ann_info, balance_ent=False):
        ner_out = {}

        # input
        batch_input = utils.get_tensors(eval_data_ids, eval_data, self.params)
        nn_tokens, nn_ids, nn_token_mask, nn_attention_mask, nn_span_indices, nn_span_labels, nn_span_labels_match_rel, nn_entity_masks, nn_trigger_masks, span_terms, \
        etypes, max_span_labels = batch_input

        # predict entity
        e_preds, e_golds, sentence_sections, span_masks, embeddings, sentence_emb, trigger_indices, ner_loss, balanced_ents = self.NER_layer(
            all_tokens=nn_tokens,
            all_ids=nn_ids,
            all_token_masks=nn_token_mask,
            all_attention_masks=nn_attention_mask,
            all_entity_masks=nn_entity_masks,
            all_trigger_masks=nn_trigger_masks,
            all_span_labels=nn_span_labels,
            balance_data=balance_ent
        )

        all_span_masks = span_masks.detach() > 0

        if balance_ent:
            sentence_split = sentence_sections

            emb_split_list = torch.sum(all_span_masks, dim=-1).tolist()
            curr_thr = 0
            new_thr_list = []
            for thr in emb_split_list:
                curr_thr += thr
                prev_thr = curr_thr - thr
                new_thr = len([el for el in balanced_ents if el in range(prev_thr, curr_thr)])
                new_thr_list.append(new_thr)

            # Embedding of each span
            embeddings = torch.split(embeddings, new_thr_list)
            sent_sec = 0
            sentence_sections = []
            for i, el in enumerate(new_thr_list):
                sent_sec += el
                sentence_sections.append(sent_sec)

            sentence_sections = np.asarray(sentence_sections[:-1])

            flattened_span_indices = torch.flatten(nn_span_indices, end_dim=1)
            pos_span_indices = flattened_span_indices[flattened_span_indices[:, 0] >= 0]
            balanced_ents1 = [2 * idx for idx in balanced_ents]
            balanced_ents2 = [idx + 1 if (idx % 2) == 0 else idx - 1 for idx in balanced_ents1]
            balanced_ents1.extend(balanced_ents2)
            balanced_ents1.sort()
            bl_span_indices = pos_span_indices[balanced_ents1]
            bl_span_indices = np.split(bl_span_indices, 2 * sentence_sections)
            balanced_ents_sent = np.split(np.asarray(balanced_ents), 2 * sentence_sections)

            for i in range(1, len(balanced_ents_sent)):
                balanced_ents_sent[i] = balanced_ents_sent[i] - sentence_split[i - 1].item()

        else:
            sentence_sections = sentence_sections.detach().cpu().numpy()[:-1]
            embeddings = torch.split(embeddings, torch.sum(all_span_masks, dim=-1).tolist())

        e_golds = np.split(e_golds.astype(int), sentence_sections)
        e_golds = [gold.flatten() for gold in e_golds]
        ner_out['preds'] = e_golds

        # use predictions for entities and triggers (DeepEventMine gold) or just triggers
        self.params["ner_predict_all"] = True
        if self.params["ner_predict_all"]:
            for items in span_terms:
                items.term2id.clear()
                items.id2term.clear()

            # Overwrite triggers
            if self.trigger_id == -1:
                self.trigger_id = utils.get_max_entity_id(span_terms) + 10000

            trigger_idx = self.trigger_id + 1
            for sentence_idx, span_preds in enumerate(e_golds):
                for pred_idx, label_id in enumerate(span_preds):
                    if label_id > 0:
                        # if it starts with T it means that it's an entity
                        term = "T" + str(trigger_idx)

                        # check trigger
                        if label_id in self.params['mappings']['nn_mapping']['trTypes_Ids']:
                            # if it starts with TR it means that it's a trigger
                            term = "TR" + str(trigger_idx)

                        span_terms[sentence_idx].id2term[pred_idx] = term
                        span_terms[sentence_idx].term2id[term] = pred_idx
                        trigger_idx += 1

            self.trigger_id = trigger_idx

            # given gold entity, predict trigger only
        else:
            # Overwrite triggers
            if self.trigger_id == -1:
                self.trigger_id = utils.get_max_entity_id(span_terms) + 10000

            trigger_idx = self.trigger_id + 1
            for sentence_idx, span_preds in enumerate(e_golds):

                # store gold entity index (a1)
                a1ent_set = set()

                for span_idx, span_term in span_terms[sentence_idx].id2term.items():

                    # replace for entity (using gold entity label)
                    if span_term != "O" and not span_term.startswith("TR") and span_preds[span_idx] != 255:

                        # but do not replace for entity in a2 files
                        span_label = span_terms[sentence_idx].id2label[span_idx]
                        if span_label not in self.params['a2_entities']:
                            span_preds[span_idx] = e_golds[sentence_idx][span_idx]

                            # save this index to ignore prediction
                            a1ent_set.add(span_idx)

                for pred_idx, label_id in enumerate(span_preds):
                    span_term = span_terms[sentence_idx].id2term.get(pred_idx, "O")

                    # if this entity in a1: skip this span
                    if pred_idx in a1ent_set:
                        continue

                    remove_span = False

                    # add prediction for trigger or entity a2
                    if label_id > 0:

                        term = ''

                        # is trigger
                        if self.is_tr(label_id):
                            term = "TR" + str(trigger_idx)

                        # is entity
                        else:
                            etype_label = self.params['mappings']['nn_mapping']['id_tag_mapping'][label_id]

                            # check this entity type in a2 or not
                            if etype_label in self.params['a2_entities']:
                                term = "T" + str(trigger_idx)
                            else:
                                remove_span = True

                        if len(term) > 0:
                            span_terms[sentence_idx].id2term[pred_idx] = term
                            span_terms[sentence_idx].term2id[term] = pred_idx
                            trigger_idx += 1

                    # null prediction
                    if label_id == 0 or remove_span:

                        # do not write anything
                        span_preds[pred_idx] = 0

                        # remove this span
                        if span_term.startswith("T"):
                            del span_terms[sentence_idx].id2term[pred_idx]
                            del span_terms[sentence_idx].term2id[span_term]

                span_preds[span_preds == 255] = 0
            self.trigger_id = trigger_idx

        ner_out['terms'] = span_terms

        # create a dictionary for each entity where all its relevant information is stored
        ent_ann = get_entities_info_input(eval_data, eval_data_ids, ner_out, nn_span_indices)
        if balance_ent:
            ent_ann['span_indices'] = bl_span_indices
        entity_map = get_entities_info(ent_ann, self.params)

        # for each sentence we get a num_padding vector if the original
        # vectors are shorter we pad them with -1
        num_padding = max_span_labels * self.params["ner_label_limit"]
        e_golds = [np.pad(gold, (0, num_padding - gold.shape[0]),
                          'constant', constant_values=-1) for gold in e_golds]

        e_golds = torch.tensor(e_golds, device=self.device)

        # pad each embedding to max number of spans in a sentence in batch
        embeddings = [f.pad(embedding, (0, 0, 0, max_span_labels - embedding.shape[0]),
                            'constant', value=0) for embedding in embeddings]

        embeddings = torch.stack(embeddings)
        embeddings = embeddings.unsqueeze(dim=2).expand(-1, -1, self.params["ner_label_limit"], -1)

        # we have 2304-dimensional embedding for each span in a sentence. the number of spans is
        # max_number_of_spans in a sentence * 2 because each span can be labeled as 2 different entities
        # like "transformed" in the paper example
        embeddings = embeddings.reshape(embeddings.size(0), -1, embeddings.size(-1))

        ent_embeds, tr_embeds, ent_types, tr_ids, pairs_idx = self.generate_entity_pairs_4rel(
            embeddings,
            preds=e_golds
        )

        fids = [
            eval_data["fids"][data_id] for data_id in eval_data_ids[0].tolist()
        ]

        ner_preds = {'preds': e_golds, 'embeddings': embeddings,
                     'ent_embeds': ent_embeds, 'tr_embeds': tr_embeds, 'tr_ids': tr_ids,
                     'ent_types': ent_types, 'pairs_idx': pairs_idx, 'e_types': etypes.long(),
                     'sentence_embeds': sentence_emb, 'fids': fids,
                     'all_ann_info': all_ann_info, 'params': self.params, 'entity_map': entity_map}

        # Role layer
        actual_role_labels_lr, actual_role_labels_rl = self.REL_layer.labels_count(ner_preds)

        return actual_role_labels_lr, actual_role_labels_rl

    def forward(self, batch_input, params):

        ner_preds, rel_preds, ev_preds = self.calculate(batch_input)

        return ner_preds, rel_preds, ev_preds
