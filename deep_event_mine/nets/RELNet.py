import torch
import torch.nn.functional as f
from torch import nn

from deep_event_mine.eval.evalRE import calc_stats

import random
import math


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(math.pi / 2) * (x + 0.044715 * x ** 3)))


class RELModel(nn.Module):

    def __init__(self, params, sizes):
        super(RELModel, self).__init__()

        # num_embeddings=59 because we have 58 entities + the non-entity label
        # we define the type_embedding to have 300 dimensions
        # 58 is the index relating to the ignore label
        self.type_embed = nn.Embedding(num_embeddings=sizes['etype_size'] + 1,
                                       embedding_dim=params['etype_dim'],
                                       padding_idx=sizes['etype_size'])

        # e.g. 768*3 + 300 = 2604
        ent_dim = params['bert_dim'] * 3 + params['etype_dim']  

        # takes as input the [v_t;v_a;c] vectors of length e.g 5976 = (768 + 2604*2)
        # the output dimension for each vector is params['hidden_dim'] e.g 1000
        self.hidden_layer1 = nn.Linear(in_features=2 * ent_dim + params['bert_dim'],
                                       out_features=params['hidden_dim'], bias=False)
        # e.g. from 1000 to 500
        self.hidden_layer2 = nn.Linear(in_features=params['hidden_dim'],
                                       out_features=params['rel_reduced_size'], bias=False)

        # takes as input all the r_i vectors and computes the logits for classification
        # e.g from 500 to 19 that it's the number of all roles + the non-role label
        self.l_class = nn.Linear(in_features=params['rel_reduced_size'],
                                 out_features=sizes['rel_size'])

        self.cel = nn.CrossEntropyLoss()

        self.device = params['device']
        self.params = params
        self.sizes = sizes

    def embedding_layer(self, bert_out, ents_etype_):

        self.b, self.w, _ = bert_out.shape
        self.e = ents_etype_.shape[1]

        # convert the index for the non-entity spans from -1 to 58
        ents_etype_[ents_etype_ == -1] = self.sizes['etype_size']
        type_embeds = self.type_embed(ents_etype_)  # (B, E, params['etype_dim']) e.g. (16, 1694, 300)

        return type_embeds

    def pair_representation(self, ent_embeds, type_embeds):

        pairs4class = torch.cat((ent_embeds, type_embeds), dim=2)

        enttoks_type_embeds = pairs4class.clone()

        return pairs4class, enttoks_type_embeds

    def get_pairs(self, pairs4class, pair_context, pairs_idx, direction, use_gold, use_context):
        indices = pairs_idx

        if direction == 'lr':
            # whether we want to add context c or not
            if use_context:
                return torch.cat(
                    (pairs4class[(indices[0], indices[1])], pairs4class[(indices[0], indices[2])], pair_context),
                    dim=-1)
            else:
                return torch.cat((pairs4class[(indices[0], indices[1])], pairs4class[(indices[0], indices[2])]), dim=-1)
        else:
            if use_context:
                return torch.cat(
                    (pairs4class[(indices[0], indices[2])], pairs4class[(indices[0], indices[1])], pair_context),
                    dim=-1)
            else:
                return torch.cat((pairs4class[(indices[0], indices[2])], pairs4class[(indices[0], indices[1])]), dim=-1)

    def create_target_features(self, batch_input):

        pairs_idx = batch_input["pairs_idx"]
        fids = batch_input["fids"]
        entity_map = batch_input["entity_map"]
        relations = batch_input["all_ann_info"]["relations"]
        entities = batch_input["all_ann_info"]["entities"]
        params = batch_input["params"]
        rel_size = params['voc_sizes']['rel_size']
        rel_dict = {v: k for k, v in params['mappings']['rev_rel_map'].items()}

        actual_role_labels_lr = []
        actual_role_labels_rl = []

        for pair in pairs_idx.T:
            pair = pair.tolist()
            sentence_id = pair[0]
            element1 = entity_map[(sentence_id, pair[1])]
            element2 = entity_map[(sentence_id, pair[2])]
            pmid = fids[sentence_id]
            relations_pmid = relations[pmid]
            entities_pmid = entities[pmid]

            target_lr = rel_size - 1
            target_rl = rel_size - 1
            for relation in relations_pmid.items():
                relation = relation[1]
                left_arg = relation['left_arg']['id']
                left_arg_start_end = (entities_pmid[left_arg]['start'], entities_pmid[left_arg]['end'])

                right_arg = relation['right_arg']['id']
                right_arg_start_end = (entities_pmid[right_arg]['start'], entities_pmid[right_arg]['end'])

                if element1[4] == left_arg_start_end and element2[4] == right_arg_start_end:
                    target_lr = rel_dict['1:{}:2'.format(relation['role'])]
                    target_rl = rel_dict['2:{}:1'.format(relation['role'])]
                    break

            actual_role_labels_lr.append(int(target_lr))
            actual_role_labels_rl.append(int(target_rl))

        actual_role_labels_lr = torch.Tensor(actual_role_labels_lr).type(torch.LongTensor).to(self.device)
        actual_role_labels_rl = torch.Tensor(actual_role_labels_rl).type(torch.LongTensor).to(self.device)

        return actual_role_labels_lr, actual_role_labels_rl

    def balance_dataset(self, pairs_idx_, actual_role_labels_lr, actual_role_labels_rl):

        positive_ents = []
        negative_ents = []
        for idx, ent in enumerate(actual_role_labels_lr):
            if ent != 18:
                positive_ents.append(idx)
            else:
                negative_ents.append(idx)

        sample_size = self.params['rel_balance_mult'] * len(positive_ents)
        if sample_size <= len(negative_ents):
            negative_ents = random.sample(negative_ents, sample_size)
            balanced_ents = positive_ents + negative_ents

            pairs_idx_ = pairs_idx_.T[balanced_ents].T
            actual_role_labels_lr = actual_role_labels_lr[balanced_ents]
            actual_role_labels_rl = actual_role_labels_rl[balanced_ents]

        return pairs_idx_, actual_role_labels_lr, actual_role_labels_rl

    def compute_metrics(self, predictions, actual_labels):

        predictions = [18 if el == -1 else el for el in predictions]
        actual_labels = actual_labels.tolist()

        pos_golds = [rel for rel in actual_labels if rel != 18]
        tp = [i for i, rel in enumerate(actual_labels) if actual_labels[i] == predictions[i] and rel != 18]
        tp_fp = [ent for ent in predictions if ent != 18]

        if len(pos_golds) > 0:
            rel_recall = len(tp) / len(pos_golds)
            print('')
            print(f'number of matching relations: {len(tp)} out of {len(pos_golds)} actual positive  {rel_recall}')

        else:
            print('no positive relations')

        if len(tp_fp) > 0:
            rel_precision = len(tp) / len(tp_fp)
            print(f'number of matching relations: {len(tp)} out of {len(tp_fp)} predicted as positive {rel_precision}')

    def classification(self, pairs4class, pairs_idx_, sent_embeds):

        if self.params['predict']:
            
            pair_context = sent_embeds[pairs_idx_[0]]

            # creates [v_t;v_a;c]  by concatenating the vectors corresponding to each entity in the pair
            # and the context c resulting to a 5976-dimensional vector (768 + 2604*2) for each pair
            # e.g (235, 5976) where 235 is the number of pairs
            l2r_pairs = self.get_pairs(pairs4class, pair_context, pairs_idx_, 'lr', False,
                                           self.params['use_context'])

        # this two operations compute pair representation r_i
        l2r_pairs = gelu(self.hidden_layer1(l2r_pairs))  # (number_of_pairs, params['hidden_dim']) e.g (235, 1000)
        l2r_pairs = gelu(self.hidden_layer2(l2r_pairs))  # (number_of_pairs, params['rel_reduced_size']) e.g (235, 500)

        # computes the logits for role classification
        pairs_preds_l2r = self.l_class(l2r_pairs)  # (number_of_pairs, number_of_roles) e.g (235, 19)

        # compute also other way around where we put the entity before the trigger
        # so we have [v_a;v_t;c] rather than [v_t;v_a;c]
        if self.params['direction'] != 'l2r':
            
            if self.params['predict']:
                pair_context = sent_embeds[pairs_idx_[0]]
                r2l_pairs = self.get_pairs(pairs4class, pair_context, pairs_idx_, 'rl', False,
                                               self.params['use_context'])

            r2l_pairs = gelu(self.hidden_layer1(r2l_pairs))
            r2l_pairs = gelu(self.hidden_layer2(r2l_pairs))

            pairs_preds_r2l = self.l_class(r2l_pairs)

            return pairs_preds_l2r, pairs_preds_r2l, l2r_pairs, r2l_pairs, pairs4class, pairs_idx_
        else:
            return pairs_preds_l2r, pairs4class, pairs_idx_

    def calculate(self, batch_input):
        # computes the s_t and s_a type embeddings
        # (batch_size, max_span_labels*2, params['etype_dim']) e.g. (16, 1694, 300)
        type_embeds = self.embedding_layer(batch_input['embeddings'], batch_input['ent_types'])

        # embeddings for each sentence to give the context representation c
        # (batch_size, params['bert_dim']) e.g (16, 768)
        sent_embeds = batch_input['sentence_embeds']

        # concatenate span embeddings and type embeddings obtaining
        # v_t = [m_t;s_t] for triggers and v_a = [m_a;s_a] for entities
        # pair4class.shape (batch_size, max_span_labels*2, params['bert_dim'] * 3 + params['etype_dim']) e.g (16, 1694, 768*3 + 300)
        pairs4class, enttoks_type_embeds = self.pair_representation(
            ent_embeds=batch_input['ent_embeds'],
            type_embeds=type_embeds)

        # apparently doesn't change anything
        pairs4class = pairs4class.view(self.b, self.e, pairs4class.shape[2])

        # first two tensors contain the logits for [v_t;v_a;c] and [v_a;v_t;c]
        forw_comp_res = self.classification(pairs4class=pairs4class,
                                            pairs_idx_=batch_input['pairs_idx'],
                                            sent_embeds=sent_embeds)

        return forw_comp_res, enttoks_type_embeds

    def forward(self, batch_input, warm_up=False, balance_data=False):
        if len(batch_input['pairs_idx']) > 0:

            if self.params['compute_dem_loss']:
                actual_role_labels_lr, actual_role_labels_rl = self.create_target_features(batch_input)

                if 'rel_balance' in self.params:
                    balance_data = self.params['rel_balance']

                if balance_data:
                   batch_input['pairs_idx'], actual_role_labels_lr, actual_role_labels_rl = self.balance_dataset(batch_input['pairs_idx'],
                                                                                                                 actual_role_labels_lr,
                                                                                                                 actual_role_labels_rl)

            fcomp_res, enttoks_type_embeds = self.calculate(batch_input)

            # if we consider [v_t;v_a;c] and [v_a;v_t;c]
            if self.params['direction'] != 'lr2':

                # - preds_l2r and preds_r2l are the logits for [v_t;v_a;c] and [v_a;v_t;c] respectively
                # - l2r_pairs and r2l_pairs are the embeddings representations of each span pair r_i
                #   in both directions
                # - pair4class concatenate span embeddings and type embeddings obtaining
                #   v_t = [m_t;s_t] for triggers and v_a = [m_a;s_a] for entities
                preds_l2r, preds_r2l, l2r_pairs, r2l_pairs, pair4class, pairs_idx = fcomp_res

                # computes probabilities for [v_t;v_a;c] and [v_a;v_t;c] and retruns 2 tensors of size
                # (number_of_pairs, number_role_labels) e.g. (235, 19)
                preds = (f.softmax(preds_l2r, dim=1).data, f.softmax(preds_r2l, dim=1).data)

                if self.params['compute_dem_loss']:

                    if warm_up:
                        new_preds = actual_role_labels_lr

                    loss_lr = self.cel(preds_l2r, actual_role_labels_lr)
                    loss_rl = self.cel(preds_r2l, actual_role_labels_rl)

                    loss = loss_lr + loss_rl

                else:
                    loss = None

            # if we consider[v_t;v_a;c] only
            else:
                preds_l2r, l2r_pairs, pair4class, pairs_idx, positive_indices = fcomp_res
                preds = f.softmax(preds_l2r, dim=1).data

                if self.params['compute_dem_loss']:

                    if warm_up:
                        new_preds = actual_role_labels_lr

                    loss = self.cel(preds_l2r, actual_role_labels_lr)
                else:
                    loss = None

            # among all the possible cases for l2r and r2l return the correct label
            # tensor containing the role label for each pair
            # (number_of_pairs,) e.g. (235,)
            if self.params['compute_dem_loss']:
                if not warm_up:

                    new_preds = calc_stats(preds, self.params)

                else:
                    if self.params['compute_metrics']:
                        predictions = calc_stats(preds, self.params)
                        self.compute_metrics(predictions, new_preds)

            else:
                new_preds = calc_stats(preds, self.params)

            return {'next': True,
                    'preds': new_preds, 'enttoks_type_embeds': enttoks_type_embeds,
                    'pairs_idx': pairs_idx, 'rel_embeds': l2r_pairs,
                    'pair4class': pair4class,  'rel_loss': loss
                    }

        else:
            return {'next': False}

    def labels_count(self, batch_input):
        actual_role_labels_lr, actual_role_labels_rl = self.create_target_features(batch_input)

        return actual_role_labels_lr, actual_role_labels_rl


