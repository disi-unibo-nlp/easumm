""" Build the EVENT prediction network.

"""
import numpy as np
import collections

import torch
from torch import nn
import torch.nn.functional as F

cpu_device = torch.device("cpu")

# use gelu instead of relu activation function
import math


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(math.pi / 2) * (x + 0.044715 * x ** 3)))


from deep_event_mine.nets.EVGen import EV_Generator


class EVModel(nn.Module):
    """CLASS FOR EVENT LAYERS."""

    def __init__(self, params, sizes):
        super(EVModel, self).__init__()

        # parameters
        self.params = params

        # dimensions
        if params['ner_reduce'] == False:

            # 768*3 given by m_k_l + 300 given by the entity embedding
            ent_dim = params['bert_dim'] * 3 + params['etype_dim']  # no reduce
        else:
            # 500 + 300
            ent_dim = params['ner_reduced_size'] + params['etype_dim']

        rel_dim = params['rel_reduced_size'] + params['rtype_dim'] + ent_dim

        ev_dim = ent_dim + params['role_dim']

        # to create event candidates
        self.ev_struct_generator = EV_Generator(params)

        # relation type embeddings
        self.rtype_layer = nn.Embedding(num_embeddings=sizes['rel_size'] + 1, embedding_dim=params['rtype_dim'])

        # IN argument embeddings: argument IN structure
        self.in_arg_layer = nn.Linear(in_features=rel_dim, out_features=params['role_dim'], bias=False)

        # OUT argument embeddings: argument NOT IN structure
        self.out_arg_layer = nn.Linear(in_features=rel_dim, out_features=params['role_dim'], bias=False)

        # for event classification
        self.hidden_layer1 = nn.Linear(in_features=ev_dim, out_features=params['hidden_dim'])
        self.hidden_layer2 = nn.Linear(in_features=params['hidden_dim'], out_features=params['ev_reduced_size'])
        self.l_class = nn.Linear(in_features=params['ev_reduced_size'], out_features=1)

        # reduce event embeds to replace entity
        self.ev2ent_reduce = nn.Linear(in_features=params['ev_reduced_size'], out_features=ent_dim)

        # predict modality
        self.modality_layer = nn.Linear(in_features=params['ev_reduced_size'], out_features=sizes['ev_size'])

        # others
        self.device = params['device']

        self.bce = nn.BCELoss()
        self.cel = nn.CrossEntropyLoss()

    def get_rel_input(self, rel_preds):
        """Read relation input."""

        l2r = rel_preds['pairs_idx']
        rpreds_ = rel_preds['preds'].data

        # mapping relation type for 'OTHER' type to -1
        rpred_types = self.params['mappings']['rel2rtype_map'][rpreds_]

        # extract only relation type != 'OTHER' (valid relations)
        # from all the original pairs we keep only the actual ones
        # e.g. from 235 to 44
        rpred_ids = (rpreds_ != self.params['voc_sizes']['rel_size'] - 1).nonzero(as_tuple=False).transpose(0, 1)[0]
        rpred_ids = rpred_ids.to(cpu_device)  # list: contain indices of the valid relations

        return l2r, rpred_types, rpred_ids

    def rtype_embedding_layer(self, rtype_):
        """Relation type embeddings."""

        # rtype_ contains the role id for each pair e.g. [12, -1, 16, -1, -1,....]

        # replace the -1 relation type by SPECIAL TYPE (rel-size) e.g 19 so that it is NO RELATION TYPE
        if np.ndim(rtype_) > 0:
            # replace -1 with 19 for a relation equal to 'Other'
            rtype_[rtype_ == -1] = self.params['voc_sizes']['rel_size']

            # relation type embedding (number_of_pairs, params['rtype_dim']) e.g (228, 150)
            # we have a different embedding for each relation type
            rtype_embeds = self.rtype_layer(torch.tensor(rtype_, dtype=torch.long, device=self.device))
            has_no_rel = False
            for xx, rtypeid in enumerate(rtype_):
                if rtypeid == self.params['voc_sizes']['rel_size']:
                    # get an index for NO RELATION TYPE, using later for event with no-argument
                    no_rel_type_embed = rtype_embeds[xx]

                    has_no_rel = True
                    break
            if not has_no_rel:
                no_rel_type_embed = rtype_embeds[0]

        else:
            rtype_embeds = torch.zeros(self.params['rtype_dim'], dtype=torch.float32, device=self.device)
            no_rel_type_embed = torch.zeros(self.params['rtype_dim'], dtype=torch.float32, device=self.device)

        return rtype_embeds, no_rel_type_embed

    def get_arg_embeds(self, ent_embeds, rel_embeds, rtype_embeds, ev_arg_ids4nn):
        """Argument embeddings for each trigger.
            - Each trigger has a two-element tuple of
                1. trigger embedding
                2. a list of argument embedding: (relation emb, relation type emb, argument entity emb)
        """

        arg_embed_triggers = collections.OrderedDict()

        # loop over all triggers
        for trid, arg_data in ev_arg_ids4nn.items():
            # trid e.g (0, 8) trigger index
            # argdata e.g [0, 2], [(0, 226), (0, 358)]
            # first list contains pair_id
            # second list contains index for a2 (argument)

            # v_t = [m_t;s_t]
            tr_embeds = ent_embeds[trid]

            # no-argument
            if len(arg_data) == 1:
                arg_embed_triggers[trid] = [tr_embeds]

            # has argument:
            else:
                rids = arg_data[0]
                a2ids = arg_data[1]
                a2ids_ = np.vstack(a2ids).transpose()

                # r_j = GELU(W_r[v_t;v_a;c] + b_r) pair representation
                # (number_of_arguments, embedding_dim) e.g. (2, 500)
                r_embeds = rel_embeds[rids]

                # argument embedding v_a = [m_a;s_a]
                # (number_of_arguments, embedding_dim) e.g (2, 2604
                a2_embeds = ent_embeds[(a2ids_[0], a2ids_[1])]

                # u_j embeddings of role type
                rt_embeds = rtype_embeds[rids]

                # [r_j;u_j;v_a_j]
                args_embeds = torch.cat([r_embeds, rt_embeds, a2_embeds],
                                        dim=-1)  # [number of arguments, rdim+rtypedim+edim]

                # store in a map to use later
                arg_embed_triggers[trid] = [tr_embeds, args_embeds]

        return arg_embed_triggers

    def event_representation(self, arg_embed_triggers, ev_cand_ids4nn, no_rel_type_embed):
        """Create event representation."""

        # get indices
        trids_ = ev_cand_ids4nn['trids_']
        io_ids_ = ev_cand_ids4nn['io_ids_']
        ev_structs_ = ev_cand_ids4nn['ev_structs_']

        # store event embeds in a list, return an array later
        ev_embeds_ = []

        # create embedding for each candidate indices
        # loop over all candidate events
        for xx, trid in enumerate(trids_):

            # corresponding event trigger embed v_t = [m_t;s_t] e.g. (2604,)
            tr_embed = arg_embed_triggers[trid][0]

            # store reduced argument embeds in a list
            args_embeds_list = []

            # get event structure e.g. [Counter({(12, 42): 1}), [(12, 42)]]
            ev_struct = ev_structs_[xx]

            # no-argument
            if len(ev_struct[1]) == 0:

                # since there is no argument, rel_embed is set as zeros
                no_rel_emb = torch.zeros((self.params['rel_reduced_size']), dtype=no_rel_type_embed.dtype,
                                         device=self.device)

                # argument emb is itself: trigger embed
                # then concatenate e.g. (3254,) [r_j;u_j;v_a_j]
                arg_embed = torch.cat([no_rel_emb, no_rel_type_embed, tr_embed])

                # put to IN ARGUMENT layer
                # W_p[r_j;u_j;v_a_j]
                reduced_arg_embed = self.in_arg_layer(arg_embed)
                args_embeds_list.append(reduced_arg_embed)

                # check whether this trigger has other arguments, then set as OUT
                if len(arg_embed_triggers[trid]) > 1:

                    # argument embed
                    args_embeds = arg_embed_triggers[trid][1]

                    # calculate argument embedding
                    for xx, arg_embed in enumerate(args_embeds):
                        # OUT argument via OUT-ARG LAYER
                        reduced_arg_embed = self.out_arg_layer(arg_embed)

                        # store
                        args_embeds_list.append(reduced_arg_embed)

            # has argument
            else:

                # [r_j;u_j;v_a_j] argument embedding e.g. (2, 2604 + 500 + 150) 2 indicates the number of arguments
                # we can have multiple ones if the the trigger accepts multiple arguments
                args_embeds = arg_embed_triggers[trid][1]

                # check IN/OUT
                io_ids = io_ids_[xx]

                # calculate argument embedding
                for ioid, arg_embed in enumerate(args_embeds):

                    # for arguments that are in the event structure
                    # W_p[r_j;u_j;v_a_j]
                    if ioid in io_ids:
                        reduced_arg_embed = self.in_arg_layer(arg_embed)

                    # for arguments that aren't in the event structure
                    # for example in the case of event with songle argument representation
                    # the other argument is not the event structure
                    # W_n[r_j;u_j;v_a_j]
                    else:
                        reduced_arg_embed = self.out_arg_layer(arg_embed)

                    # store
                    args_embeds_list.append(reduced_arg_embed)

            # calculate argument embed: by sum up all arguments or average, etc
            # we sum alla argument embeddings that are and aren't in the event structure
            args_embed = torch.sum(torch.stack(args_embeds_list, dim=0), dim=0)

            # event embed: concatenate trigger embed and argument embed
            # [v_t;sum_j_in_e(W_n[r_j;u_j;v_a_j]) + sum_j_notin_e(W_n[r_j;u_j;v_a_j])] e.g. 3604 = 2604 + 1000
            ev_embeds_.append(torch.cat([tr_embed, args_embed], dim=-1))

        # return tensor [number of event, dim] e.g. (60, 3604)
        ev_embeds = torch.stack(ev_embeds_, dim=0)

        # dropout
        if self.training:
            if self.params['dropout'] > 0:
                ev_embeds = F.dropout(ev_embeds, p=self.params['dropout'])

        return ev_embeds

    def get_nest_arg_embeds(self, ent_embeds, rel_embeds, rtype_embeds, ev_arg_ids4nn, all_ev_embeds):
        """Argument embeddings for each trigger.
            - Each trigger has a two-element tuple of
                1. trigger embedding
                2. a list of argument embedding: (relation emb, relation type emb, argument entity emb)
        """

        # store a list of embeds for each trigger: trigger embeds, relation embeds, relation type embeds
        arg_embed_triggers = collections.OrderedDict()

        for trid, arg_data in ev_arg_ids4nn.items():
            tr_embeds = ent_embeds[trid]

            rids = arg_data[0]
            a2ids = arg_data[1]
            a2ids_ = np.vstack(a2ids).transpose()
            # r_i embeddings that give us a representation for each pair
            # (number_of_pairs, params['rel_reduced_size']) e.g. (228, 500)
            r_embeds = rel_embeds[rids]
            # v_t = [m_t;s_t] given that the argument is a trigger
            a2_embeds = ent_embeds[(a2ids_[0], a2ids_[1])]
            # role type embeds
            rt_embeds = rtype_embeds[rids]

            # replace event embeds for event arguments
            ev_argids_ = arg_data[2]

            # store event argument embeds by (argument id, event id)
            ev_arg_embeds_list = [[] for argid in range(len(rids))]

            for argid, ev_argids in enumerate(ev_argids_):

                # event argument
                if len(ev_argids) > 0:

                    # store event argument embeds with key is event id
                    ev_arg_embeds = collections.OrderedDict()

                    for pid in ev_argids:
                        # pid: (level, positive_event_id)

                        # store event argument embed by argument id and event id
                        ev_arg_emb = all_ev_embeds[pid[0]][pid[1]]
                        ev_rel_emb = r_embeds[argid]
                        ev_rtype_emb = rt_embeds[argid]

                        # concatenate with relation and relation type embeds
                        # [r_j;u_j;v_a_j] where v_a_j=e_a_j
                        ev_arg_embeds[pid] = torch.cat([ev_rel_emb, ev_rtype_emb, ev_arg_emb], dim=-1)

                    # add to the list
                    ev_arg_embeds_list[argid] = ev_arg_embeds

            # concatenate for argument embeddings: [rel_embed, rel_type_embed, entity_embed]
            # [r_j;u_j;v_a_j] where v_a_j is left unchanged
            args_embeds = torch.cat([r_embeds, rt_embeds, a2_embeds],
                                    dim=-1)  # [number of arguments, rdim+rtypedim+edim]

            # store in a map to use later
            arg_embed_triggers[trid] = [tr_embeds, args_embeds, ev_arg_embeds_list]

        return arg_embed_triggers

    def event_nest_representation(self, arg_embed_triggers, ev_cand_ids4nn, no_rel_type_embed):
        """Create event representation."""

        # get indices
        trids_ = ev_cand_ids4nn['trids_']
        io_ids_ = ev_cand_ids4nn['io_ids_']
        ev_structs_ = ev_cand_ids4nn['ev_structs_']
        pos_ev_ids_ = ev_cand_ids4nn['pos_ev_ids_']

        # store event embeds in a list, return an array later
        ev_embeds_ = []

        # create embedding for each candidate indices
        for xx, trid in enumerate(trids_):

            # trigger embed
            tr_embed = arg_embed_triggers[trid][0]

            # store reduced argument embeds in a list
            args_embeds_list = []

            # get ev_struct
            ev_struct = ev_structs_[xx]

            # no-argument
            if len(ev_struct[1]) == 0:

                # since there is no argument, rel_embed is set as zeros
                no_rel_emb = torch.zeros((self.params['rel_reduced_size']), dtype=no_rel_type_embed.dtype,
                                         device=self.device)

                # argument emb is itself: trigger embed
                # then concatenate
                arg_embed = torch.cat([no_rel_emb, no_rel_type_embed, tr_embed])

                # put to IN ARGUMENT layer
                reduced_arg_embed = self.in_arg_layer(arg_embed)
                args_embeds_list.append(reduced_arg_embed)

                # check whether this trigger has other arguments, then set as OUT
                if len(arg_embed_triggers[trid]) > 1:

                    # argument embed
                    args_embeds = arg_embed_triggers[trid][1]

                    # calculate argument embedding
                    for xx, arg_embed in enumerate(args_embeds):
                        # OUT argument via OUT-ARG LAYER
                        reduced_arg_embed = self.out_arg_layer(arg_embed)

                        # store
                        args_embeds_list.append(reduced_arg_embed)

            # has argument
            else:

                # argument embed v_a_j
                args_embeds = arg_embed_triggers[trid][1]

                # event argument embeds e_a_j
                ev_args_embeds = arg_embed_triggers[trid][2]

                # check IN/OUT
                io_ids = io_ids_[xx]

                # positive event ids
                pos_ids = pos_ev_ids_[xx]

                # calculate argument embedding
                for ioid, arg_embed in enumerate(args_embeds):

                    # IN argument via IN-ARG LAYER
                    if ioid in io_ids:

                        for xx2, inid in enumerate(io_ids):
                            if inid == ioid:
                                pid = pos_ids[xx2]

                                # entity argument
                                if pid == (-1, -1):
                                    reduced_arg_embed = self.in_arg_layer(arg_embed)
                                    args_embeds_list.append(reduced_arg_embed)

                                # event argument
                                else:
                                    ev_arg_embed = ev_args_embeds[ioid][pid]
                                    reduced_arg_embed = self.in_arg_layer(ev_arg_embed)
                                    args_embeds_list.append(reduced_arg_embed)

                    # OUT argument via OUT-ARG LAYER
                    else:

                        # entity argument
                        if len(ev_args_embeds[ioid]) == 0:
                            reduced_arg_embed = self.out_arg_layer(arg_embed)
                            args_embeds_list.append(reduced_arg_embed)

                        # event arguments: run with all event arguments for this trigger
                        else:
                            for pid in ev_args_embeds[ioid]:
                                ev_arg_embed = ev_args_embeds[ioid][pid]
                                reduced_arg_embed = self.out_arg_layer(ev_arg_embed)
                                args_embeds_list.append(reduced_arg_embed)

                    # store
                    # args_embeds_list.append(reduced_arg_embed)

            # calculate argument embed: by sum up all arguments or average, etc
            args_embed = torch.sum(torch.stack(args_embeds_list, dim=0), dim=0)

            # event embed: concatenate trigger embed and argument embed
            ev_embeds_.append(torch.cat([tr_embed, args_embed], dim=-1))

        # return tensor [number of event, dim]
        ev_embeds = torch.stack(ev_embeds_, dim=0)

        # dropout
        if self.training:
            if self.params['dropout'] > 0:
                ev_embeds = F.dropout(ev_embeds, p=self.params['dropout'])

        return ev_embeds

    def predict(self, event_embeds):
        """Prediction."""

        threshold = self.params['ev_threshold']

        # e.g  from (60, 3604) to (60, 1000) where 60 is the number of candidate events
        event4class = gelu(self.hidden_layer1(event_embeds))
        # e.g from (60, 1000) to (60, 500)
        event4class = gelu(self.hidden_layer2(event4class))
        # from (60, 500) to (60, 1)
        prediction = self.l_class(event4class)

        # modality
        ev_lbls = prediction.clone()

        # apply sigmoid function to all logits
        norm_preds = torch.sigmoid(ev_lbls)
        nnp = norm_preds.view(norm_preds.size()[0])
        norm_preds = norm_preds.detach().cpu().numpy()

        # array of size (n_candidate_events, 1) e.g (60,1) with just 0.5
        threshols = threshold * np.ones(ev_lbls.shape, dtype=np.float32)
        # predictions greater than thresholds True or False
        pred_mask = np.asarray(np.greater(norm_preds.data, threshols), dtype=np.int32)
        # indices from event candidates classified as event candidates
        positive_idx = np.where(pred_mask.ravel() != 0)

        # 500-dimensional embedding for positive event_candidates
        positive_ev_embs = event4class[positive_idx]

        prediction = prediction.flatten()

        # return prediction, modality_pred, positive_idx, positive_ev # revise
        return event4class, prediction, positive_idx, positive_ev_embs, nnp, pred_mask.tolist()

    def predict_modality(self, positive_ev_embs, positive_ev_idx, mod_labels_):
        """Predict modality, return modality predictions."""

        # get labels
        mod_labels = np.vstack(mod_labels_).ravel()
        positive_labels = mod_labels.copy()
        positive_labels[positive_labels > 0] = 1

        possitive_lbl = torch.tensor((mod_labels[positive_ev_idx] - 1), dtype=torch.long,
                                     device=self.device)

        # prediction
        if possitive_lbl[possitive_lbl >= 0].shape[0] > 0:

            # prediction e.g. (28, 3)
            modality_preds_ = self.modality_layer(positive_ev_embs)
            modality_preds = modality_preds_[possitive_lbl >= 0]
            modality_pred = modality_preds_.detach().cpu().numpy()

            # compute probabilities for each positive event candidate
            modality_pred = F.softmax(torch.tensor(modality_pred), dim=-1).data

            # get modality index that maximises the probability
            mod_preds = modality_pred.argmax(dim=-1)

        else:
            mod_preds = []
            modality_preds_ = None

        return mod_preds, modality_preds_

    def create_output(self, all_ev_preds):
        """Create output for writing events."""

        all_ev_output = []

        for level, ev_preds in enumerate(all_ev_preds):
            # store output in a list
            ev_output = []

            # get indices
            ev_cands_ids = ev_preds[0]
            ev_args_ids = ev_preds[1]
            positive_idx = ev_preds[2]
            modality_preds = ev_preds[3]

            # input indices
            trids_ = ev_cands_ids['trids_']
            io_ids_ = ev_cands_ids['io_ids_']
            ev_structs_ = ev_cands_ids['ev_structs_']

            # positive ev ids
            if level > 0:
                pos_ev_ids_ = ev_cands_ids['pos_ev_ids_']
            else:
                pos_ev_ids_ = []

            for xx1, pid in enumerate(positive_idx):

                # trigger id
                trid = trids_[pid]

                # structure
                ev_struct = ev_structs_[pid]

                # argument: relation id and entity id
                arg_data = ev_args_ids[trid]

                # check argument
                if len(arg_data) > 1:

                    # store argument list
                    # flat
                    if level == 0:
                        a2ids = [arg_data[1][inid] for inid in io_ids_[pid]]

                    # nested
                    else:
                        a2ids = []
                        for (inid, posid) in zip(io_ids_[pid], pos_ev_ids_[pid]):

                            # it is an entity argument
                            if posid == (-1, -1):
                                a2ids.append(arg_data[1][inid][0])

                            # or index of event
                            else:
                                a2ids.append((-1, -1, posid))  # add -1 to check later

                # no-argument: return empty list
                else:
                    a2ids = []

                # check modality
                if len(modality_preds) > 0:
                    mod_pred = modality_preds[xx1].item()
                else:
                    mod_pred = -1

                # store output
                ev_output.append([trid, ev_struct, a2ids, mod_pred])

            # store the output of this level
            all_ev_output.append(ev_output)

        return all_ev_output

    def create_target_features(self, ev_cand_ids4nn, ev_arg_ids4nn, all_ann_info, entity_map, fids, mappings):

        trids = ev_cand_ids4nn['trids_']
        io_ids = ev_cand_ids4nn['io_ids_']
        ev_structs = ev_cand_ids4nn['ev_structs_']
        modality_dict = {v: k for k, v in mappings['rev_modality_map'].items()}
        actual_event_labels = []
        actual_modality_labels = []
        event_id_list = []

        for xx, trid in enumerate(trids):
            tr_start_end = entity_map[trid][4]
            sentence_id = trid[0]
            pmid = fids[sentence_id]
            io_id = io_ids[xx]
            roles = ev_structs[xx][1]
            roles_text = [mappings['rev_rtype_map'][role[0]] for role in roles]

            true_events_pmid = all_ann_info['events'][pmid]
            true_entities_pmid = all_ann_info['entities'][pmid]
            true_modalities_pmid = all_ann_info['modalities'][pmid]

            label = 0
            modality_id = 0
            event_id = 'non_event'

            # events without arguments
            if len(roles) == 0:
                for event in true_events_pmid.items():
                    event = event[1]
                    trigger_id = event['trigger_id']
                    true_tr_start = true_entities_pmid[trigger_id]['start']
                    true_tr_end = true_entities_pmid[trigger_id]['end']
                    true_start_end = (true_tr_start, true_tr_end)
                    if len(event['args']) == 0 and tr_start_end == true_start_end:

                        event_id = event['id']
                        for modality in true_modalities_pmid.items():
                            modality = modality[1]
                            if event_id in modality['reference_ids']:
                                modality_id = modality_dict[modality['type']] - 1

                        # print(event_id, event['args'])
                        label = 1

                actual_modality_labels.append(modality_id)
                actual_event_labels.append(label)
                event_id_list.append(event_id)

                continue

            args = [ev_arg_ids4nn[trid][1][id] for id in io_id]

            args_start_end_roles = []
            for arg_id, arg in enumerate(args):
                arg = arg if type(arg) is tuple else arg[0]
                start_end = entity_map[arg][4]
                args_start_end_roles.append((tr_start_end, start_end, roles_text[arg_id]))

            for event in true_events_pmid.items():
                event = event[1]
                if len(args) == len(event['args']):
                    trigger_id = event['trigger_id']
                    true_tr_start = true_entities_pmid[trigger_id]['start']
                    true_tr_end = true_entities_pmid[trigger_id]['end']
                    true_args_start_end_roles = []
                    for true_arg in event['args']:
                        true_arg_id = true_arg['id']

                        if true_arg_id[0] == 'E':
                            continue
                        arg_start = true_entities_pmid[true_arg_id]['start']
                        arg_end = true_entities_pmid[true_arg_id]['end']
                        true_args_start_end_roles.append(
                            ((true_tr_start, true_tr_end), (arg_start, arg_end), true_arg['role']))

                    check = all(item in true_args_start_end_roles for item in args_start_end_roles)
                    if check:
                        event_id = event['id']
                        for modality in true_modalities_pmid.items():
                            modality = modality[1]
                            if event_id in modality['reference_ids']:
                                modality_id = modality_dict[modality['type']] - 1

                        # print(event_id, event['args'])
                        label = 1
                        break

            actual_modality_labels.append(modality_id)
            actual_event_labels.append(label)
            event_id_list.append(event_id)

        ev_cand_ids4nn['ev_labels_'] = actual_event_labels
        ev_cand_ids4nn['mod_labels'] = actual_modality_labels
        ev_cand_ids4nn['event_ids'] = event_id_list

    def create_nested_target_features(self, ev_nest_cand_ids4nn,
                                      ev_nest_arg_ids4nn,
                                      all_ann_info, entity_map, fids,
                                      mappings, all_preds_output):

        trids = ev_nest_cand_ids4nn['trids_']
        io_ids = ev_nest_cand_ids4nn['io_ids_']
        ev_structs = ev_nest_cand_ids4nn['ev_structs_']
        modality_dict = {v: k for k, v in mappings['rev_modality_map'].items()}
        actual_event_labels = []
        actual_modality_labels = []
        event_id_list = []

        for xx, trid in enumerate(trids):
            tr_start_end = entity_map[trid][4]
            sentence_id = trid[0]
            pmid = fids[sentence_id]
            io_id = io_ids[xx]
            roles = ev_structs[xx][1]
            roles_text = [mappings['rev_rtype_map'][role[0]] for role in roles]

            true_events_pmid = all_ann_info['events'][pmid]
            true_entities_pmid = all_ann_info['entities'][pmid]
            true_modalities_pmid = all_ann_info['modalities'][pmid]

            label = 0
            modality_id = 0
            event_id = 'non_event'

            arg_events = ev_nest_cand_ids4nn['pos_ev_ids_'][xx]
            arg_entity = [ev_nest_arg_ids4nn[trid][1][id] for id in io_id]

            args_start_end_roles = []
            for arg_id, arg in enumerate(arg_events):

                if arg[0] == -1:
                    arg = arg_entity[arg_id]
                    arg = arg if type(arg) is tuple else arg[0]
                    start_end = entity_map[arg][4]
                    args_start_end_roles.append((tr_start_end, start_end, roles_text[arg_id]))

                else:
                    arg = arg if type(arg) is tuple else arg[0]
                    nested_level = arg[0]
                    event_id_int = arg[1]
                    event = all_preds_output[nested_level][0]['only_pos_events'][event_id_int]
                    args_start_end_roles.append((tr_start_end, event, roles_text[arg_id]))

            for event in true_events_pmid.items():
                event = event[1]
                if len(arg_events) == len(event['args']):
                    trigger_id = event['trigger_id']
                    true_tr_start = true_entities_pmid[trigger_id]['start']
                    true_tr_end = true_entities_pmid[trigger_id]['end']
                    true_args_start_end_roles = []
                    for true_arg in event['args']:
                        true_arg_id = true_arg['id']

                        if true_arg_id[0] == 'E':
                            true_args_start_end_roles.append(
                                ((true_tr_start, true_tr_end), true_arg_id, true_arg['role']))

                        else:
                            arg_start = true_entities_pmid[true_arg_id]['start']
                            arg_end = true_entities_pmid[true_arg_id]['end']
                            true_args_start_end_roles.append(
                                ((true_tr_start, true_tr_end), (arg_start, arg_end), true_arg['role']))

                    check = all(item in true_args_start_end_roles for item in args_start_end_roles)
                    if check:
                        event_id = event['id']
                        for modality in true_modalities_pmid.items():
                            modality = modality[1]
                            if event_id in modality['reference_ids']:
                                modality_id = modality_dict[modality['type']] - 1

                        # print(event_id, event['args'])
                        label = 1
                        break
            actual_modality_labels.append(modality_id)
            actual_event_labels.append(label)
            event_id_list.append(event_id)

        ev_nest_cand_ids4nn['ev_labels_'] = actual_event_labels
        ev_nest_cand_ids4nn['mod_labels'] = actual_modality_labels
        ev_nest_cand_ids4nn['event_ids'] = event_id_list

    def compute_metrics(self, actual_event_labels, pred_events_list, actual_modalities_list, pred_mod_list,
                        enable_modality):
        pos_actual_list = [ev for ev in actual_event_labels.tolist() if ev > 0]

        tp = [i for i, ev in enumerate(actual_event_labels.tolist()) if
              actual_event_labels[i] == pred_events_list[i] and ev > 0]

        print(len([ev for ev in pred_events_list if ev > 0]))
        if len(pos_actual_list) > 0:
            ev_recall = len(tp) / len(pos_actual_list)
            print('')
            print(f'number of matching positive events: {len(tp)} out of {len(pos_actual_list)}  {ev_recall}')

        if enable_modality:
            actual_modalities_labels = torch.cat(actual_modalities_list)
            match = [i for i, ev in enumerate(actual_modalities_labels.tolist()) if
                     actual_modalities_labels[i] == pred_mod_list[i]]

            if len(actual_modalities_labels) > 0:
                mod_accuracy = len(match) / len(actual_modalities_labels)
                print(
                    f'number of matching modalities: {len(match)} out of {len(actual_modalities_labels)}  {mod_accuracy}')

    def calculate(self, ent_embeds, rel_embeds, rpred_types, ev_ids4nn, all_ann_info, entity_map, fids):
        """
        Create embeddings, prediction.

        :param ent_embeds: [batch x a1id x embeds]
        :param rel_embeds: [rids x embeds]
        :param rpred_types: [rids] # predicted relation types
        :param ev_ids4nn: generated event canddiate indices
            + ev_cand_ids4nn: event candidates indices
                + trids_: list of trigger ids corresponding to the list of events
                + ev_labels_: list of corresponding labels
                + mod_labels_: modality labels
                io_ids_: in/out indices
            + ev_arg_ids4nn: event argument indices (a map of argument indices for each trigger)
                + list of rids
                + list of argument ids

        :return: prediction
        """

        # store output
        all_preds_output = []
        actual_modalities_list = []
        actual_events_list = []
        event_preds_list = []
        modality_preds_list = []
        pred_events_list = []
        pred_mod_list = []

        enable_nested_ev = True
        enable_modality = True

        # store all predictions for flat and nested, maximum as 3 nested levels
        # positive ids: the current predicted indices; tr_ids: trigger indices of the candidate list
        all_positive_ids = -1 * np.ones((self.params['max_ev_level'] + 1), dtype=np.object)
        all_positive_tr_ids = -1 * np.ones((self.params['max_ev_level'] + 1), dtype=np.object)

        # store predicted events embeds
        all_positive_ev_embs = []

        # for flat events
        # 1-candidate input
        ev_flat_cand_ids4nn = ev_ids4nn['ev_cand_ids4nn']
        ev_flat_arg_ids4nn = ev_ids4nn['ev_arg_ids4nn']

        # 2-relation type embeddings
        # rpred_types contains the role id for each pair e.g. [12, -1, 16, -1, -1,....]
        # rtype_embeds relation type embedding (number_of_pairs, params['rtype_dim']) e.g (228, 150)
        # we have a different embedding for each relation type
        # we are talking about the u_j embeddings
        rtype_embeds, no_rel_type_embed = self.rtype_embedding_layer(rpred_types)

        # 3-argument embeddings for each trigger
        # INPUT:
        # - ent_embeds: v_t = [m_t;s_t] and v_a = [m_a;s_a] e.g. (16, 1694, 2604)
        # - rel_embeds: r_j = GELU(W_r[v_t;v_a;c] + b_r) e.g. (228, 500)
        # - rtype_embeds: u_j e.g (228, 150)
        #
        # Output:
        # - dictionary for each trigger where the key is represented by the trigger id e.g (0, 8)
        #   for each trigger we have:
        #   - v_t = [m_t;s_t] embedding of the trigger
        #   - [r_j;u_j;v_a_j] argument embedding (number_of_arguments, embedding_dim) e.g (2, 2604 + 500 + 150)
        arg_embed_triggers = self.get_arg_embeds(ent_embeds, rel_embeds, rtype_embeds, ev_flat_arg_ids4nn)

        # 4-create event representation
        # we get [v_t;sum_j_in_e(W_p[r_j;u_j;v_a_j]) + sum_j_notin_e(W_n[r_j;u_j;v_a_j])] e.g. 3604 = 2604 + 1000
        # ev_embeds (n_candidate_events, trigger_embedding_dim + arg_reduced_size) e.g. (60, 3604)
        ev_embeds = self.event_representation(arg_embed_triggers, ev_flat_cand_ids4nn, no_rel_type_embed)

        if self.params['compute_dem_loss']:
            self.create_target_features(ev_flat_cand_ids4nn,
                                        ev_flat_arg_ids4nn, all_ann_info,
                                        entity_map, fids, self.params['mappings'])

            actual_events_list.append(torch.Tensor(ev_flat_cand_ids4nn['ev_labels_']))

        # 5-prediction
        # - event4class e_i candidate event embeddings e.g (60, 500)
        # - embeddings just for the positive event candidates e.g. (28, 500)
        # - prediction containing un-normalised logits
        # - loss_flat_events containing the loss if train == True otherwise None
        # -  positive_idx: positive candidate events indexes
        event4class, prediction, positive_idx, positive_ev_embs, flat_event_preds, flat_pred_mask = self.predict(
            ev_embeds)

        event_preds_list.append(flat_event_preds)

        if self.params['compute_dem_loss']:
            pred_event_labels = [item for sublist in flat_pred_mask for item in sublist]
            pred_events_list.extend(pred_event_labels)
            # save true event_id e.g. E1 or -1 if it's labeled by the predictor as non-event
            ev_flat_cand_ids4nn['truth_ids_'] = [ev_flat_cand_ids4nn['event_ids'][i] if label == 1 else -1 for i, label
                                                 in enumerate(pred_event_labels)]
            # save only the candidate events labeled as positive
            ev_flat_cand_ids4nn['only_pos_events'] = [ev_flat_cand_ids4nn['event_ids'][i] for i in positive_idx[0]]

        empty_pred = True

        # for modality
        if enable_modality and len(positive_idx[0]) > 0:
            if self.params['compute_dem_loss']:
                actual_modality_labels = [ev_flat_cand_ids4nn['mod_labels'][i] for i in positive_idx[0]]
                actual_modality_labels = torch.Tensor(actual_modality_labels).type(torch.LongTensor)
                actual_modalities_list.append(actual_modality_labels)

            # modality indixes for each positive event candidate e.g. (28,)
            mod_preds, flat_modality_preds = self.predict_modality(positive_ev_embs, positive_idx,
                                                                   ev_flat_cand_ids4nn['mod_labels_'])
            modality_preds_list.append(flat_modality_preds)
            pred_mod_list.extend(mod_preds)
        else:
            mod_preds = []

        # init current nested level
        current_nested_level = 0
        current_tr_ids = ev_flat_cand_ids4nn['trids_']
        current_truth_ids = ev_flat_cand_ids4nn['truth_ids_']

        # store positive ids
        current_positive_ids = positive_idx[0]
        all_positive_ids[current_nested_level] = current_positive_ids
        # candidate nested event triggers
        ev_nest_cand_triggers = ev_ids4nn['ev_nest_cand_triggers']

        # for output
        # at the moment we just have the output related to the flat events
        all_preds_output.append([ev_flat_cand_ids4nn, ev_flat_arg_ids4nn, current_positive_ids, mod_preds])

        # loop until stop nested event prediction or no more events predicted, or in limited nested levels
        while enable_nested_ev and len(current_positive_ids) > 0 and current_nested_level < self.params['max_ev_level']:

            # update trigger indices and predicted positive indices
            # positive trigger indices
            current_positive_tr_ids = [current_tr_ids[pos_id] for pos_id in current_positive_ids]
            current_positive_truth_ids = [current_truth_ids[pos_id] for pos_id in current_positive_ids]
            all_positive_tr_ids[current_nested_level] = current_positive_tr_ids

            # reduce event embeds to replace entity, from the paper:
            # "When the event candidate has a trigger with a
            # fixed event as its argument, we replace vaj by the event candidate
            # representation eaj in Equation (5) after adjusting the dimensions via a hidden layer, e.g. from 500 to 2604"
            reduced_ev_emb = self.ev2ent_reduce(positive_ev_embs)
            all_positive_ev_embs.append(reduced_ev_emb)

            # generate nested candidate indices
            ev_nest_ids4nn = self.ev_struct_generator._generate_nested_candidates(current_nested_level,
                                                                                  ev_nest_cand_triggers,
                                                                                  current_positive_tr_ids,
                                                                                  current_positive_truth_ids)

            # get candidate indices, updated by the previous level output
            ev_nest_cand_ids4nn = ev_nest_ids4nn['ev_nest_cand_ids']
            ev_nest_arg_ids4nn = ev_nest_ids4nn['ev_nest_arg_ids']
            ev_nest_cand_triggers = ev_nest_ids4nn['ev_nest_cand_triggers']
            current_tr_ids = ev_nest_cand_ids4nn['trids_']
            current_truth_ids = ev_nest_cand_ids4nn['truth_ids_']

            empty_pred = False

            # check non-empty
            if len(ev_nest_cand_ids4nn['trids_']) > 0:

                # 3-argument embeddings for each trigger
                # dictionary for each trigger where the key is represented by the trigger id e.g (0, 8)
                #  for each trigger we have:
                #   - v_t = [m_t;s_t] embedding of the trigger
                #   - [r_j;u_j;e_a_j] argument embedding if  the event candidate has a trigger with a fixed event as its argument
                #   - [r_j;u_j;v_a_j] argument embedding (number_of_arguments, embedding_dim) e.g (2, 2604 + 500 + 150)
                arg_embed_triggers = self.get_nest_arg_embeds(ent_embeds, rel_embeds, rtype_embeds, ev_nest_arg_ids4nn,
                                                              all_positive_ev_embs)

                # event representation e.g. (6, 3604)
                # [v_t;sum_j_in_e(W_p[r_j;u_j;e_a_j]) + sum_j_notin_e(W_n[r_j;u_j;e_a_j])] e.g. 3604 = 2604 + 1000
                ev_embeds = self.event_nest_representation(arg_embed_triggers, ev_nest_cand_ids4nn, no_rel_type_embed)

                # create target features for event, non-event and modality
                if self.params['compute_dem_loss']:
                    self.create_nested_target_features(ev_nest_cand_ids4nn,
                                                       ev_nest_arg_ids4nn,
                                                       all_ann_info,
                                                       entity_map, fids,
                                                       self.params['mappings'],
                                                       all_preds_output)

                    actual_events_list.append(torch.Tensor(ev_nest_cand_ids4nn['ev_labels_']))

                # prediction
                event4class, prediction, positive_idx, positive_ev_embs, nested_event_preds, nest_pred_mask = self.predict(
                    ev_embeds)
                event_preds_list.append(nested_event_preds)

                if self.params['compute_dem_loss']:
                    pred_event_labels = [item for sublist in nest_pred_mask for item in sublist]
                    pred_events_list.extend(pred_event_labels)
                    # save true event_id e.g. E1 or -1 if it's labeled by the predictor as non-event
                    ev_nest_cand_ids4nn['truth_ids_'] = [ev_nest_cand_ids4nn['event_ids'][i] if label == 1 else -1 for
                                                         i, label in enumerate(pred_event_labels)]
                    # save only the candidate events labeled as positive
                    ev_nest_cand_ids4nn['only_pos_events'] = [ev_nest_cand_ids4nn['event_ids'][i] for i in
                                                              positive_idx[0]]

                    current_truth_ids = ev_nest_cand_ids4nn['truth_ids_']

                # for modality
                if enable_modality and len(positive_idx[0]) > 0:
                    if self.params['compute_dem_loss']:
                        actual_nest_modality_labels = [ev_nest_cand_ids4nn['mod_labels'][i] for i in positive_idx[0]]
                        actual_nest_modality_labels = torch.Tensor(actual_nest_modality_labels).type(torch.LongTensor)
                        actual_modalities_list.append(actual_nest_modality_labels)

                    mod_preds, nested_modality_preds = self.predict_modality(positive_ev_embs, positive_idx,
                                                                             ev_nest_cand_ids4nn['mod_labels_'])
                    modality_preds_list.append(nested_modality_preds)
                    pred_mod_list.extend(mod_preds)
                else:
                    mod_preds = []

                # count nested level
                current_nested_level += 1
                current_positive_ids = positive_idx[0]

                # store positive ids
                all_positive_ids[current_nested_level] = current_positive_ids

                all_preds_output.append([ev_nest_cand_ids4nn, ev_nest_arg_ids4nn, current_positive_ids, mod_preds])

            # otherwise: stop loop
            else:
                enable_nested_ev = False

        if self.params['compute_dem_loss']:
            actual_event_labels = torch.flatten(torch.cat(actual_events_list)).to(self.device)
            event_preds = torch.flatten(torch.cat(event_preds_list))

            loss_events = self.bce(event_preds, actual_event_labels)

            if enable_modality and len(actual_modalities_list) > 0:
                actual_modalities_labels = torch.cat(actual_modalities_list).to(self.device)
                modality_preds = torch.cat(modality_preds_list, dim=0)

                loss_modalities = self.cel(modality_preds, actual_modalities_labels)

                loss_EV_layer = loss_events + loss_modalities

            else:
                loss_EV_layer = loss_events

            if self.params['compute_metrics']:
                self.compute_metrics(actual_event_labels, pred_events_list, actual_modalities_list, pred_mod_list,
                                     enable_modality)
        else:
            loss_EV_layer = None

        pred_ev_output = self.create_output(all_preds_output)

        return pred_ev_output, empty_pred, loss_EV_layer

    def forward(self, ner_preds, rel_preds):
        """Forward.
            Given entities and relations, event structures, return event prediction.
        """

        # check empty relation prediction
        if len(rel_preds['preds'].data) == 0:
            ev_preds = None
            empty_pred = True
            loss_EV_layer = None
            ev_accuracy = None

        else:
            # 1-get input
            # (batch_size, num_padding) e.g (16, 1694)
            # where num_padding is the maximum number of spans in a sentence
            # multiplied by two given the fact that we have 2 labels for each span
            etypes = ner_preds['ent_types']
            etypes = etypes.to(torch.device("cpu"))

            # entity and trigger embeddings [bert + type embeddings]
            # (batch_size, num_padding, embedding_size) e.g (16, 1694, 2604)
            # v_t = [m_t;s_t] for triggers and v_a = [m_a;s_a] for entities
            ent_embeds = rel_preds['enttoks_type_embeds']

            # all the triggers in the corpus with corresponding sentence and entity id e.g (tensor(0), tensor(8))
            tr_ids = (ner_preds['tr_ids'] == 1).nonzero(as_tuple=False).transpose(0, 1)
            tr_ids = list(zip(tr_ids[0], tr_ids[1]))

            # l2r is pairs_idx
            # >>> l2r.T[0]
            # tensor([  0,   8, 226])
            # rpred_types contains all the annotated pairs with corresponding role id e.g [12, -1, 16,.....]
            # rpred_ids contains the ids of the pairs with a role !='Other' e.g [0,2,6,.....,210] rpred_ids.size()=43
            l2r, rpred_types, rpred_ids = self.get_rel_input(rel_preds)

            if np.ndim(rpred_types) > 0:
                # r_i embeddings that give us a representation for each pair
                # (number_of_pairs, params['rel_reduced_size']) e.g. (228, 500)
                rel_embeds = rel_preds['rel_embeds']
            else:
                rel_embeds = torch.zeros((1, self.params['rel_reduced_size']), dtype=torch.float32, device=self.device)

                # avoid scalar error
                rpred_types = np.array([rpred_types])

            # 2-generate event candidates
            # As it's written on the paper "the event layer enumerates
            # all legal combinations of role pairs (We build templates from
            # the event structure definition.) to construct event candidates for
            # each trigger. These event candidates include events with no arguments."
            # about ev_ids4nn it's a dictionary containing three elements:
            # - 'ev_cand_ids4nn': contains all the information about the event candidates:
            #    - 'trids_': contains all the triggers ids for each flat event candidates e.g. [(0, 8), (0, 8),......,(15,476)]
            #    - 'io_ids': useful for obtaining the event arguments
            #    - 'ev_structs_': event structure, we have a list of arguments we can have multiple arguments for each event.
            #       We build different combinations for each trigger, for example if a trigger has 2 possible arguments we can have 3 possible
            #       events, 2 with the single arguments and 1 with both of them e.g. 'E1	Planned_process:TR48 Instrument:T1 Theme:T3'
            #       that are represented by a tuple where the first element indicates the role type and the second one the entity type e.g. (12, 42)
            # - 'ev_arg_ids4nn': for each trigger we have the corresponding arguments ids e.g.
            #    >>> ev_ids4nn['ev_arg_ids4nn'][(0,8)]
            #    [[0, 2], [(0, 226), (0, 358)]]  in this case the (0,8) trigger has 2 arguments
            #    in the first list we have the trigger-argument pair id and in the second one the arguments ids where the
            #    the first element in the pair indicates the sentence and the second one the actual id
            # - 'ev_nest_cand_triggers': the triggers for candidate nested events
            ev_ids4nn = self.ev_struct_generator._generate(etypes, tr_ids, l2r, rpred_types, rpred_ids)

            if self.params['compute_dem_loss']:
                all_ann_info = rel_preds['all_ann_info']
                entity_map = rel_preds['entity_map']
                fids = rel_preds['fids']
            else:
                all_ann_info = None
                entity_map = None
                fids = None

            # 3-embeds, prediction
            # check empty
            if len(ev_ids4nn['ev_cand_ids4nn']['trids_']) > 0:
                ev_preds, empty_pred, loss_EV_layer = self.calculate(ent_embeds, rel_embeds, rpred_types,
                                                                     ev_ids4nn,
                                                                     all_ann_info,
                                                                     entity_map, fids)

            else:
                ev_preds = None
                empty_pred = True
                loss_EV_layer = None
                ev_accuracy = None

        return ev_preds, empty_pred, loss_EV_layer
