import pickle
import numpy as np
import torch
import networkx as nx
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import GAT as GATPYG

from graph_augmented_sum.model.attention import badanau_attention
from graph_augmented_sum.model.util import len_mask, sequence_mean, sequence_loss
from graph_augmented_sum.model.summ import Seq2SeqSumm, AttentionalLSTMDecoder
from graph_augmented_sum.model import beam_search as bs
from graph_augmented_sum.model.extract import MeanSentEncoder
from graph_augmented_sum.model.rnn import lstm_multiembedding_encoder
from graph_augmented_sum.model.roberta import RobertaEmbedding
from graph_augmented_sum.model.scibert import ScibertEmbedding
from graph_augmented_sum.model.rnn import MultiLayerLSTMCells
from graph_augmented_sum.model.graph_enc import Block, RGCN

INIT = 1e-2
BERT_MAX_LEN = 512


class _CopyLinear(nn.Module):
    def __init__(self, context_dim, state_dim, input_dim, side_dim1, side_dim2=None, bias=True):
        super().__init__()
        self._v_c = nn.Parameter(torch.Tensor(context_dim))
        self._v_s = nn.Parameter(torch.Tensor(state_dim))
        self._v_i = nn.Parameter(torch.Tensor(input_dim))
        self._v_s1 = nn.Parameter(torch.Tensor(side_dim1))
        init.uniform_(self._v_c, -INIT, INIT)
        init.uniform_(self._v_s, -INIT, INIT)
        init.uniform_(self._v_i, -INIT, INIT)
        init.uniform_(self._v_s1, -INIT, INIT)
        if side_dim2 is not None:
            self._v_s2 = nn.Parameter(torch.Tensor(side_dim2))
            init.uniform_(self._v_s2, -INIT, INIT)
        else:
            self._v_s2 = None

        if bias:
            self._b = nn.Parameter(torch.zeros(1))
        else:
            self._b = None

    def forward(self, context, state, input_, side1, side2=None):
        output = (torch.matmul(context, self._v_c.unsqueeze(1))
                  + torch.matmul(state, self._v_s.unsqueeze(1))
                  + torch.matmul(input_, self._v_i.unsqueeze(1))
                  + torch.matmul(side1, self._v_s1.unsqueeze(1)))
        if side2 is not None and self._v_s2 is not None:
            output += torch.matmul(side2, self._v_s2.unsqueeze(1))
        if self._b is not None:
            output = output + self._b.unsqueeze(0)
        return output


class CopySummGat(Seq2SeqSumm):
    def __init__(self, vocab_size, lstm_dim,
                 n_hidden, bidirectional, n_layer, side_dim, etype_path, is_bipartite, bert_model, dropout=0.0, bert_length=512, gnn_model='gat'):
        super().__init__(vocab_size, lstm_dim,
                         n_hidden, bidirectional, n_layer, dropout)

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        assert bert_model in ['scibert', 'roberta-base']
        if bert_model == 'scibert':
            self._bert_model = ScibertEmbedding()
        elif bert_model == 'roberta-base':
            self._bert_model = RobertaEmbedding()

        self._eos_id = self._bert_model._eos_id
        self._bos_id = self._bert_model._bos_id
        self._unk_id = self._bert_model._unk_id
        self._pad_id = self._bert_model._pad_id
        self._embedding = self._bert_model._embedding
        self._embedding.weight.requires_grad = False
        self._emb_dim = self._embedding.weight.size(1)
        self._bert_max_length = bert_length
        self._enc_lstm = nn.LSTM(
            self._emb_dim, n_hidden, n_layer,
            bidirectional=bidirectional, dropout=dropout
        )

        self._projection = nn.Sequential(
            nn.Linear(2 * n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, self._emb_dim, bias=False)
        )
        self._dec_lstm = MultiLayerLSTMCells(
            self._emb_dim * 2, n_hidden, n_layer, dropout=dropout
        )
        # overlap between 2 lots to avoid breaking paragraph
        self._bert_stride = 256

        self._copy = _CopyLinear(n_hidden, n_hidden, 2 * self._emb_dim, side_dim, side_dim)

        if etype_path != '':
            etype_dict = pickle.load(open(etype_path, 'rb'))
            self._use_etype = True
            self._e_type_embs = etype_dict['embeddings'].to(self._device)
            self._e_type_mapping = etype_dict['mapping']
            self._node_size = n_hidden + self._e_type_embs.size()[1]
        else:
            self._use_etype = False
            self._node_size = n_hidden

        graph_hsz = self._node_size
        self._graph_hsz = graph_hsz

        self.gnn_model = gnn_model
        if gnn_model == 'gat':
            self.gnn = Block({'graph_hsz': graph_hsz, 'node_size': self._node_size})

        elif gnn_model == 'rgcn':
            with open('deep_event_mine/type_embs/dem_edge_types_mapping.pkl', 'rb') as etm:
                edge_types_mapping = pickle.load(etm)
            self.gnn = RGCN(graph_hsz, self._node_size, len(edge_types_mapping))

        elif gnn_model == 'rgat':
            with open('deep_event_mine/type_embs/dem_edge_types_mapping.pkl', 'rb') as etm:
                edge_types_mapping = pickle.load(etm)

            emb_matrix = torch.eye(len(edge_types_mapping))
            self.one_hot_embedding = torch.nn.Embedding.from_pretrained(emb_matrix, freeze=True)
            self.gnn = GATPYG(in_channels=self._node_size, hidden_channels=graph_hsz, num_layers=2, edge_dim=len(edge_types_mapping))

        self._node_enc = MeanSentEncoder()

        self._is_bipartite = is_bipartite
        if self._is_bipartite:
            self._r_type_embs = etype_dict['rel_embeddings'].to(self._device)
            self._r_type_mapping = etype_dict['rel_mapping']
            e_size = self._emb_dim + self._r_type_embs.size()[1]
            self._edge_proj = nn.Linear(e_size, graph_hsz)

        enc_lstm_in_dim = self._emb_dim
        self._enc_lstm = nn.LSTM(
            enc_lstm_in_dim, n_hidden, n_layer,
            bidirectional=bidirectional, dropout=dropout
        )

        # node attention
        self._attn_s1 = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self._attns_wm = nn.Parameter(torch.Tensor(graph_hsz, n_hidden))
        self._attns_wq = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self._attns_v = nn.Parameter(torch.Tensor(n_hidden))
        init.xavier_normal_(self._attns_wm)
        init.xavier_normal_(self._attns_wq)
        init.xavier_normal_(self._attn_s1)
        init.uniform_(self._attns_v, -INIT, INIT)

        self._projection_decoder = nn.Sequential(
            nn.Linear(3 * n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, self._emb_dim, bias=False)
        )

        self._decoder = CopyDecoderGraph(
            self._copy, self._eos_id, self._attn_s1, self._attns_wm, self._attns_wq, self._attn_v,
            self._embedding, self._dec_lstm, self._attn_wq, self._projection_decoder, self._attn_wb, self._attn_v,
        )

    def forward(self, artinfo, absinfo, ninfo, einfo):
        """
        - article: Tensor of size (n_split_docs, 512) where n_split_docs doesn't correspond to the number of documents
                   involved, but to the number of lots of tokens necessary to represent the original articles. E.g. when
                   we have 1 document as input the size is (2, 512), each element represents a token index.

        - art_lens: batch_size-dimensional list containing the number of tokens in each article e.g [745, 631, .....]

        - extend_art: tensor of size (batch_size, max(art_lens)), for each article we have all the token indexes for each article
                      padded to the maximum number of tokens in an article e.g (32, 745)

        - extend_vsize: highest value for a token index

        - abstract: tensor of size (batch_size, max_len_abstract) e.g. (32, 51), each token index represents the tokens
            contained in the abstract
        """

        article, art_lens, extend_art, extend_vsize, articles_ids = artinfo
        abstract, target = absinfo
        nodewords, nmask, adjs, node_num, node_type = ninfo

        # attention: contains the W_6 * h_k vectors (32, 775, 256)
        # init_dec_states[0]][0]: final hidden state ready for the decoder single-layer unidirectional LSTM
        # init_dec_states[0][1]: final cell state ready for the decoder single-layer unidirectional LSTM
        # init_dec_states[1] it's going do be used together with self._embedding(tok).squeeze(1)
        # as the feature embeddings of the decoder single-layer unidirectional LSTM
        # basically it's the concatenation of the final hidden state and the average of all
        # the token embeddings W_6 * h_k in the article e.g. (32, 768)
        attention, init_dec_states = self.encode(article, art_lens)

        node_vec, node_num = self._encode_graph(attention, nodewords, nmask, adjs, node_num, node_type, einfo)

        mask = len_mask(art_lens, attention.device).unsqueeze(-2)

        # logit: contains all the logits for each prediction that has to be made in the batch e.g. (190, 50265)
        #        where 190 is the number of tokens that have to be predicted in the 3 target documents
        logit = self._decoder(
            (attention, mask, extend_art, extend_vsize),
            abstract,
            node_vec,
            node_num,
            init_dec_states,
            )

        nll = lambda logit, target: F.nll_loss(logit, target, reduction='mean')
        loss = sequence_loss(logit, target, nll, self._pad_id)

        return loss

    def encode(self, article, art_lens=None):

        # We employ LSTM models with 256-dimensional
        # hidden states for the document encoder (128 each
        # direction) and the decoder
        # size = (2, 32, 256)
        # 2 = n_layer * 2 because bidirectional = True and n_layer = 1
        # 32 is batch_size
        size = (
            self._init_enc_h.size(0),
            len(art_lens) if art_lens else 1,
            self._init_enc_h.size(1)
        )

        # initial encoder states
        # _init_enc_h initial hidden state (2, 32, 256) we have a 2 because we have a bidirectional LSTM
        # _init_enc_c initial cell state (2, 32, 256)
        init_enc_states = (
            self._init_enc_h.unsqueeze(1).expand(*size),
            self._init_enc_c.unsqueeze(1).expand(*size)
        )

        # e.g. self._bert_max_length=1024
        if self._bert_max_length > 512:
            source_nums = art_lens

        # no finetuning Bert weights
        with torch.no_grad():
            # bert_out[0] (n_split_docs, 512, 768) contains token embeddings
            # bert_out[1] (n_split_docs, 768) contains sentence embeddings
            bert_out = self._bert_model(article)

        # e.g. (n_split_docs, 512, 768) so we obtain an embedding for each token
        # in the 'lots'
        bert_hidden = bert_out[0]
        if self._bert_max_length > 512:
            # e.g. 768
            hsz = bert_hidden.size(2)
            batch_id = 0

            # max(art_lens) e.g 775
            max_source = max(source_nums)

            bert_hiddens = []
            # e.g. 512
            max_len = bert_hidden.size(1)
            for source_num in source_nums:
                # tensor of zeros of size (max(art_lens), 768) e.g. (775, 768)
                source = torch.zeros(max_source, hsz).to(bert_hidden.device)
                if source_num < BERT_MAX_LEN:
                    source[:source_num, :] += bert_hidden[batch_id, :source_num, :]
                    batch_id += 1
                else:
                    # fill the first 512 tokens of the article
                    source[:BERT_MAX_LEN, :] += bert_hidden[batch_id, :BERT_MAX_LEN, :]
                    batch_id += 1
                    start = BERT_MAX_LEN
                    # now we deal with the remaining  source_num - BERT_MAX_LEN tokens e.g. 745 - 212
                    while start < source_num:
                        # print(start, source_num, max_source)
                        if start - self._bert_stride + BERT_MAX_LEN < source_num:
                            end = start - self._bert_stride + BERT_MAX_LEN
                            batch_end = BERT_MAX_LEN
                        else:
                            end = source_num
                            batch_end = source_num - start + self._bert_stride
                        source[start:end, :] += bert_hidden[batch_id, self._bert_stride:batch_end, :]
                        batch_id += 1
                        start += (BERT_MAX_LEN - self._bert_stride)
                bert_hiddens.append(source)

            # now bert hidden has changed size (batch_size, max(art_lens), 768) e.g. (32, 775, 768)
            # so now we have the token embeddings organised for each article
            bert_hidden = torch.stack(bert_hiddens)
        # article = self._bert_relu(self._bert_linear(bert_hidden))
        article = bert_hidden

        # enc_arts (max(art_lens), batch_size, 512) e.g. (775, 32, 512) each vector represents h_k of size 512
        # final_states: tuple of size 2 with each element of size e.g. (2, 32, 256)
        #               final_states[0] contains the final hidden states in both directions that's why we have a 2
        #               final_states[1] contains the final cell states in both directions
        enc_art, final_states = lstm_multiembedding_encoder(
            article, self._enc_lstm, art_lens,
            init_enc_states, None, {}, {}
        )

        if self._enc_lstm.bidirectional:
            h, c = final_states
            # final_states: tuple of size 2 with each element of size e.g. (1, 32, 512)
            # basically we concatenate the final hidden and cell states from both direction
            final_states = (
                torch.cat(h.chunk(2, dim=0), dim=2),
                torch.cat(c.chunk(2, dim=0), dim=2)
            )

        # in_features=512, out_features=256
        init_h = torch.stack([self._dec_h(s)
                              for s in final_states[0]], dim=0)
        init_c = torch.stack([self._dec_c(s)
                              for s in final_states[1]], dim=0)

        # init_dec_states[0]: final hidden state ready for the decoder single-layer unidirectional LSTM
        # init_dec_states[1]: final cell state ready for the decoder single-layer unidirectional LSTM
        init_dec_states = (init_h, init_c)

        # self._attn_wm is of size e.g (512, 256) so we get from (775, 32, 512)
        # to (775, 32, 256) and finally after transposing to (32, 775, 256)
        # basically we perform W_6 * h_k
        attention = torch.matmul(enc_art, self._attn_wm).transpose(0, 1)

        # we write init_h[-1] because we want the last layer output
        # we can have multiple layers, by default we just have 1
        # init_attn_out it's going do be used together with self._embedding(tok).squeeze(1)
        # as the feature embeddings of the decoder single-layer unidirectional LSTM
        # basically it's the concatenation of the final hidden state and the average of all
        # the token embeddings W_6 * h_k in the article e.g. (32, 768)
        init_attn_out = self._projection(torch.cat(
            [init_h[-1], sequence_mean(attention, art_lens, dim=1)], dim=1
        ))
        return attention, (init_dec_states, init_attn_out)

    def _encode_graph(self, articles, nodes, nmask, adjs, node_num, node_type, einfo):

        # embedding dimensionality of each token e.g.  256
        d_word = articles.size(-1)

        # (batch_size, max(ninfo[2]), max_words_node)
        bs, n_node, n_word = nodes.size()

        # we get the tokens embeddings of each node from the Roberta layers as explained
        # in the figure 2 of the paper in the Node Initialization phase
        # size of nodes is now (batch_size, max(ninfo[2]), max_words_node, d_word) e.g (32, 45, 20, 256)
        init_n_list = []
        etype_embs_list = []
        for did, doc in enumerate(nodes):
            for nid, nod in enumerate(doc):
                for wor in nod:

                    if wor == 0:
                        init_n_list.append(torch.zeros(d_word).to(self._device))
                    else:
                        init_n_list.append(articles[did, wor, :])

                if self._use_etype:

                    if sum(nod).item() > 0 or nid < node_num[did]:
                        n_type = node_type[did][nid]
                        etype_embs_list.append(self._e_type_embs[self._e_type_mapping[n_type]])

                    else:
                        etype_embs_list.append(torch.zeros(self._e_type_embs.size()[1]).to(self._device))

        nodes = torch.stack(init_n_list).view(bs, n_node, n_word, d_word)
        nmask = nmask.unsqueeze(3).expand(bs, n_node, n_word, d_word)

        # averaging the tokens embeddings that form the node text representation
        # size of node is now (batch_size, max(ninfo[2]), d_word) e.g. (32, 45, 256)
        nodes = self._node_enc(nodes, mask=nmask)

        if self._use_etype:
            etype_embs = torch.stack(etype_embs_list).view(bs, n_node, self._e_type_embs.size()[1])
            nodes = torch.cat((nodes, etype_embs), -1)

        if self._is_bipartite:

            edge_ids, edgetypes, edgewords, _ = einfo
            _, n_edge, n_eword = edgewords.size()
            nums_edge = [len(et) for et in edgetypes]

            with torch.no_grad():
                # edge_outputs[0] contains token embeddings
                # edge_outputs[1] (n_split_docs, 768) contains sentence embeddings
                edge_outputs = self._bert_model(edge_ids)[0]

            init_e_list = []
            rtype_embs_list = []
            for did, doc in enumerate(edgewords):
                for nid, nod in enumerate(doc):

                    for wor in nod:
                        if wor == -1:
                            init_e_list.append(torch.zeros(self._emb_dim).to(self._device))
                        else:
                            init_e_list.append(edge_outputs[did, wor, :])

                    if sum(nod) == -n_eword:
                        rtype_embs_list.append(torch.zeros(self._r_type_embs.size()[1]).to(self._device))
                    else:
                        e_type = edgetypes[did][nid]
                        rtype_embs_list.append(self._r_type_embs[self._r_type_mapping[e_type]].to(self._device))

            enodes = torch.stack(init_e_list).view(bs, n_edge, n_eword, self._emb_dim)
            ewords_mask = edgewords.unsqueeze(3).expand(bs, n_edge, n_eword, self._emb_dim).ne(-1).byte()
            enodes = self._node_enc(enodes, mask=ewords_mask)

            rtype_embs = torch.stack(rtype_embs_list).view(bs, n_edge, self._r_type_embs.size()[1])
            enodes = torch.cat((enodes, rtype_embs), -1)

            enodes = self._edge_proj(enodes)

            nodes_list = []
            for i, node in enumerate(nodes):
                full_nodes = torch.cat([node[:node_num[i], :], enodes[i, :nums_edge[i], :]], dim=0)
                nodes_list.append(full_nodes)
            node_num = [nnl.size(0) for nnl in nodes_list]
            nodes = pad_sequence(nodes_list, batch_first=True)

        init_nodes = nodes

        if self.gnn_model == 'gat':
            triple_outs = []
            for _i, adj in enumerate(adjs):

                # number of nodes e.g 31
                N = len(adj)

                if N > 0:

                    # just get the relevant nodes of the _i-th document graph
                    # e.g size (31, 256) where nodes size is e.g. (32, 45, 256)
                    ngraph = nodes[_i, :N, :]  # N * d
                    mask = (adj == 0)  # N * N
                    triple_out = self.gnn(ngraph, ngraph, mask)

                else:
                    triple_out = None

                triple_outs.append(triple_out)

            max_n = max(node_num)

            nodes_list = []
            for s, n in zip(triple_outs, node_num):
                if n == 0:
                    nodes_list.append(torch.zeros(max_n - n, self._graph_hsz).to(self._device))
                elif n != max_n:
                    nodes_list.append(torch.cat([s, torch.zeros(max_n - n, self._graph_hsz).to(self._device)], dim=0))
                else:
                    nodes_list.append(s)

            # e.g. (3, 45, 256)
            nodes = torch.stack(nodes_list, dim=0)

        else:
            _, _, _, edge_features = einfo

            if len(edge_features) == 0:
                nodes = torch.zeros_like(init_nodes).to(self._device)
            else:
                batch_graph = nx.DiGraph()
                batch_nodes = []
                for batch_i, adj in enumerate(adjs):
                    N = len(adj)

                    if N > 0:
                        graph = nx.from_numpy_matrix(np.matrix(adj.cpu()))
                        batch_graph = nx.union(batch_graph, graph, rename=('', f'{batch_i}-'))
                        batch_nodes.append(nodes[batch_i, :N, :])

                batch_adj = torch.from_numpy(nx.adjacency_matrix(batch_graph).todense()).to(self._device)
                edge_index = batch_adj.nonzero().t().contiguous()
                batch_nodes = torch.cat(batch_nodes, dim=0)

                if self.gnn_model == 'rgat':
                    edge_attr = self.one_hot_embedding.weight[edge_features]
                    nodes = self.gnn(batch_nodes, edge_index, edge_attr=edge_attr)

                elif self.gnn_model == 'rgcn':
                    nodes = self.gnn(batch_nodes, edge_index, edge_features)

                node_out_list = []
                curr_batch_num = 0
                prev_batch_num = 0
                for batch_id, batch_num in enumerate(node_num):
                    curr_batch_num += batch_num
                    node_out_list.append(nodes[prev_batch_num:curr_batch_num])
                    prev_batch_num += batch_num

                nodes = pad_sequence(node_out_list, batch_first=True)

        nodes = init_nodes + nodes

        return nodes, node_num

    def batched_beamsearch(self, info, max_len, beam_size,
                           diverse=1.0, min_len=35):

        article, art_lens, extend_art, extend_vsize, articles_ids = info[0]
        nodewords, nmask, adjs, node_num, node_type = info[1]

        batch_size = len(art_lens)

        # vocabulary size e.g. 50265
        vsize = self._embedding.num_embeddings
        attention, init_dec_states = self.encode(article, art_lens)

        nodes, node_num = self._encode_graph(attention, nodewords, nmask, adjs, node_num, node_type, info[2])

        mask = len_mask(art_lens, attention.device).unsqueeze(-2)
        all_attention = (attention, mask, extend_art, extend_vsize)
        attention = all_attention

        # h size e.g. (1, 32, 256)
        # prev size e.g. (3, 768)
        (h, c), prev = init_dec_states

        all_beams = [bs.init_beam(self._bos_id, (h[:, i, :], c[:, i, :], prev[i]))
                     for i in range(batch_size)]
        max_node_num = max(node_num)

        all_nodes = [(nodes[i, :, :], node_num[i]) for i in range(len(node_num))]
        finished_beams = [[] for _ in range(batch_size)]
        outputs = [None for _ in range(batch_size)]

        for t in range(max_len):
            toks = []
            all_states = []
            for beam in filter(bool, all_beams):
                token, states = bs.pack_beam(beam, article.device)
                toks.append(token)
                all_states.append(states)
            token = torch.stack(toks, dim=1)
            states = ((torch.stack([h for (h, _), _ in all_states], dim=2),
                       torch.stack([c for (_, c), _ in all_states], dim=2)),
                      torch.stack([prev for _, prev in all_states], dim=1))
            token.masked_fill_(token >= vsize, self._unk_id)

            filtered_nodes = torch.stack([all_nodes[i][0] for i, _beam in enumerate(all_beams) if _beam != []], dim=0)
            filtered_node_num = [all_nodes[i][1] for i, _beam in enumerate(all_beams) if _beam != []]

            if t < min_len:
                force_not_stop = True
            else:
                force_not_stop = False
            topk, lp, states, attn_score = self._decoder.topk_step(
                token, states, attention, beam_size, filtered_nodes, filtered_node_num,
                max_node_num, force_not_stop)

            batch_i = 0
            for i, (beam, finished) in enumerate(zip(all_beams,
                                                     finished_beams)):
                if not beam:
                    continue
                finished, new_beam = bs.next_search_beam(
                    beam, beam_size, finished, self._eos_id,
                    topk[:, batch_i, :], lp[:, batch_i, :],
                    (states[0][0][:, :, batch_i, :],
                     states[0][1][:, :, batch_i, :],
                     states[1][:, batch_i, :]),
                    attn_score[:, batch_i, :],
                    diverse
                )
                batch_i += 1
                if len(finished) >= beam_size:
                    all_beams[i] = []
                    outputs[i] = finished[:beam_size]
                    # exclude finished inputs
                    (attention, mask, extend_art, extend_vsize
                     ) = all_attention
                    masks = [mask[j] for j, o in enumerate(outputs)
                             if o is None]
                    ind = [j for j, o in enumerate(outputs) if o is None]
                    ind = torch.LongTensor(ind).to(attention.device)
                    attention, extend_art = map(
                        lambda v: v.index_select(dim=0, index=ind),
                        [attention, extend_art]
                    )
                    if masks:
                        mask = torch.stack(masks, dim=0)
                    else:
                        mask = None
                    attention = (
                        attention, mask, extend_art, extend_vsize)
                else:
                    all_beams[i] = new_beam
                    finished_beams[i] = finished
            if all(outputs):
                break
        else:
            for i, (o, f, b) in enumerate(zip(outputs,
                                              finished_beams, all_beams)):
                if o is None:
                    outputs[i] = (f + b)[:beam_size]
        return outputs


class CopyDecoderGraph(AttentionalLSTMDecoder):
    def __init__(self, copy, eos, attn_s1, attns_wm, attns_wq, attns_v, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._copy = copy
        self._eos = eos
        self._attn_s1 = attn_s1
        self._attns_wm = attns_wm
        self._attns_wq = attns_wq
        self._attns_v = attns_v

    def __call__(self, attention, target, nodes, node_num, init_states):
        # max abstract length in the batch
        max_len = target.size(1)
        states = init_states
        logits = []

        # loop over all target tokens
        for i in range(max_len):
            # target token index for each document e.g. [0, 12, ....] (32, 1)
            tok = target[:, i:i + 1]

            # tensor of size (batch_size, vobulary_size) e.g. (2, 50265)
            # contains all the y_hat_j_t for each document
            logit, states = self._step(tok, attention, nodes, node_num, states)
            logits.append(logit)

        logit = torch.stack(logits, dim=1)

        return logit

    def _step(self, tok, attention, nodes, node_num, states):

        # (32, 768)
        # prev_out  basically it's the concatenation of the final hidden state
        # and the average of all the token embeddings W_6 * h_k in the article
        prev_states, prev_out = states

        # self._embedding(tok).squeeze(1) gets the target token embeddings given by Roberta e.g. (32, 768)
        # lstm has size e.g. (32, 1536) where 1536 is given by the concatenation of 2 768-dimensional embeddings
        decoder_in = torch.cat(
            [self._embedding(tok).squeeze(1), prev_out],
            dim=1)

        # decoder_in the input x of the LSTM cell
        # prev_states[0] contains the last hidden state
        # prev_states[1] contains the last cell state
        states = self._lstm(decoder_in, prev_states)

        # we save the last layer state, by default we just have 1 layer so we just
        # get the first and last state
        # state[0] e.g. (1, 32, 256) ---> decoder_out (32, 256)
        # decoder_out stores the 32 s_t hidden states, 1 for each document
        decoder_out = states[0][-1]

        # W_5 * s_t e.g. (32, 256)
        query = torch.mm(decoder_out, self._attn_w)
        attention, attn_mask, extend_src, extend_vsize = attention

        # W_4 * G^ where each element of each G^ is v_i^ e.g. (32, 45, 256)
        nodes = torch.matmul(nodes, self._attns_wm)

        # W_3 * s_t e.g. (32, 256)
        query_s = torch.mm(decoder_out, self._attns_wq)

        # (32, 1, 45)
        nmask = len_mask(node_num, attention.device).unsqueeze(-2)

        # side_n is of size(32, 256), each of the 256-dimensional vector represents c_t^v
        # score_n is of size e.g. (32, 45) each the 45 scalar represents alpha_i_t^v
        side_n, _ = badanau_attention(query_s, nodes, nodes, mem_mask=nmask, v=self._attns_v,
                                            sigmoid=False)

        # W_7 *  c_t^v e.g (32, 256)
        side_n = torch.mm(side_n, self._attn_s1)

        # context is of size(32, 256), each of the 256-dimensional vector represents c_t
        # score is of size e.g. (32, 775) each the 775 scalar represents alpha_k_t
        context, score = badanau_attention(
            query, attention, attention, mem_mask=attn_mask, bias=self._attn_b, v=self._attn_v, side=side_n)

        # self._projection applies an hyperplane 768*256,
        # a tanh activation function and another hyperplane 256*768
        # dec_out is of size (32, 768), it's going to be used also as prev_out
        # at the next step
        # torch.cat([decoder_out, context, side_n] = [s_t|c_t|c_t^v]
        dec_out = self._projection(torch.cat([decoder_out, context, side_n], dim=1))

        # extend generation prob to extended vocabulary
        # softmax(W_out * [s_t|c_t|c_t^v])
        gen_prob = self._compute_gen_prob(dec_out, extend_vsize)

        # P_copy = σ(W_copy[s_t|c_t|c_t^v|y_t−1]) e.g. (32, 1)
        copy_prob = torch.sigmoid(self._copy(context, decoder_out, decoder_in, side_n))

        # add the copy prob to existing vocab distribution
        lp = torch.log(
            ((-copy_prob + 1) * gen_prob
             ).scatter_add(
                dim=1,
                index=extend_src.expand_as(score),
                src=score * copy_prob
            ) + 1e-10)  # numerical stability for log

        return lp, (states, dec_out)

    def decode_step(self, tok, states, attention, nodes, node_num, side_mask=None, output_attn=False, ext_info=None,
                    paras=None):
        logit, states, score, score_n = self._step(tok, states, attention, nodes, node_num, side_mask, output_attn,
                                                   ext_info, paras)
        out = torch.max(logit, dim=1, keepdim=True)[1]
        return out, states, score, score_n

    def sample_step(self, tok, states, attention, nodes, node_num, side_mask=None, output_attn=False, ext_info=None,
                    paras=None):
        logit, states, score, score_n = self._step(tok, states, attention, nodes, node_num, side_mask, output_attn,
                                                   ext_info, paras)
        # logprob = F.log_softmax(logit, dim=1)
        logprob = logit
        score = torch.exp(logprob)
        # out = torch.multinomial(score, 1).detach()
        out = torch.multinomial(score, 1)
        sampleProb = logprob.gather(1, out)
        return out, states, score, sampleProb

    def topk_step(self, tok, states, attention, k, nodes, node_num, max_node_num, force_not_stop=False):

        # h size (1, beam_size, batch_size, 256) e.g. (1, 5, 3, 256) for t > 0
        (h, c), prev_out = states

        # lstm is not beamable
        # nl indicates the number of lstm layers default 1
        nl, _, _, d = h.size()

        # beam = 1 at t = 0, 5 else
        beam, batch = tok.size()

        # self._embedding(tok) gets the target token embeddings given by Roberta e.g. (32, 768)
        # lstm has size e.g. (32, 1536) where 1536 is given by the concatenation of 2 768-dimensional embeddings
        decoder_in = torch.cat(
            [self._embedding(tok), prev_out], dim=-1)

        # (beam * batch, 768) e.g. (15, 768) for t > 0
        lstm_in = decoder_in.contiguous().view(beam * batch, -1)
        prev_states = (h.contiguous().view(nl, -1, d),
                       c.contiguous().view(nl, -1, d))

        # h size is (nl, beam_size*batch_size, 256) e.g. (1, 15, 256)
        # decoder_in the input x of the LSTM cell
        # h contains the last hidden state
        # c contains the last cell state
        h, c = self._lstm(lstm_in, prev_states)

        # we go back to the (nl, beam_size, batch_size, 256) e.g. (1, 5, 3, 256)
        states = (h.contiguous().view(nl, beam, batch, -1),
                  c.contiguous().view(nl, beam, batch, -1))

        # get hidden state of last layer e.g (5, 3, 256)
        decoder_out = states[0][-1]

        # W_5 * s_t
        query = torch.matmul(decoder_out, self._attn_w)
        attention, attn_mask, extend_src, extend_vsize = attention

        # W_4 * G^ where each element of each G^ is v_i^
        nodes = torch.matmul(nodes, self._attns_wm)

        # W_3 * s_t e.g.
        query_s = torch.matmul(decoder_out, self._attns_wq)
        nmask = len_mask(node_num, attention.device, max_num=max_node_num).unsqueeze(-2)

        # side_n: each of the 256-dimensional vector represents c_t^v
        # score_n: each the 45 scalar represents alpha_i_t^v
        side_n, score_n = badanau_attention(query_s, nodes, nodes, mem_mask=nmask, v=self._attns_v)

        # W_7 *  c_t^v
        side_n = torch.matmul(side_n, self._attn_s1)

        # context: each of the 256-dimensional vector represents c_t
        # score: each the 775 scalar represents alpha_k_t
        context, score = badanau_attention(
            query, attention, attention, mem_mask=attn_mask, bias=self._attn_b, v=self._attn_v, side=side_n)

        # self._projection applies an hyperplane 768*256,
        # a tanh activation function and another hyperplane 256*768
        # dec_out it's going to be used also as prev_out
        # at the next step
        # torch.cat([decoder_out, context, side_n] = [s_t|c_t|c_t^v]
        dec_out = self._projection(torch.cat([decoder_out, context, side_n], dim=-1))
        score_copy = score

        # extend generation prob to extended vocabulary
        # softmax(W_out * [s_t|c_t|c_t^v])
        # copy mechanism is not beamable
        gen_prob = self._compute_gen_prob(
            dec_out.contiguous().view(batch * beam, -1), extend_vsize)

        # P_copy = σ(W_copy[s_t|c_t|c_t^v|y_t−1])
        copy_prob = torch.sigmoid(self._copy(context, decoder_out, decoder_in, side_n)).contiguous().view(-1, 1)

        lp = torch.log(
            ((-copy_prob + 1) * gen_prob
             ).scatter_add(
                dim=1,
                index=extend_src.expand_as(score_copy).contiguous().view(
                    beam * batch, -1),
                src=score_copy.contiguous().view(beam * batch, -1) * copy_prob
            ) + 1e-8).contiguous().view(beam, batch, -1)

        if force_not_stop:
            lp[:, :, self._eos] = -1e8

        k_lp, k_tok = lp.topk(k=k, dim=-1)

        return k_tok, k_lp, (states, dec_out), score_copy

    def _compute_gen_prob(self, dec_out, extend_vsize, eps=1e-6):

        # what is represented as in the paper W_out * [s_t|c_t|c_t^v]
        logit = torch.mm(dec_out, self._embedding.weight.t())
        bsize, vsize = logit.size()

        # if we have a token with larger index than vsize that is the vocabulary
        # size of Roberta, we fill the logit vectors with values approximately
        # equal to zero 1e-6
        if extend_vsize > vsize:
            ext_logit = torch.Tensor(bsize, extend_vsize - vsize
                                     ).to(logit.device)
            ext_logit.fill_(eps)
            gen_logit = torch.cat([logit, ext_logit], dim=1)
        else:
            gen_logit = logit

        # softmax(W_out * [s_t|c_t|c_t^v])
        gen_prob = F.softmax(gen_logit, dim=-1)
        return gen_prob

    def _compute_copy_activation(self, context, state, input_, score):
        copy = self._copy(context, state, input_) * score
        return copy