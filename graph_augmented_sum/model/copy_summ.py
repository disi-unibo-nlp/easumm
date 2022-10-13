import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

from graph_augmented_sum.model.attention import badanau_attention
from graph_augmented_sum.model.util import len_mask, sequence_mean, sequence_loss
from graph_augmented_sum.model.summ import Seq2SeqSumm, AttentionalLSTMDecoder
from graph_augmented_sum.model import beam_search as bs
from graph_augmented_sum.model.rnn import lstm_multiembedding_encoder
from graph_augmented_sum.model.roberta import RobertaEmbedding
from graph_augmented_sum.model.scibert import ScibertEmbedding
from graph_augmented_sum.model.rnn import MultiLayerLSTMCells

INIT = 1e-2
BERT_MAX_LEN = 512


class _CopyLinear(nn.Module):
    def __init__(self, context_dim, state_dim, input_dim, bias=True):
        super().__init__()
        self._v_c = nn.Parameter(torch.Tensor(context_dim))
        self._v_s = nn.Parameter(torch.Tensor(state_dim))
        self._v_i = nn.Parameter(torch.Tensor(input_dim))
        init.uniform_(self._v_c, -INIT, INIT)
        init.uniform_(self._v_s, -INIT, INIT)
        init.uniform_(self._v_i, -INIT, INIT)

        if bias:
            self._b = nn.Parameter(torch.zeros(1))
        else:
            self._b = None

    def forward(self, context, state, input_):
        output = (torch.matmul(context, self._v_c.unsqueeze(1))
                  + torch.matmul(state, self._v_s.unsqueeze(1))
                  + torch.matmul(input_, self._v_i.unsqueeze(1))
                  )

        if self._b is not None:
            output = output + self._b.unsqueeze(0)
        return output


class CopySumm(Seq2SeqSumm):
    def __init__(self, vocab_size, emb_dim,
                 n_hidden, bidirectional, n_layer, dropout, bert_length, bert_model):
        super().__init__(vocab_size, emb_dim,
                         n_hidden, bidirectional, n_layer, dropout)

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
        emb_dim = self._embedding.weight.size(1)
        self._bert_max_length = bert_length
        self._enc_lstm = nn.LSTM(
            emb_dim, n_hidden, n_layer,
            bidirectional=bidirectional, dropout=dropout
        )
        self._projection = nn.Sequential(
            nn.Linear(2 * n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, emb_dim, bias=False)
        )
        self._dec_lstm = MultiLayerLSTMCells(
            emb_dim * 2, n_hidden, n_layer, dropout=dropout
        )
        self._bert_stride = 256

        self._copy = _CopyLinear(n_hidden, n_hidden, 2 * emb_dim)

        self._decoder = CopyLSTMDecoder(
            self._copy, self._eos_id, self._embedding, self._dec_lstm,
            self._attn_wq, self._projection, self._attn_wb, self._attn_v
        )

    def forward(self, artinfo, absinfo):
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

        article, art_lens, extend_art, extend_vsize = artinfo
        abstract, target = absinfo

        # attention: contains the W_6 * h_k vectors (32, 775, 256)
        # init_dec_states[0]][0]: final hidden state ready for the decoder single-layer unidirectional LSTM
        # init_dec_states[0][1]: final cell state ready for the decoder single-layer unidirectional LSTM
        # init_dec_states[1] it's going do be used together with self._embedding(tok).squeeze(1)
        # as the feature embeddings of the decoder single-layer unidirectional LSTM
        # basically it's the concatenation of the final hidden state and the average of all
        # the token embeddings W_6 * h_k in the article e.g. (32, 768)
        attention, init_dec_states = self.encode(article, art_lens)

        mask = len_mask(art_lens, attention.device).unsqueeze(-2)

        # logit: contains all the logits for each prediction that has to be made in the batch e.g. (190, 50265)
        #        where 190 is the number of tokens that have to be predicted in the 3 target documents
        logit = self._decoder(
            (attention, mask, extend_art, extend_vsize),
            abstract,
            init_dec_states)

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

    def greedy(self, artinfo, max_len):

        article, art_lens, extend_art, extend_vsize = artinfo

        batch_size = len(art_lens)
        vsize = self._embedding.num_embeddings

        # attention: contains the W_6 * h_k vectors (32, 775, 256)
        # init_dec_states[0]][0]: final hidden state ready for the decoder single-layer unidirectional LSTM
        # init_dec_states[0][1]: final cell state ready for the decoder single-layer unidirectional LSTM
        # init_dec_states[1] it's going do be used together with self._embedding(tok).squeeze(1)
        # as the feature embeddings of the decoder single-layer unidirectional LSTM
        # basically it's the concatenation of the final hidden state and the average of all
        # the token embeddings W_6 * h_k in the article e.g. (32, 768)
        attention, init_dec_states = self.encode(article, art_lens)

        mask = len_mask(art_lens, attention.device).unsqueeze(-2)
        attention = (attention, mask, extend_art, extend_vsize)

        tok = torch.LongTensor([self._bos_id] * batch_size).to(article.device)

        outputs = []
        attns = []
        states = init_dec_states
        for i in range(max_len):
            tok, states, attn_score, node_attn_score = self._decoder.decode_step(
                tok, attention, states)

            if i == 0:
                unfinished = (tok != self._eos_id)
                # print('greedy tok:', tok)
            else:
                it = tok * unfinished.type_as(tok)
                unfinished = unfinished * (it != self._eos_id)
            attns.append(attn_score.detach())
            if i == 0:
                outputs.append(tok[:, 0].clone())
            else:
                outputs.append(it[:, 0].clone())
            tok.masked_fill_(tok >= vsize, self._unk_id)
            if unfinished.data.sum() == 0:
                break

        return outputs, attns

    def batched_beamsearch(self, artinfo, max_len, beam_size, diverse=1.0, min_len=35):

        article, art_lens, extend_art, extend_vsize = artinfo
        batch_size = len(art_lens)

        # vocabulary size e.g. 50265
        vsize = self._embedding.num_embeddings

        # attention: contains the W_6 * h_k vectors (32, 775, 256)
        # init_dec_states[0]][0]: final hidden state ready for the decoder single-layer unidirectional LSTM
        # init_dec_states[0][1]: final cell state ready for the decoder single-layer unidirectional LSTM
        # init_dec_states[1] it's going do be used together with self._embedding(tok).squeeze(1)
        # as the feature embeddings of the decoder single-layer unidirectional LSTM
        # basically it's the concatenation of the final hidden state and the average of all
        # the token embeddings W_6 * h_k in the article e.g. (32, 768)
        attention, init_dec_states = self.encode(article, art_lens)

        mask = len_mask(art_lens, attention.device).unsqueeze(-2)
        all_attention = (attention, mask, extend_art, extend_vsize)
        attention = all_attention

        # h size e.g. (1, 32, 256)
        # prev size e.g. (3, 768)
        (h, c), prev = init_dec_states

        # list of length batch_size e.g. 32 where all the beam search initialisations are stored
        all_beams = [bs.init_beam(self._bos_id, hists=(h[:, i, :], c[:, i, :], prev[i]))
                     for i in range(batch_size)]

        finished_beams = [[] for _ in range(batch_size)]
        outputs = [None for _ in range(batch_size)]

        for t in range(max_len):

            all_states = []
            toks = []
            # with t = 0 len(beam) = 1, with t > 0 len(beam) = 5
            # at the first step we just have 1 because we consider just the eos token
            # we don't have to store the top 5 most likely hypothesis
            for beam in filter(bool, all_beams):
                token, states = bs.pack_beam(beam, article.device)
                all_states.append(states)

                toks.append(token)

            token = torch.stack(toks, dim=1)
            # mask tokens that are not in the vocabulary with the unk token
            token.masked_fill_(token >= vsize, self._unk_id)

            # states[0][0] contains the hidden states e.g. (1, 1, 32, 256) at t=0 and (1, 5, 32, 256) at t > 0
            # states[0][1] contains the cell states e.g. (1, 1, 32, 256) at t=0 and (1, 5, 32, 256) at t > 0
            # state[1] contains the prev_states e.g. (1, 32, 768) at t=0 and (5, 32, 768) at t > 0
            states = ((torch.stack([h for (h, _), _ in all_states], dim=2),
                       torch.stack([c for (_, c), _ in all_states], dim=2)),
                      torch.stack([prev for _, prev in all_states], dim=1))

            if t < min_len:
                force_not_stop = True
            else:
                force_not_stop = False

            topk, lp, states, attn_score = self._decoder.topk_step(token, attention, states, beam_size, force_not_stop)

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


class CopyLSTMDecoder(AttentionalLSTMDecoder):
    def __init__(self, copy, eos, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._copy = copy
        self._eos_id = eos

    def __call__(self, attention, target, init_states):
        # max abstract length in the batch
        max_len = target.size(1)
        states = init_states
        logits = []

        # loop over all target tokens
        for i in range(max_len):
            # target token index for each document e.g. [0, 12, ....] (32, 1)
            tok = target[:, i:i+1]

            # tensor of size (batch_size, vobulary_size) e.g. (2, 50265)
            # contains all the y_hat_j_t for each document
            logit, states = self._step(tok, attention, states)
            logits.append(logit)

        logit = torch.stack(logits, dim=1)

        return logit

    def _step(self, tok, attention, states):

        # (32, 768)
        # prev_out  basically it's the concatenation of the final hidden state
        # and the average of all the token embeddings W_6 * h_k in the article
        prev_states, prev_out = states

        # self._embedding(tok).squeeze(1) gets the target token embeddings given by Roberta e.g. (32, 768)
        # lstm has size e.g. (32, 1536) where 1536 is given by the concatenation of 2 768-dimensional embeddings
        lstm_in = torch.cat(
            [self._embedding(tok).squeeze(1), prev_out],
            dim=1)

        # decoder_in the input x of the LSTM cell
        # prev_states[0] contains the last hidden state
        # prev_states[1] contains the last cell state
        states = self._lstm(lstm_in, prev_states)

        # we save the last layer state, by default we just have 1 layer so we just
        # get the first and last state
        # state[0] e.g. (1, 32, 256) ---> decoder_out (32, 256)
        # decoder_out stores the 32 s_t hidden states, 1 for each document
        lstm_out = states[0][-1]

        # W_5 * s_t e.g. (32, 256)
        query = torch.mm(lstm_out, self._attn_w)
        attention, attn_mask, extend_src, extend_vsize = attention

        # context is of size(32, 256), each of the 256-dimensional vector represents c_t
        # score is of size e.g. (32, 775) each the 775 scalar represents alpha_k_t
        context, score = badanau_attention(
            query, attention, attention, mem_mask=attn_mask, bias=self._attn_b, v=self._attn_v)

        # self._projection applies an hyperplane 768*256,
        # a tanh activation function and another hyperplane 256*768
        # dec_out is of size (32, 768), it's going to be used also as prev_out
        # at the next step
        # torch.cat([decoder_out, context] = [s_t|c_t]
        dec_out = self._projection(torch.cat([lstm_out, context], dim=1))

        # extend generation prob to extended vocabulary
        # softmax(W_out * [s_t|c_t])
        gen_prob = self._compute_gen_prob(dec_out, extend_vsize)

        # P_copy = σ(W_copy[s_t|c_t|y_t−1]) e.g. (32, 1)
        copy_prob = torch.sigmoid(self._copy(context, states[0][-1], lstm_in))

        # add the copy prob to existing vocab distribution
        lp = torch.log(
            ((-copy_prob + 1) * gen_prob
             ).scatter_add(
                dim=1,
                index=extend_src.expand_as(score),
                src=score * copy_prob
            ) + 1e-8)  # numerical stability for log

        return lp, (states, dec_out)

    def decode_step(self, tok, attention, states):
        logit, states= self._step(tok, attention, states)
        out = torch.max(logit, dim=1, keepdim=True)[1]
        return out, states, None, None

    def topk_step(self, tok, attention, states, k, force_not_stop):

        # h size (1, beam_size, batch_size, 256) e.g. (1, 5, 3, 256) for t > 0
        (h, c), prev_out = states

        # lstm is not beamable
        nl, _, _, d = h.size()

        # beam = 1 at t = 0, 5 else
        beam, batch = tok.size()
        decoder_in = torch.cat(
            [self._embedding(tok), prev_out], dim=-1)

        # (beam * batch, 768) e.g. (15, 768) for t > 0
        lstm_in = decoder_in.contiguous().view(beam * batch, -1)
        prev_states = (h.contiguous().view(nl, -1, d),
                       c.contiguous().view(nl, -1, d))

        # h size is (1, beam_size*batch_size, 256) e.g. (1, 15, 256)
        h, c = self._lstm(lstm_in, prev_states)
        states = (h.contiguous().view(nl, beam, batch, -1),
                  c.contiguous().view(nl, beam, batch, -1))

        decoder_out = states[0][-1]

        query = torch.matmul(decoder_out, self._attn_w)
        attention, attn_mask, extend_src, extend_vsize = attention

        context, score = badanau_attention(
            query, attention, attention, mem_mask=attn_mask, bias=self._attn_b, v=self._attn_v)

        dec_out = self._projection(torch.cat([decoder_out, context], dim=-1))
        score_copy = score

        # copy mechanism is not beamable
        gen_prob = self._compute_gen_prob(
            dec_out.contiguous().view(batch * beam, -1), extend_vsize)

        copy_prob = torch.sigmoid(self._copy(context, decoder_out, decoder_in)).contiguous().view(-1, 1)

        lp = torch.log(
            ((-copy_prob + 1) * gen_prob
             ).scatter_add(
                dim=1,
                index=extend_src.expand_as(score_copy).contiguous().view(
                    beam * batch, -1),
                src=score_copy.contiguous().view(beam * batch, -1) * copy_prob
            ) + 1e-8).contiguous().view(beam, batch, -1)

        if force_not_stop:
            lp[:, :, self._eos_id] = -1e8

        k_lp, k_tok = lp.topk(k=k, dim=-1)

        return k_tok, k_lp, (states, dec_out), score_copy

    def _compute_gen_prob(self, dec_out, extend_vsize, eps=1e-6):

        # what is represented as in the paper W_out * [s_t|c_t]
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

        # softmax(W_out * [s_t|c_t])
        gen_prob = F.softmax(gen_logit, dim=-1)
        return gen_prob
