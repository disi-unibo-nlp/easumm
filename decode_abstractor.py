import json
import argparse
import os
import shutil
from os.path import join, exists

import torch
from torch import multiprocessing as mp
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, AutoTokenizer

from cytoolz import compose, curry, concat
from itertools import starmap, product
from collections import Counter, defaultdict
from nltk import sent_tokenize
from tqdm import tqdm
from codecarbon import EmissionsTracker
from timeit import default_timer as timer

from graph_augmented_sum.utils import load_latest_ckpt
from graph_augmented_sum.model.copy_summ_multiencoder import CopySummGat
from graph_augmented_sum.model.copy_summ import CopySumm
from graph_augmented_sum.data.batcher import coll_fn
from graph_augmented_sum.data.data import csvDataset
from graph_augmented_sum.data.batcher import prepro_fn_gat_bert, convert_batch_gat_bert, batchify_fn_gat_bert, \
    batchify_fn_copy_bert, convert_batch_bert, prepro_fn_bert


class BeamAbstractor(object):
    def __init__(self, tokenizer, cuda, min_len, max_len, max_art, max_abs, nograph_channel, is_bipartite):
        self._min_len = min_len
        self._max_len = max_len
        self._unk = tokenizer.unk_token_id
        self._bos = tokenizer.cls_token_id
        self._eos = tokenizer.eos_token_id
        self._id2word = {i: w for w, i in tokenizer.encoder.items()}
        if nograph_channel:
            self._batchify = compose(
                batchify_fn_copy_bert(tokenizer, cuda=cuda, test=True),
                convert_batch_bert(tokenizer, max_art),
                prepro_fn_bert(tokenizer, max_art, max_abs)
            )

        else:
            self._batchify = compose(
                batchify_fn_gat_bert(tokenizer, cuda=cuda, is_bipartite=args.is_bipartite, test=True),
                convert_batch_gat_bert(tokenizer, max_art),
                prepro_fn_gat_bert(tokenizer, max_art, max_abs, 'test', args.dem_model,
                                   is_bipartite=args.is_bipartite)
            )

    def __call__(self, net, batch, beam_size=5, diverse=1.0):
        net.eval()
        batch = self._batchify(batch)
        dec_args = (batch, self._max_len, beam_size, diverse, self._min_len)

        all_beams = net.batched_beamsearch(*dec_args)

        all_beams = list(starmap(_process_beam(self._id2word),
                                 zip(all_beams)))
        return all_beams


@curry
def _process_beam(id2word, beam):
    def process_hyp(hyp):
        seq = []
        for i, attn in zip(hyp.sequence[1:], hyp.attns[:-1]):
            if i == 3:
                seq.append('unk')
            else:
                seq.append(id2word[i])
        hyp.sequence = seq
        del hyp.hists
        del hyp.attns
        # del hyp.coverage
        return hyp

    return list(map(process_hyp, beam))


def configure_decoding(model_dir, cuda, min_len, max_len, max_art):

    abs_args = json.load(open(join(model_dir, 'meta.json')))
    net_args = abs_args['net_args']
    bert_model = net_args['bert_model']
    ckpt = load_latest_ckpt(model_dir)

    if abs_args['nograph_channel']:
        net = CopySumm(**net_args)
    else:
        net = CopySummGat(**net_args)

    tokenizer, loader = build_loaders(bert_model)

    net.load_state_dict(ckpt['state_dict'])
    if cuda:
        net.cuda()

    abstractor = BeamAbstractor(tokenizer, cuda, min_len, max_len, max_art,
                                700, abs_args['nograph_channel'], abs_args['is_bipartite'])

    return net, abstractor, loader, tokenizer


def build_loaders(bert_model):

    assert bert_model in ['roberta-base', 'scibert']

    args.bert_model = bert_model

    if bert_model == 'scibert':
        tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_cased')
        tokenizer.encoder = tokenizer.vocab
        tokenizer.bos_token = '[unused1]'
        tokenizer.eos_token = '[unused2]'
    elif bert_model == 'roberta-base':
        tokenizer = RobertaTokenizer.from_pretrained(bert_model)

    # coll_fn is needed to filter out too short abstracts (<100) and articles (<300)
    test_loader = DataLoader(
        csvDataset('test', args.data_dir), batch_size=args.batch_size,
        shuffle=True,
        num_workers=4 if args.cuda else 0,
        collate_fn=coll_fn
    )

    return tokenizer, test_loader


def decode(cuda, min_len, max_len, max_art, model_dir):
    net, abstractor, loader, tokenizer = configure_decoding(model_dir, cuda, min_len, max_len, max_art)

    with torch.no_grad():
        tracker = EmissionsTracker(measure_power_secs=100000, save_to_file=True)
        tracker.start()
        start = timer()
        for batch in tqdm(loader):
            doc_ids = [doc[2] for doc in batch]
            # for each document in the batch we have a list of all the beam_size
            # hypothesis returned by beam search
            all_beams = abstractor(net, batch)
            beam_inds = []
            for j in range(len(all_beams)):
                beam_inds += [(len(beam_inds), 1)]

            # For each document we have the list of tokens of the best beam
            dec_outs = rerank_mp(all_beams, beam_inds)

            decoded_sents = []
            for dec_out in dec_outs:

                if args.bert_model == 'roberta-base':
                    text = ''.join(dec_out)
                    # decoded list of tokens (replaced 'Ġ' with ' ')
                    dec_out = bytearray([tokenizer.byte_decoder[c] for c in text]).decode('utf-8', errors=tokenizer.errors)

                elif args.bert_model == 'scibert':
                    dec_out = ' '.join(dec_out).replace(' ##', '')

                # each sentence is splitted using ' .'
                dec_out = sent_tokenize(dec_out)
                decoded_sents.append(dec_out)

            for i, dec_sent in enumerate(decoded_sents):

                with open(join(args.model_dir, 'output/{}.dec'.format(doc_ids[i])),
                          'w') as f:
                    f.write('\n'.join(dec_sent))

        end = timer()
        emissions = tracker.stop()
        print('Total CO2 emission (in Kg): {}'.format(emissions))
        print(f'Inference took: {end - start} seconds')


_PRUNE = defaultdict(
    lambda: 2,
    {1: 5, 2: 5, 3: 5, 4: 5, 5: 5, 6: 4, 7: 3, 8: 3}
)


def rerank_mp(all_beams, ext_inds):
    beam_lists = [all_beams[i: i + n] for i, n in ext_inds if n > 0]
    with mp.Pool(8) as pool:
        reranked = pool.map(rerank_one, beam_lists)
    return list(concat(reranked))


def rerank_one(beams):
    @curry
    def process_beam(beam, n):
        for b in beam[:n]:
            b.gram_cnt = Counter(_make_n_gram(b.sequence))
        return beam[:n]

    beams = map(process_beam(n=_PRUNE[len(beams)]), beams)
    best_hyps = max(product(*beams), key=_compute_score)
    dec_outs = [h.sequence for h in best_hyps]
    return dec_outs


def _make_n_gram(sequence, n=2):
    return (tuple(sequence[i:i + n]) for i in range(len(sequence) - (n - 1)))


def length_wu(cur_len, alpha=0.):
    """GNMT length re-ranking score.
    See "Google's Neural Machine Translation System" :cite:`wu2016google`.
    """
    return ((5 + cur_len) / 6.0) ** alpha


def make_html_safe(s):
    """Rouge use html, has to make output html safe"""
    return s.replace("<", "&lt;").replace(">", "&gt;")


def _compute_score(hyps):
    # all_cnt = reduce(op.iadd, (h.gram_cnt for h in hyps), Counter())
    # # repeat = sum(c-1 for g, c in all_cnt.items() if c > 1)
    # lp = sum(h.logprob for h in hyps) / sum(len(h.sequence) for h in hyps)
    try:
        lp = sum(h.logprob for h in hyps) / sum(length_wu(len(h.sequence) + 1, alpha=0.9) for h in hyps)
    except ZeroDivisionError:
        lp = -1e5
    return lp


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', help='root of the abstractor model', required=True)
    parser.add_argument('--data_dir', action='store', default='CDSR_data',
                        help='directory where the data is stored')

    # decode options
    parser.add_argument('--max_input', type=int, default=10000, help='maximum input length')
    parser.add_argument('--batch_size', type=int, action='store', default=16,
                        help='batch size of faster decoding')
    parser.add_argument('--max_dec_word', type=int, action='store', default=700,
                        help='maximum words to be decoded for the abstractor')
    parser.add_argument('--min_dec_word', type=int, action='store', default=0,
                        help='minimum words to be decoded for the abstractor')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--no-cuda', action='store_true',
                        help='disable GPU training')
    parser.add_argument('--dem_model', action='store', default='mlee',
                        help='DeepEventMine pre-trained model name')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        torch.cuda.set_device(args.gpu_id)

    output_dir = join(args.model_dir, 'output')
    if exists(join(output_dir)):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir)

    decode(args.cuda, args.min_dec_word, args.max_dec_word, args.max_input, args.model_dir)
