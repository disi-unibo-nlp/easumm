""" evaluation scripts"""
import os
from os.path import join
import logging
import tempfile
import subprocess as sp

from nltk.translate.bleu_score import corpus_bleu
from bert_score import score
from pyrouge import Rouge155
from pyrouge.utils import log
from readability import Readability
from nltk import ngrams
import pandas as pd
import torch


def eval_rouge(dec_pattern, dec_dir, ref_pattern, ref_dir,
               cmd='-c 95 -r 1000 -n 2 -m', system_id=1):
    """ evaluate by original Perl implementation"""

    _ROUGE_PATH = 'ROUGE'

    assert _ROUGE_PATH is not None
    log.get_global_console_logger().setLevel(logging.WARNING)
    with tempfile.TemporaryDirectory() as tmp_dir:
        Rouge155.convert_summaries_to_rouge_format(
            dec_dir, join(tmp_dir, 'dec'))
        Rouge155.convert_summaries_to_rouge_format(
            ref_dir, join(tmp_dir, 'ref'))
        Rouge155.write_config_static(
            join(tmp_dir, 'dec'), dec_pattern,
            join(tmp_dir, 'ref'), ref_pattern,
            join(tmp_dir, 'settings.xml'), system_id
        )
        cmd = (join(_ROUGE_PATH, 'ROUGE-1.5.5.pl')
               + ' -e {} '.format(join(_ROUGE_PATH, 'data'))
               + cmd
               + ' -a {}'.format(join(tmp_dir, 'settings.xml')))
        output = sp.check_output(cmd.split(' '), universal_newlines=True)
    return output


def eval_blue(dec_dir, ref_dir):
    articles = os.listdir(dec_dir)
    articles = [article.replace('.dec', '') for article in articles]

    targets = []
    predictions = []
    for article in articles:
        with open('{}/{}.dec'.format(dec_dir, article)) as dec:
            dec_text = dec.read().replace('\n', ' ')
        with open('{}/{}.ref'.format(ref_dir, article)) as ref:
            or_text = ref.read().replace('\n', ' ')

        targets.append([or_text.split()])
        predictions.append(dec_text.split())

    output = '------------------------\n'
    output += 'BLUE-1: {}\n'.format(corpus_bleu(targets, predictions, weights=(1.0, 0, 0, 0)))
    output += 'BLUE-2: {}\n'.format(corpus_bleu(targets, predictions, weights=(0.5, 0.5, 0, 0)))
    output += 'BLUE-3: {}\n'.format(corpus_bleu(targets, predictions, weights=(0.3, 0.3, 0.3, 0)))
    output += 'BLUE-4: {}\n'.format(corpus_bleu(targets, predictions, weights=(0.25, 0.25, 0.25, 0.25)))
    output += '------------------------'

    return output


def eval_bertsc(dec_dir, ref_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    articles = os.listdir(dec_dir)
    articles = [article.replace('.dec', '') for article in articles]

    targets = []
    predictions = []
    for article in articles:
        with open('{}/{}.dec'.format(dec_dir, article)) as dec:
            dec_text = dec.read().replace('\n', ' ')
        with open('{}/{}.ref'.format(ref_dir, article)) as ref:
            or_text = ref.read().replace('\n', ' ')

        targets.append(or_text)
        predictions.append(dec_text)

    P, R, F1 = score(predictions, targets, lang="en", verbose=True, device=device)
    output = '------------------------\n'
    output += f'BERTScore Precision: {P.mean()*100:.2f}\n'
    output += f'BERTScore Recall: {R.mean()*100:.2f}\n'
    output += f'BERTScore F1: {F1.mean()*100:.2f}\n'
    output += '------------------------'

    return output


def eval_read(dec_dir):
    articles = os.listdir(dec_dir)

    full_text = ''
    for article in articles:
        if not article.startswith('.'):
            with open('{}/{}'.format(dec_dir, article)) as dec:
                dec_text = dec.read().replace('\n', ' ')
                full_text += dec_text

    r = Readability(full_text)

    output = '------------------------\n'
    output += f'Flesch-Kincaid grade level: {r.flesch_kincaid().score:.2f}\n'
    output += f'Gunning fog index: {r.gunning_fog().score:.2f}\n'
    output += f'Coleman-Liau index: {r.coleman_liau().score:.2f}\n'
    output += f'Average readability metrics score: {((r.flesch_kincaid().score + r.gunning_fog().score + r.coleman_liau().score)/3):.2f}\n'
    output += '------------------------'

    return output


def eval_novel_ngrams(dec_dir):
    doc_df = pd.read_csv('CDSR_data/test.csv')
    articles = list(doc_df['article_id'])
    source_docs = list(doc_df['source'])

    predictions = []
    for article in articles:
        with open('{}/{}.dec'.format(dec_dir, article)) as dec:
            dec_text = dec.read().replace('\n', ' ')
        predictions.append(dec_text)

    onegrams = __compute_abstractness(source_docs, predictions, 1)
    bigrams = __compute_abstractness(source_docs, predictions, 2)
    trigrams = __compute_abstractness(source_docs, predictions, 3)
    fourgrams = __compute_abstractness(source_docs, predictions, 4)

    output = '------------------------\n'
    output += f'Proportion of novel 1-gram: {onegrams*100:.2f}\n'
    output += f'Proportion of novel 2-gram: {bigrams*100:.2f}\n'
    output += f'Proportion of novel 3-gram: {trigrams*100:.2f}\n'
    output += f'Proportion of novel 4-gram: {fourgrams*100:.2f}\n'
    output += f'Average proportion of novel n-grams: {((onegrams + bigrams + trigrams + fourgrams)/4)*100:.2f}\n'
    output += '------------------------'

    return output


def __compute_abstractness(res_references, res_predictions, n):
    total_match = 0
    n_words = 0
    for reference, candidate in zip(res_references, res_predictions):
        match = 0
        monograms = candidate.split(" ")
        n_words = n_words + len(monograms)  # count all words in test set
        if n > len(monograms):
            return "Not possible to create " + str(n) + "-grams, too many few words"
        for w2 in ngrams(monograms, n):
            substr = " ".join(w2)
            if substr not in reference:
                match = match + 1
        # n_words=n_words+1 #counter for total n-gram number
        total_match = total_match + match
    return total_match / n_words







