""" Evaluate the baselines ont ROUGE/METEOR"""
import argparse
from os.path import join, exists
import pandas as pd
import os
import json
import shutil
from nltk import sent_tokenize

from evaluate import eval_rouge, eval_blue, eval_bertsc, eval_read, eval_novel_ngrams


def filter_byevs(dec_dir):

    articles = os.listdir(dec_dir)
    articles = [article for article in articles if article[-4:] == '.dec']
    cdsr_df = pd.read_csv('CDSR_DATA/test.csv')
    files = os.listdir('deep_event_mine/a2_files_{}/test/ev-tok-a2'.format(args.dem_model))
    number_evs = []
    numb_sents = []
    for article in articles:
        article = article.replace('.dec', '')
        article_a2 = article + '.a2'
        if article_a2 in files:
            with open('deep_event_mine/a2_files_{}/test/ev-tok-a2/{}'.format(args.dem_model, article_a2), 'r') as f:
                lines = f.readlines()
                evs = [line for line in lines if line.startswith('E')]
                number_evs.append(len(evs))
        else:
            number_evs.append(0)

        article_text = cdsr_df.loc[cdsr_df['article_id'] == article]['source'].values[0]
        numb_sents.append(len(sent_tokenize(article_text)))

    prop_evs = [number_evs[i] / numb_sents[i] for i in range(len(number_evs))]

    files_evs = []
    for i, numb in enumerate(prop_evs):
        if numb > float(args.evs_filter):
            files_evs.append(articles[i].replace('.dec', ''))

    new_dir = dec_dir + '_moregraphs'
    if exists(new_dir):
        shutil.rmtree(new_dir)

    os.makedirs(new_dir)

    for fg in articles:
        if fg.replace('.dec', '') in files_evs:
            shutil.copy(join(dec_dir, fg), new_dir)

    return new_dir


def main():

    dec_dir = join(args.decode_dir, args.output_name)
    ref_dir = join(args.data_dir, 'refs')
    assert args.metric in ['rouge', 'blue', 'readability', 'novel_ngram', 'bert_score']

    if args.evs_filter is not None:
        dec_dir = filter_byevs(dec_dir)

    if args.metric == 'rouge':

        if args.data_dir == 'CDSR_data':
            dec_pattern = r'CD(\d+).dec'
            ref_pattern = 'CD#ID#.ref'
        elif args.data_dir == 'CG_data':
            dec_pattern = r'PMID-(\d+).dec'
            ref_pattern = 'PMID-#ID#.ref'

        output = eval_rouge(dec_pattern, dec_dir, ref_pattern, ref_dir)

        print(output)
        with open(join(args.decode_dir, '{}.txt'.format(args.metric)), 'w') as f:
            f.write(output)

    elif args.metric == 'blue':
        output = eval_blue(dec_dir, ref_dir)

        print(output)
        with open(join(args.decode_dir, '{}.txt'.format(args.metric)), 'w') as f:
            f.write(output)

    elif args.metric == 'readability':
        output = eval_read(dec_dir)
        print(output)
        with open(join(args.decode_dir, '{}.txt'.format(args.metric)), 'w') as f:
            f.write(output)

    elif args.metric == 'novel_ngram':

        output = eval_novel_ngrams(dec_dir)

        print(output)
        with open(join(args.decode_dir, '{}.txt'.format(args.metric)), 'w') as f:
            f.write(output)

    elif args.metric == 'bert_score':
        output = eval_bertsc(dec_dir, ref_dir)
        print(output)
        with open(join(args.decode_dir, '{}.txt'.format(args.metric)), 'w') as f:
            f.write(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate the output files for the RL full models')

    parser.add_argument('--decode_dir', action='store', required=True,
                        help='directory of decoded summaries')
    parser.add_argument('--metric', action='store', default='rouge',
                        help='metric for evaluating summary')
    parser.add_argument('--data_dir', action='store', default='CDSR_data',
                        help='folder where reference data is stored')
    parser.add_argument('--evs_filter', action='store', default=None,
                        help='Consider only articles with a proportion of events'
                             'larger than evs_filter')
    parser.add_argument('--output_name', action='store', default='output',
                        help='folder inside decode_dir where the summaries are stored')
    parser.add_argument('--dem_model', action='store', default='mlee',
                        help='DeepEventMine model used for extracting events')

    args = parser.parse_args()
    main()
