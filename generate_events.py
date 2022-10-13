import argparse

import torch
import pandas as pd
from tqdm import tqdm

from torch.utils.data import DataLoader, Dataset

from deep_event_mine.loader.prepData import prepdata
from deep_event_mine import configdem
from deep_event_mine.eval.evalEV import write_events_wpos
from deep_event_mine.utils import utils
from deep_event_mine.nets import deepEM
from deep_event_mine.bert.tokenization import BertTokenizer


class DocsDataset(Dataset):
    def __init__(self, split: str):
        self._dset = split
        self._data_path = f'{args.data_dir}/{self._dset}.csv'
        self._data_df = pd.read_csv(self._data_path)

    def __len__(self):
        return len(self._data_df)

    def __getitem__(self, i):
        source = self._data_df.loc[i, 'source']
        article_id = self._data_df.loc[i, 'article_id']

        return source, article_id


def coll_fn(data):

    batch = list(data)
    return batch


def main(args):
    cuda = True if torch.cuda.is_available() else False
    dset = args.dset
    dmodel = args.dmodel

    d_loader = DataLoader(
        DocsDataset(dset), batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=4 if cuda else 0,
        collate_fn=coll_fn
    )

    pred_params, params = configdem.config(f'{dmodel}_{dset}.yaml')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    deepee_model = deepEM.DeepEM(params)
    deepee_model.to(device)
    utils.handle_checkpoints(model=deepee_model,
                             checkpoint_dir=params['model_path'],
                             params={
                                 'device': device
                             },
                             resume=True)

    # nner: Using subwords:
    tokenizer = BertTokenizer.from_pretrained(
        params['bert_model'], do_lower_case=False
    )

    with torch.no_grad():

        for batch in tqdm(d_loader):
            all_sentences = {}

            for article, article_id in batch:

                article = article.replace('e.g.', 'e.g,').replace('i.e.', 'i.e,')
                sentences = article.split('. ')
                all_sentences[article_id] = sentences

            _data = prepdata.prep_input_data(None, params, sentences0=all_sentences)
            nn_data, _dataloader = configdem.read_test_data(_data, params, tokenizer)
            nn_data['g_entity_ids_'] = _data['g_entity_ids_']

            generate_events(nn_data, _dataloader, deepee_model, params)


def generate_events(nn_data, _dataloader, deepee_model, params):

    result_dir = f'deep_event_mine/{args.data_dir}_a2_files_{args.dmodel}/{args.dset}/'
    # store predicted entities
    ent_preds = []

    # store predicted events
    ev_preds = []

    fidss, wordss, offsetss, sub_to_wordss, span_indicess = [], [], [], [], []
    all_ent_embs, all_ner_preds, all_ner_terms = [], [], []

    # entity and relation output
    ent_anns = []

    mapping_id_tag = params['mappings']['nn_mapping']['id_tag_mapping']

    is_eval_ev = False

    for batch in _dataloader:

        eval_data_ids = batch
        tensors = utils.get_tensors(eval_data_ids, nn_data, params)

        nn_tokens, nn_ids, nn_token_mask, nn_attention_mask, nn_span_indices, nn_span_labels, nn_span_labels_match_rel, nn_entity_masks, nn_trigger_masks, _, \
        etypes, _ = tensors

        fids = [
            nn_data["fids"][data_id] for data_id in eval_data_ids[0].tolist()
        ]
        offsets = [
            nn_data["offsets"][data_id]
            for data_id in eval_data_ids[0].tolist()
        ]
        words = [
            nn_data["words"][data_id] for data_id in eval_data_ids[0].tolist()
        ]
        sub_to_words = [
            nn_data["sub_to_words"][data_id]
            for data_id in eval_data_ids[0].tolist()
        ]
        subwords = [
            nn_data["subwords"][data_id]
            for data_id in eval_data_ids[0].tolist()
        ]

        ner_out, rel_out, ev_out = deepee_model(tensors, params)

        ner_preds = ner_out['preds']

        ner_terms = ner_out['terms']

        all_ner_terms.append(ner_terms)

        for sentence_idx, ner_pred in enumerate(ner_preds):

            pred_entities = []
            for span_id, ner_pred_id in enumerate(ner_pred):
                span_start, span_end = nn_span_indices[sentence_idx][span_id]
                span_start, span_end = span_start.item(), span_end.item()
                if (ner_pred_id > 0
                        and span_start in sub_to_words[sentence_idx]
                        and span_end in sub_to_words[sentence_idx]
                ):
                    pred_entities.append(
                        (
                            sub_to_words[sentence_idx][span_start],
                            sub_to_words[sentence_idx][span_end],
                            mapping_id_tag[ner_pred_id],
                        )
                    )
            all_ner_preds.append(pred_entities)

        # entity prediction
        ent_ann = {'span_indices': nn_span_indices, 'ner_preds': ner_out['preds'], 'words': words,
                   'offsets': offsets, 'sub_to_words': sub_to_words, 'subwords': subwords,
                   'ner_terms': ner_terms}
        ent_anns.append(ent_ann)

        fidss.append(fids)

        wordss.append(words)
        offsetss.append(offsets)
        sub_to_wordss.append(sub_to_words)

        # relation prediction
        if rel_out['next']:
            all_ent_embs.append(rel_out['enttoks_type_embeds'])
        else:
            all_ent_embs.append([])

        # event prediction
        if ev_out is not None:
            # add predicted entity
            ent_preds.append(ner_out["nner_preds"])

            # add predicted events
            ev_preds.append(ev_out)

            span_indicess.append(
                [
                    indice.detach().cpu().numpy()
                    for indice in ner_out["span_indices"]
                ]
            )
            is_eval_ev = True
        else:
            ent_preds.append([])
            ev_preds.append([])

            span_indicess.append([])

    if is_eval_ev > 0:
        write_events_wpos(fids=fidss,
                          all_ent_preds=ent_preds,
                          all_words=wordss,
                          all_offsets=offsetss,
                          all_span_terms=all_ner_terms,
                          all_span_indices=span_indicess,
                          all_sub_to_words=sub_to_wordss,
                          all_ev_preds=ev_preds,
                          g_entity_ids_=nn_data['g_entity_ids_'],
                          params=params,
                          result_dir=result_dir
                          )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='training of the abstractor (ML)'
    )
    parser.add_argument('--dset', required=True, help='target set for generating events')
    parser.add_argument('--dmodel', help='DeepEventMine pre-trained model')
    parser.add_argument('--data_dir', help='Name of the directory where Dataset is stored')
    parser.add_argument('--batch_size', default=5, help='number of documents to pass to DeepEventMine')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    if args.cuda:
        torch.cuda.set_device(args.gpu_id)
    main(args)
