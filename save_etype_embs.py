import torch
import pickle
import argparse

from os import listdir


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dem_model = args.dmodel
    weights_path = f'deep_event_mine/data/models/{dem_model}/model/' + \
                   listdir(f'deep_event_mine/data/models/{dem_model}/model/')[0]

    DEM_weights = torch.load(weights_path, map_location=device)
    params = pickle.load(open(f'deep_event_mine/data/models/{dem_model}/{dem_model}.param', 'rb'))
    etype_mapping = {v: k for k, v in params['mappings']['rev_type_map'].items()}
    etype_mapping['Other'] = len(etype_mapping)
    rel_mapping = {v: k for k, v in params['mappings']['rev_rtype_map'].items()}

    etype_dict = {'embeddings': DEM_weights['model']['REL_layer.type_embed.weight'],
                  'mapping': etype_mapping, 'rel_embeddings': DEM_weights['model']['EV_layer.rtype_layer.weight'], 'rel_mapping': rel_mapping}

    pickle.dump(etype_dict, open(f'deep_event_mine/type_embs/etype_dict_{dem_model}.pkl', 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='training of the abstractor (ML)'
    )
    parser.add_argument('--dmodel', action='store', default='mlee',
                        help='DeepEventMine pre-trained model name')
    args = parser.parse_args()

    main(args)
