import torch
import pickle
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from deep_event_mine.utils import utils
from deep_event_mine.loader.prepNN import prep4nn


def read_test_data(test_data, params, tokenizer):
    test = prep4nn.data2network(test_data, 'predict', params, tokenizer)

    if len(test) == 0:
        raise ValueError("Test set empty.")

    test_data = prep4nn.torch_data_2_network(cdata2network=test, params=params, do_get_nn_data=True, tokenizer=tokenizer)

    # number of sentences
    te_data_size = len(test_data['nn_data']['ids'])

    test_data_ids = TensorDataset(torch.arange(te_data_size))
    test_sampler = SequentialSampler(test_data_ids)
    test_dataloader = DataLoader(test_data_ids, sampler=test_sampler, batch_size=params['batchsize'])
    return test_data, test_dataloader


def config(config_file):
    config_path = 'deep_event_mine/configs/{}'.format(config_file)

    with open(config_path, 'r') as stream:
        pred_params = utils._ordered_load(stream)

    # Load pre-trained parameters
    with open(pred_params['saved_params'], "rb") as f:
        parameters = pickle.load(f)

    # build l2r_pairs
    parameters['predict'] = True

    # Set predict settings value for params
    parameters['batchsize'] = pred_params['batchsize']
    device = torch.device("cuda:" + str(pred_params['gpu']) if torch.cuda.is_available() else "cpu")

    parameters['device'] = device
    parameters['compute_metrics'] = pred_params['compute_metrics']
    parameters['bert_model'] = pred_params['bert_model']
    parameters['bert_vocab'] = pred_params['bert_vocab']
    parameters['model_path'] = pred_params['model_path']
    parameters['raw_text'] = pred_params['raw_text']
    parameters['ner_predict_all'] = pred_params['ner_predict_all']
    parameters['compute_dem_loss'] = pred_params['compute_dem_loss']
    parameters['a2_entities'] = pred_params['a2_entities']

    return pred_params, parameters,

