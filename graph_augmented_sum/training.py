""" module providing basic training utilities"""
#from coherence_interface.coherence_inference import coherence_infer, batch_global_infer
from os.path import join
from time import time
from datetime import timedelta
from itertools import starmap

from cytoolz import curry, reduce

import torch
from torch.nn.utils import clip_grad_norm_

import logging
logging.getLogger('transformers.tokenization_utils').setLevel(logging.ERROR)
logging.getLogger('transformers.tokenization_utils').disabled = True
logging.basicConfig(level=logging.ERROR)


def get_basic_grad_fn(net, clip_grad, max_grad=1e2):
    def f():
        grad_norm = clip_grad_norm_(
            [p for p in net.parameters() if p.requires_grad], clip_grad)
        try:
            grad_norm = grad_norm.item()
        except AttributeError:
            grad_norm = grad_norm
        if max_grad is not None and grad_norm >= max_grad:
            print('WARNING: Exploding Gradients {:.2f}'.format(grad_norm))
            grad_norm = max_grad
        grad_log = {}
        grad_log['grad_norm'] = grad_norm
        return grad_log
    return f

def get_loss_args(net_out, bw_args):
    if isinstance(net_out, tuple):
        loss_args = net_out + bw_args
    else:
        loss_args = (net_out, ) + bw_args
    return loss_args

@curry
def compute_loss(net, fw_args):
    loss = net(*fw_args)
    return loss

@curry
def val_step(loss_step, fw_args):
    loss = loss_step(fw_args)
    try:
        n_data = loss.size(0)
        return n_data, loss.sum().item()
    except:
        n_data = 1
        return n_data, loss.item()


@curry
def basic_validate(net, val_batches):
    print('running validation ... ', end='')
    net.eval()
    start = time()
    with torch.no_grad():
        validate_fn = val_step(compute_loss(net))
        n_data, tot_loss = reduce(
            lambda a, b: (a[0]+b[0], a[1]+b[1]),
            starmap(validate_fn, val_batches),
            (0, 0)
        )
    val_loss = tot_loss / n_data
    print(
        'validation finished in {}                                    '.format(
            timedelta(seconds=int(time()-start)))
    )
    print('validation loss: {:.4f} ... '.format(val_loss))
    return {'loss': val_loss}


class BasicPipeline(object):
    def __init__(self, name, net,
                 train_batcher, val_batcher, batch_size,
                 val_fn, criterion, optim, grad_fn=None):
        self.name = name
        self._net = net
        self._train_batcher = train_batcher
        self._val_batcher = val_batcher
        self._criterion = criterion
        self._opt = optim
        # grad_fn is calleble without input args that modifyies gradient
        # it should return a dictionary of logging values
        self._grad_fn = grad_fn
        self._val_fn = val_fn

        self._n_epoch = 0  # epoch not very useful?
        self._batch_size = batch_size
        self._batches = self.batches()

    def batches(self):
        while True:
            for fw_args in self._train_batcher(self._batch_size):
                yield fw_args
            self._n_epoch += 1

    def get_loss_args(self, net_out, bw_args):
        if isinstance(net_out, tuple):
            loss_args = net_out + bw_args
        else:
            loss_args = (net_out, ) + bw_args
        return loss_args

    def train_step(self):

        self._net.train()

        fw_args = next(self._batches)

        loss = self._net(*fw_args)

        loss.backward()

        log_dict = {}
        log_dict['loss'] = loss.item()

        if self._grad_fn is not None:
            log_dict.update(self._grad_fn())
        self._opt.step()
        self._net.zero_grad()

        return log_dict

    def validate(self):
        return self._val_fn(self._val_batcher(self._batch_size))

    def checkpoint(self, save_path, step, val_metric=None):
        save_dict = {}
        if val_metric is not None:
            name = 'ckpt-{:6f}-{}'.format(val_metric, step)
            save_dict['val_metric'] = val_metric
        else:
            name = 'ckpt-{}'.format(step)

        save_dict['state_dict'] = self._net.state_dict()
        save_dict['optimizer'] = self._opt.state_dict()
        torch.save(save_dict, join(save_path, name))

    def terminate(self):
        self._train_batcher.terminate()
        self._val_batcher.terminate()


