""" train the abstractor"""
import argparse
import json
import os

from codecarbon import EmissionsTracker

import wandb
from os.path import join, exists
import pickle as pkl

from cytoolz import compose
from transformers import RobertaTokenizer, AutoTokenizer
from tqdm import tqdm

import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from graph_augmented_sum.model.copy_summ_multiencoder import CopySummGat
from graph_augmented_sum.model.copy_summ import CopySumm
from graph_augmented_sum.utils import load_latest_ckpt
from graph_augmented_sum.training import get_basic_grad_fn
from graph_augmented_sum.data.batcher import coll_fn
from graph_augmented_sum.data.data import csvDataset
from graph_augmented_sum.data.batcher import prepro_fn_gat_bert, convert_batch_gat_bert, batchify_fn_gat_bert, \
    batchify_fn_copy_bert, convert_batch_bert, prepro_fn_bert


def configure_net(vocab_size, emb_dim, n_hidden, bidirectional,
                  n_layer, max_art, dropout, no_etype, is_bipartite, dem_model, nograph_channel, bert_model, rgat):
    net_args = {}
    net_args['vocab_size'] = vocab_size
    net_args['lstm_dim'] = emb_dim
    net_args['n_hidden'] = n_hidden
    net_args['bidirectional'] = bidirectional
    net_args['n_layer'] = n_layer
    net_args['bert_model'] = bert_model
    net_args['dropout'] = dropout
    net_args['bert_length'] = max_art
    net_args['is_bipartite'] = is_bipartite

    if nograph_channel:
        net = CopySumm(**net_args)
    else:
        if no_etype:
            net_args['etype_path'] = ''
        else:
            net_args['etype_path'] = f'deep_event_mine/type_embs/etype_dict_{dem_model}.pkl'

        net_args['side_dim'] = n_hidden
        net = CopySummGat(**net_args)

    if args.cuda:
        net.cuda()

    return net, net_args


def build_loaders(bert_model):
    assert bert_model in ['roberta-base', 'scibert']

    if bert_model == 'scibert':
        tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_cased')
        tokenizer.encoder = tokenizer.vocab
        tokenizer.bos_token = '[unused1]'
        tokenizer.eos_token = '[unused2]'
    elif bert_model == 'roberta-base':
        tokenizer = RobertaTokenizer.from_pretrained(bert_model)

    cuda = True if torch.cuda.is_available() else False

    # coll_fn is needed to filter out too short abstracts (<100) and articles (<300)
    train_loader = DataLoader(
        csvDataset('train', args.data_dir), batch_size=args.batch_size,
        shuffle=True,
        num_workers=4 if cuda else 0,
        collate_fn=coll_fn
    )

    val_loader = DataLoader(
        csvDataset('val', args.data_dir), batch_size=args.val_batch_size,
        shuffle=False, num_workers=4 if cuda else 0,
        collate_fn=coll_fn
    )

    return train_loader, val_loader, tokenizer


class BasicTrainer:
    """ Basic trainer with minimal function and early stopping"""

    def __init__(self, ckpt_freq, patience, scheduler, cuda, grad_fn, tokenizer, save_dir, stats):

        self._ckpt_freq = ckpt_freq
        self._patience = patience
        self._save_dir = save_dir

        self._scheduler = scheduler
        self._total_loss = 0
        self._step = 0

        if stats is None:
            self._epoch = 0
            self._best_val = None
            self._current_p = 0
        else:
            self._epoch = stats['epoch']
            self._best_val = stats['best_val']
            self._current_p = stats['current_p']

        self._grad_fn = grad_fn

        if args.nograph_channel:
            self._batchify_train = self._batchify_val = compose(
                batchify_fn_copy_bert(tokenizer, cuda=cuda),
                convert_batch_bert(tokenizer, args.max_art),
                prepro_fn_bert(tokenizer, args.max_art, args.max_abs)
            )

        else:
            self._batchify_train = compose(
                batchify_fn_gat_bert(tokenizer, cuda=cuda, is_bipartite=args.is_bipartite),
                convert_batch_gat_bert(tokenizer, args.max_art),
                prepro_fn_gat_bert(tokenizer, args.max_art, args.max_abs, 'train', args.dem_model,
                                   is_bipartite=args.is_bipartite)
            )
            self._batchify_val = compose(
                batchify_fn_gat_bert(tokenizer, cuda=cuda, is_bipartite=args.is_bipartite),
                convert_batch_gat_bert(tokenizer, args.max_art),
                prepro_fn_gat_bert(tokenizer, args.max_art, args.max_abs, 'val', args.dem_model,
                                   is_bipartite=args.is_bipartite)
            )

    def train(self, net, train_loader, val_loader, optimizer):

        total_emissions = 0
        while True:
            tracker = EmissionsTracker(measure_power_secs=100000, save_to_file=True)
            tracker.start()
            wandb.watch(net)
            net.train()
            self._epoch += 1

            pgb = tqdm(train_loader, leave=False)
            for batch in pgb:

                self._step += 1

                fw_args = self._batchify_train(batch)
                loss = net(*fw_args)
                self._total_loss += loss.item()
                pgb.set_postfix({'loss': loss.item(),
                                 'epochs': self._epoch})

                loss.backward()

                log_dict = {'loss': loss.item()}

                if self._grad_fn is not None:
                    log_dict.update(self._grad_fn())

                optimizer.step()
                net.zero_grad()

                if self._step % self._ckpt_freq == 0:

                    stop = self.checkpoint(val_loader, net, optimizer)
                    if stop:
                        break

                    net.train()
            else:
                emissions = tracker.stop()
                total_emissions += emissions
                wandb.log({'CO2 emission (in Kg)': emissions})
                wandb.log({'Total CO2 emission (in Kg)': total_emissions})
                continue
            break

    def checkpoint(self, val_loader, net, optimizer):
        # compute loss on validation set
        train_loss = self._total_loss / self._ckpt_freq
        print('average train loss at step {}: {:.4f}'.format(self._step, train_loss))
        val_metric = self.validate(val_loader, net)

        wandb.log({
            'training_loss': train_loss,
            'validation_loss': val_metric,
            'epoch': self._epoch,
            'step': self._step,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'current_p': self._current_p
        })

        self._scheduler.step(val_metric)

        # check if the number of times in a row that we don't experience an improvement
        # is greater than patience e.g. 5, if that's the case we interrupt training
        stop = self.check_stop(val_metric, net, optimizer)

        self._total_loss = 0

        return stop

    def validate(self, val_loader, net):

        print('running validation')
        net.eval()
        tot_loss = 0
        n_data = len(val_loader)

        with torch.no_grad():
            for batch in tqdm(val_loader, leave=False):
                fw_args = self._batchify_val(batch)

                tot_loss += net(*fw_args).item()

            val_loss = tot_loss / n_data
            print('validation loss for epoch {}, step {}: {:.4f}'.format(self._epoch, self._step, val_loss))

        return val_loss

    def check_stop(self, val_metric, net, optimizer):
        if self._best_val is None:
            self._best_val = val_metric
            # save model weights and optimizer
            save_dict = {}
            save_path = join(self._save_dir, 'ckpt')
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            name = 'best_ckpt-{}'.format(self._epoch)
            save_dict['state_dict'] = net.state_dict()
            save_dict['optimizer_state_dict'] = optimizer.state_dict()
            save_dict['stats'] = {'epoch': self._epoch,
                                  'current_p': self._current_p,
                                  'best_val': self._best_val,
                                  'val_metric': val_metric,
                                  'step': self._step
                                  }
            torch.save(save_dict, join(save_path, name))

        elif val_metric + args.sched_thr < self._best_val:
            self._current_p = 0
            self._best_val = val_metric

            print('Saving new best model')
            # save model weights and optimizer
            save_dict = {}
            save_path = join(self._save_dir, 'ckpt')
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            name = 'best_ckpt-{}'.format(self._epoch)
            save_dict['state_dict'] = net.state_dict()
            save_dict['optimizer_state_dict'] = optimizer.state_dict()
            save_dict['stats'] = {'epoch': self._epoch,
                                  'current_p': self._current_p,
                                  'best_val': self._best_val,
                                  'val_metric': val_metric,
                                  'step': self._step
                                  }
            torch.save(save_dict, join(save_path, name))

        else:
            self._current_p += 1

        return self._current_p >= self._patience


def main(args):
    if not args.wandb_log:
        os.environ["WANDB_DISABLED"] = "true"
    else:
        wandb.init(entity=args.wandb_entity)

    train_loader, val_loader, tokenizer = build_loaders(args.bert_model)

    net, net_args = configure_net(len(tokenizer.encoder), args.emb_dim,
                                  args.n_hidden, args.bi, args.n_layer,
                                  args.max_art, args.dropout, args.no_etype, args.is_bipartite,
                                  args.dem_model, args.nograph_channel, args.bert_model, args.rgat)

    optimizer = optim.AdamW(net.parameters(), lr=args.lr)

    if args.pretrained_ckpt is not None:
        ckpt = load_latest_ckpt(args.pretrained_ckpt)
        net.load_state_dict(ckpt['state_dict'], strict=False)

    if args.resume:
        ckpt = load_latest_ckpt(args.ckpt_dir)
        net.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if args.force_lr:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
        stats = ckpt['stats']
    else:
        stats = None

    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True,
                                  factor=args.decay, min_lr=0, threshold=args.sched_thr,
                                  patience=args.lr_p)

    grad_fn = get_basic_grad_fn(net, args.clip)

    trainer = BasicTrainer(args.ckpt_freq, args.patience, scheduler, args.cuda, grad_fn, tokenizer, args.ckpt_dir,
                           stats)

    # save experiment setting
    if not exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    with open(join(args.ckpt_dir, 'vocab.pkl'), 'wb') as f:
        pkl.dump(tokenizer.encoder, f, pkl.HIGHEST_PROTOCOL)

    meta = {'clip_grad_norm': args.clip, 'batch_size': args.batch_size, 'val_batch_size': args.val_batch_size,
            'lr_decay': args.decay, 'nograph_channel': args.nograph_channel, 'net_args': net_args}

    with open(join(args.ckpt_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=4)

    wandb.init(project="EventAugmentedSumm", config=meta)

    print('')
    print('start training with the following hyper-parameters:')
    print(meta)
    trainer.train(net, train_loader, val_loader, optimizer)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='training of the abstractor'
    )
    parser.add_argument('--ckpt_dir', required=True, help='checkpoints directory')

    parser.add_argument('--emb_dim', type=int, action='store', default=128,
                        help='the dimension of LSTM word embedding')
    parser.add_argument('--n_hidden', type=int, action='store', default=256,
                        help='the number of hidden units of LSTM')
    parser.add_argument('--n_layer', type=int, action='store', default=2,
                        help='the number of layers of LSTM')
    parser.add_argument('--bert_model', action='store', default='scibert',
                        help='bert model used for initial token embeddings')

    parser.add_argument('--data_dir', action='store', default='CDSR_data',
                        help='directory where the data is stored')
    parser.add_argument('--no_etype', action='store_true', default=False,
                        help="Don't concatenate DeepEventMine entity type embedding to the node embedding")
    parser.add_argument('--dem_model', action='store', default='mlee',
                        help='DeepEventMine pre-trained model name')
    parser.add_argument('--nograph_channel', action='store_true', default=False,
                        help="Don't use additional graph channel")
    parser.add_argument('--pretrained_ckpt', action='store', default=None,
                        help='Pretrained model checkpoint')
    parser.add_argument('--rgat', action='store_true', default=False,
                        help='use R-GAT instead of GAT')
    parser.add_argument('--is_bipartite', action='store_true', default=False,
                        help='treat DEM edges as nodes')

    # length limit
    parser.add_argument('--max_art', type=int, action='store', default=10000,
                        help='maximun tokens in a single article')
    parser.add_argument('--max_abs', type=int, action='store', default=700,
                        help='maximun words in a single abstract')
    # training options
    parser.add_argument('--dropout', type=float, action='store', default=0.1,
                        help='dropout rate')
    parser.add_argument('--lr', type=float, action='store', default=1e-3,
                        help='learning rate')
    parser.add_argument('--decay', type=float, action='store', default=0.5,
                        help='learning rate decay ratio')
    parser.add_argument('--lr_p', type=int, action='store', default=0,
                        help='patience for learning rate decay')
    parser.add_argument('--clip', type=float, action='store', default=2.0,
                        help='gradient clipping')
    parser.add_argument('--batch_size', type=int, action='store', default=5,
                        help='training batch size')
    parser.add_argument('--val_batch_size', type=int, action='store', default=16,
                        help='validation batch size')
    parser.add_argument('--num_worker', type=int, action='store', default=4,
                        help='cpu num using for dataloader')
    parser.add_argument(
        '--ckpt_freq', type=int, action='store', default=1036,
        help='number of update steps for checkpoint and validation'
    )
    parser.add_argument('--sched_thr', type=float, action='store', default=2e-3,
                        help='Threshold for measuring the new optimum, to only focus on significant changes')
    parser.add_argument('--patience', type=int, action='store', default=3,
                        help='patience for early stopping')

    parser.add_argument('--no_cuda', action='store_true',
                        help='disable GPU training')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='resume training')
    parser.add_argument('--force_lr', action='store_true', default=False,
                        help='Force to use --lr when resume training')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--wandb_log',  action='store_true', default=False,
                        help='login to wandb')
    args = parser.parse_args()

    args.bi = True
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        torch.cuda.set_device(args.gpu_id)

    main(args)
