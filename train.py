import argparse
from argparse import Namespace
import os
import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateLogger

from models import InsTLM, BLM, LBLM
from vocab import Vocab
from dataset import load_data, get_train_dataloader, get_eval_dataloader


def get_model_class(model_type):
    if model_type == 'blm':
        return BLM
    elif model_type == 'inst':
        return InsTLM
    elif model_type == 'lblm':
        return LBLM
    else:
        raise ValueError('Unknown model ' + model_type)


def get_args(path):
    print('Load model from {}'.format(path))
    ckpt = torch.load(path)
    args = ckpt['hparams']
    return args


def main(args):
    torch.multiprocessing.set_sharing_strategy("file_system")

    pl.seed_everything(args.seed)

    lr_logger = LearningRateLogger()

    args.multigpu = torch.cuda.device_count() > 1

    train_data = load_data(args.train, args.add_eos, args.cat_sent, args.max_len)
    valid_data = load_data(args.valid, args.add_eos, args.cat_sent, args.max_len)

    os.makedirs(args.root_dir, exist_ok=True)

    vocab_file = os.path.join(args.root_dir, 'vocab.txt')
    if not os.path.isfile(vocab_file):
        if args.model_type == 'lblm':
            Vocab.build(train_data, vocab_file, args.vocab_size, args.max_len)
        else:
            Vocab.build(train_data, vocab_file, args.vocab_size)
    vocab = Vocab(vocab_file)

    train_dl = get_train_dataloader(
        train_data, vocab, args.max_tok,
        data_workers=args.data_workers if not args.multigpu else 0,
        model_type=args.model_type)
    val_dl = get_eval_dataloader(
        valid_data, vocab, args.eval_max_tok,
        data_workers=args.data_workers if not args.multigpu else 0,
        model_type=args.model_type)

    model = get_model_class(args.model_type)(vocab, args)

    if args.load_checkpoint:
        ckpt = torch.load(args.load_checkpoint)
        model.load_state_dict(ckpt['state_dict'])

    trainer = pl.Trainer(
        accumulate_grad_batches=args.accum_grad,
        callbacks=[lr_logger] if args.lr_schedule != 'fixed' else None,
        val_check_interval=args.val_check_interval if args.val_check_interval > 0 else 1.0,
        gpus=args.gpus,
        distributed_backend='ddp' if args.multigpu else None,
        amp_level=args.fp16_opt_level,
        precision=16 if args.fp16 else 32,
        default_root_dir=args.root_dir
    )

    trainer.fit(model, train_dataloader=train_dl, val_dataloaders=val_dl)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Path
    parser.add_argument('--train',
                        help='path to training file')
    parser.add_argument('--valid',
                        help='path to validation file')
    parser.add_argument('--root_dir', default='checkpoints',
                        help='directory to save checkpoints and outputs')
    parser.add_argument('--load_checkpoint', default=None,
                        help='path to load checkpoint if specified')

    # Data
    parser.add_argument('--vocab_size', type=int, default=10000,
                        help='keep N most frequent words in vocabulary')
    parser.add_argument('--max_len', type=int, default=512,
                        help='max sequence length')
    parser.add_argument('--cat_sent', action='store_true',
                        help='concat sentences and chunk into size of max_len')
    parser.add_argument('--add_eos', action='store_true',
                        help='add <eos> at the end of each sentence')

    # Model
    parser.add_argument('--model_type', default='blm',
                        choices=['blm', 'inst', 'lblm'],
                        help='model type: blm, inst or lblm')

    parser.add_argument('--d_model', type=int, default=512,
                        help='transformer dimension d_model')
    parser.add_argument('--d_inner_hid', type=int, default=2048,
                        help='transformer dimension d_inner_hid')
    parser.add_argument('--d_k', type=int, default=64,
                        help='transformer dimension d_k')
    parser.add_argument('--d_v', type=int, default=64,
                        help='transformer dimension d_v')
    parser.add_argument('--n_head', type=int, default=8,
                        help='number of attention heads')
    parser.add_argument('--n_layers', type=int, default=6,
                        help='number of layers')
    parser.add_argument('--share_emb_prj_weight', action='store_true',
                        help='share word embedding and projection weights')

    # Optimization
    parser.add_argument('--max_tok', type=int, default=10000,
                        help='max number of tokens per batch')
    parser.add_argument('--accum_grad', type=int, default=1,
                        help='accumulate gradients across N batches.')

    parser.add_argument('--adam_betas', default='(0.9, 0.999)',
                        help='adam betas')
    parser.add_argument('--adam_eps', type=float, default=1e-8,
                        help='adam eps')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='weight decay')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='dropout probability (0 = no dropout)')

    parser.add_argument('--lr_schedule', default='fixed',
                        choices=['fixed', 'triangular'],
                        help='learning rate schedule')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--warmup_steps', type=int, default=4000,
                        help='number of warmup steps (inverse_sqrt)')
    parser.add_argument('--train_steps', type=int, default=300000,
                        help='number of training steps')

    # Validation
    parser.add_argument('--eval_max_tok', type=int, default=40000,
                        help='max number of tokens per batch for evaluation')
    parser.add_argument('--val_check_interval', type=int, default=0,
                        help='check validation set every N training batches'
                             '(0 means checking once an epoch)')
    parser.add_argument('--n_mc', type=int, default=1,
                        help='num of samples for Monte Carlo estimate of ppl')

    # Others
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--data_workers', type=int, default=8,
                        help='data workers')
    parser.add_argument('--gpus', type=int, default=-1,
                        help='number of gpus to train on (-1 means all gpus)')
    parser.add_argument('--fp16', action='store_true',
                        help='whether to use 16-bit (mixed) precision '
                             '(through NVIDIA apex) instead of 32-bit')
    parser.add_argument('--fp16_opt_level', default='O1',
                        help="for fp16: Apex AMP optimization level selected "
                             "in ['O0', 'O1', 'O2', and 'O3']. see details at "
                             "https://nvidia.github.io/apex/amp.html")

    args = parser.parse_args()

    if args.load_checkpoint:
        path = args.load_checkpoint
        args_dict = get_args(args.load_checkpoint)
        args = Namespace(**args_dict)
        args.load_checkpoint = path
    else:
        args.load_checkpoint = None

    main(args)
