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
            Vocab.build(train_data, vocab_file, args.vocab_size, args.max_blank_length)
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
        log_gpu_memory=True,
        gpus=args.gpus,
        distributed_backend='ddp' if args.multigpu else None,
        amp_level=args.fp16_opt_level,
        precision=16 if args.fp16 else 32,
        default_root_dir=args.root_dir
    )

    trainer.fit(model, train_dataloader=train_dl, val_dataloaders=val_dl)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train',
                        help='path to training file')
    parser.add_argument('--valid',
                        help='path to validation file')
    parser.add_argument('--root_dir', default='checkpoints',
                        help='directory to save checkpoints and outputs')
    parser.add_argument('--load_checkpoint', default=None,
                        help='path to load checkpoint if specified')

    parser.add_argument('--model_type', default='blm',
                        choices=['blm', 'inst', 'lblm'],
                        help='model type: blm, inst or lblm')

    parser.add_argument('--accum_grad', type=int, default=1,
                        help='accumulate gradients across N batches.')
    parser.add_argument('--val_check_interval', type=int, default=0,
                        help='check validation set every N training batches'
                             '(0 means checking once an epoch)')

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

    temp_args, _ = parser.parse_known_args()

    # let the model add the options it needs
    parser = get_model_class(temp_args.model_type).add_model_specific_args(parser)

    args = parser.parse_args()

    if args.load_checkpoint:
        path = args.load_checkpoint
        args_dict = get_args(args.load_checkpoint)
        args = Namespace(**args_dict)
        args.load_checkpoint = path
    else:
        args.load_checkpoint = None

    main(args)
