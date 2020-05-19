import argparse
import time
import os
import random
import collections
from tqdm import tqdm
import numpy as np
import torch

import pytorch_lightning as pl
from pytorch_lightning.logging import NeptuneLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from logger import LearningRateLogger # TODO use the thing

from model import InsTLM
from vocab import Vocab
from dataset import load_data, get_train_dataloader, get_eval_dataloader
from utils import set_seed

def main(args):
    torch.multiprocessing.set_sharing_strategy("file_system")
    neptune_token = os.getenv('NEPTUNE_API_TOKEN')
    if not neptune_token:
        raise ValueError

    set_seed(args.seed)

    args.gpus = args.num_gpus
    if args.no_cuda:
        args.gpus = 0

    lr_logger = LearningRateLogger()

    neptune_logger = NeptuneLogger(
        project_name=args.project_name,
        experiment_name=args.name
    )

    save_dir=os.path.join(
        args.root_dir,
        args.name,
        f'version_{neptune_logger.version}',
    )

    #  tb_logger = TensorBoardLogger(
    #      save_dir=save_dir,
    #      name=""
    #  )

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(
            save_dir,
            "checkpoints"
        ),
        save_top_k=5,
        period=0,
    )

    trainer = pl.Trainer(
        val_check_interval=args.checkpoint_every if args.checkpoint_every > 0 else 1.0
,
        #  default_save_path=args.root_dir,
        logger=[
            neptune_logger,
            #  tb_logger
        ],
        checkpoint_callback=checkpoint_callback,
        accumulate_grad_batches=args.accum_grad,
        callbacks=[lr_logger] if args.lr_schedule != "fixed" else None,
        amp_level=args.fp16_opt_level,
        precision=16 if args.fp16 else 32,
        log_gpu_memory=True,
        gpus=args.gpus,
        num_sanity_val_steps=1,
        distributed_backend='ddp' if torch.cuda.device_count() > 1 else None
    )

    train_sents = load_data(args.train, args.add_eos, args.cat_sent, args.max_len)
    valid_sents = load_data(args.valid, args.add_eos, args.cat_sent, args.max_len)

    os.makedirs(save_dir, exist_ok=True)

    vocab_file = os.path.join(save_dir, 'vocab.txt')
    if not os.path.isfile(vocab_file):
        Vocab.build(train_sents, vocab_file, args.vocab_size)
    vocab = Vocab(vocab_file)

    train_dl = get_train_dataloader(train_sents, vocab, args.max_tok, data_workers=args.data_workers)
    val_dl = get_eval_dataloader(valid_sents, vocab, args.eval_max_tok, data_workers=args.data_workers)

    model = InsTLM(vocab, args)
    trainer.fit(model, train_dataloader=train_dl, val_dataloaders=val_dl)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', metavar='FILE', required=True,
                        help='path to training file')
    parser.add_argument('--valid', metavar='FILE', required=True,
                        help='path to validation file')
    parser.add_argument('--root_dir', default='checkpoints', metavar='DIR',
                        help='directory to save checkpoints and outputs')
    parser.add_argument('--name', default='default', help='model name')
    parser.add_argument('--project_name', default='varal7/blm', help='project name')

    #  parser.add_argument('--load_model', default='', metavar='FILE',
                        #  help='path to load checkpoint if specified')

    parser.add_argument('--accum_grad', type=int, default=1, metavar='N',
                        help='accumulate gradients across N minibatches.')
    parser.add_argument('--checkpoint_every', type=int, default=2000, metavar='N',
                        help='save checkpoint every N steps')

    parser.add_argument('--seed', type=int, default=1111, metavar='N',
                        help='random seed')
    parser.add_argument('--data_workers', type=int, default=8, metavar='N',
                        help='data workers')

    parser.add_argument( "--fp16",
                        action="store_true",
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--fp16_opt_level",
                        type=str,
                        default="O1",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                        "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument( "--num_gpus",
                        type=str,
                        default='-1',
                        help="List of gpus to train on. Default: -1 (all gpus) ")
    parser.add_argument('--no_cuda', action='store_true',
                        help='disable CUDA')


    parser = InsTLM.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)
