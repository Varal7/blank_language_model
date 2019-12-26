import argparse
import time
import os
import random
import collections
import numpy as np
import torch

from model import LM
from vocab import Vocab
from meters import AverageMeter
from utils import set_seed, logging, load_data
from batchify import get_batches

parser = argparse.ArgumentParser()

parser.add_argument('--train', metavar='FILE', required=True,
                    help='path to training file')
parser.add_argument('--valid', metavar='FILE', required=True,
                    help='path to validation file')
parser.add_argument('--save_dir', default='checkpoints', metavar='DIR',
                    help='directory to save checkpoints and outputs')
parser.add_argument('--load_model', default='', metavar='FILE',
                    help='path to load checkpoint if specified')

parser.add_argument('--vocab_size', type=int, default=10000, metavar='N',
                    help='keep N most frequent words in vocabulary')
parser.add_argument('--max_len', type=int, default=50, metavar='N',
                    help='max sequence length')
parser.add_argument('--cat_sent', action='store_true',
                    help='concat sentences and then chunk into size of max_len')
parser.add_argument('--add_eos', action='store_true',
                    help='add <eos> at the end of each sentence')

parser.add_argument('--d_model', type=int, default=512, metavar='N',
                    help='transformer dimension d_model')
parser.add_argument('--d_inner_hid', type=int, default=2048, metavar='N',
                    help='transformer dimension d_inner_hid')
parser.add_argument('--d_k', type=int, default=64, metavar='N',
                    help='transformer dimension d_k')
parser.add_argument('--d_v', type=int, default=64, metavar='N',
                    help='transformer dimension d_v')
parser.add_argument('--n_head', type=int, default=8, metavar='N',
                    help='number of attention heads')
parser.add_argument('--n_layers', type=int, default=6, metavar='N',
                    help='number of layers')

parser.add_argument('--share_emb_prj_weight', action='store_true',
                    help='share word embedding and projection weights')
parser.add_argument('--label_smoothing', action='store_true',
                    help='label smoothing')

parser.add_argument('--adam_betas', default='(0.9, 0.999)', metavar='(R, R)',
                    help='adam betas')
parser.add_argument('--adam_eps', type=float, default=1e-8, metavar='R',
                    help='adam eps')
parser.add_argument('--weight_decay', type=float, default=1e-5, metavar='R',
                    help='weight decay')
parser.add_argument('--dropout', type=float, default=0.1, metavar='P',
                    help='dropout probability (0 = no dropout)')

parser.add_argument('--lr_schedule', default='fixed', metavar='S',
                    choices=['fixed', 'reduce_on_plateau',
                             'inverse_sqrt', 'linear_decay'],
                    help='learning rate schedule')
parser.add_argument('--lr', type=float, default=0.0001, metavar='R',
                    help='learning rate')
parser.add_argument('--lr_decay', type=float, default=4, metavar='R',
                    help='learning rate decay factor (reduce_on_plateau)')
parser.add_argument('--warmup_steps', type=int, default=4000, metavar='N',
                    help='number of warmup steps (inverse_sqrt)')
parser.add_argument('--train_steps', type=int, default=300000, metavar='N',
                    help='number of training steps')
#parser.add_argument('--epochs', type=int, default=30, metavar='N',
#                    help='number of training epochs')

parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size')
#parser.add_argument('--max_tokens', type=int, default=4000, metavar='N',
#                    help='max number of tokens per batch')
parser.add_argument('--accum_grad', type=int, default=1, metavar='N',
                    help='accumulate gradients across N minibatches.')

parser.add_argument('--eval_batch_size', type=int, default=512, metavar='N',
                    help='batch size for evaluation')
parser.add_argument('--n_mc', type=int, default=1, metavar='N',
                    help='num of samples for monte carlo estimate of ppl')
                    
parser.add_argument('--checkpoint_every', type=int, default=2000, metavar='N',
                    help='save checkpoint every N steps')
parser.add_argument('--log_every', type=int, default=100, metavar='N',
                    help='report loss every N steps')
parser.add_argument('--seed', type=int, default=1111, metavar='N',
                    help='random seed')
parser.add_argument('--no_cuda', action='store_true',
                    help='disable CUDA')

def evaluate(model, device, batches, m):
    model.eval()
    meters = collections.defaultdict(lambda: AverageMeter())
    total_nll = 0.
    n_words = 0
    with torch.no_grad():
        for batch in batches:
            seq, n = map(lambda x: x.to(device), batch)
            losses = model.losses(seq, n)
            for k, v in losses.items():
                meters[k].update(v.item())
            total_nll += model.nll_mc(seq, n, m).sum().item()
            n_words += n.sum().item()
    meters['ppl'].update(np.exp(total_nll / n_words))
    return meters

def main():
    args = parser.parse_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    log_file = os.path.join(args.save_dir, 'log.txt')
    logging(str(args), log_file)

    # Prepare data
    train_sents = load_data(args.train, args.add_eos, args.cat_sent, args.max_len)
    valid_sents = load_data(args.valid, args.add_eos, args.cat_sent, args.max_len)
    vocab_file = os.path.join(args.save_dir, 'vocab.txt')
    if not os.path.isfile(vocab_file):
        Vocab.build(train_sents, vocab_file, args.vocab_size)
    vocab = Vocab(vocab_file)
    train_batches, _ = get_batches(train_sents, vocab, args.batch_size)
    valid_batches, _ = get_batches(valid_sents, vocab, args.eval_batch_size, same_len=True)
    logging('# train sents {}, tokens {}, batches {}'.format(len(train_sents),
        sum(len(s) for s in train_sents), len(train_batches)), log_file)
    logging('# valid sents {}, tokens {}, batches {}'.format(len(valid_sents),
        sum(len(s) for s in valid_sents), len(valid_batches)), log_file)
    logging('# vocab size {}'.format(vocab.size), log_file)

    set_seed(args.seed)
    cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    model = LM(vocab, args).to(device)
    if args.load_model:
        logging('Load model from {}'.format(args.load_model), log_file)
        ckpt = torch.load(args.load_model)
        model.load_state_dict(ckpt['model'])
    logging('# model parameters: {}'.format(
        sum(x.data.nelement() for x in model.parameters())), log_file)

    logging('-' * 80, log_file)
    start_time = time.time()
    random.shuffle(train_batches)
    model.train()
    meters = collections.defaultdict(lambda: AverageMeter())
    best_val_ppl = None
    index = 0
    for step in range(1, 1 + args.train_steps):
        model.opt.zero_grad()
        for _ in range(args.accum_grad):
            seq, n = map(lambda x: x.to(device), train_batches[index])
            losses = model.losses(seq, n)
            for k, v in losses.items():
                meters[k].update(v.item())
                losses[k] /= args.accum_grad
            losses['loss'].backward()
            index = (index + 1) % len(train_batches)
        model.opt.step()

        if step % args.log_every == 0:
            log = '| step {:6d}/{:6d} |'.format(step, args.train_steps)
            for k, meter in meters.items():
                log += ' {} {:.2f},'.format(k, meter.avg)
                meter.clear()
            logging(log, log_file)

        if step % args.checkpoint_every == 0:
            logging('-' * 80, log_file)
            valid_meters = evaluate(model, device, valid_batches, args.n_mc)
            model.train()
            ckpt = {'args': args, 'model': model.state_dict()}
            if not best_val_ppl or valid_meters['ppl'].avg < best_val_ppl:
                best_val_ppl = valid_meters['ppl'].avg
                torch.save(ckpt, os.path.join(args.save_dir, 'model_best.pt'))
            elif args.lr_schedule == 'reduce_on_plateau':
                model.opt.lr /= args.lr_decay
                model.opt.set_lr()
            torch.save(ckpt, os.path.join(args.save_dir, 'model_last.pt'))
            log = '| step {:6d} | time {:5.0f}s | lr {:.7f} | valid'.format(
                step, time.time() - start_time, model.opt.lr)
            for k, meter in valid_meters.items():
                log += ' {} {:.2f},'.format(k, meter.avg)
            log += ' | best {:.2f}'.format(best_val_ppl)
            logging(log, log_file)
            logging('-' * 80, log_file)

if __name__ == '__main__':
    main()
