import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F

from vocab import Vocab
from model import LM
from utils import set_seed, load_data, load_sent
from batchify import get_batches

parser = argparse.ArgumentParser()

parser.add_argument('--checkpoint', metavar='DIR', required=True,
                    help='checkpoint directory')
parser.add_argument('--data', metavar='FILE', required=True,
                    help='data file to evaluate')
parser.add_argument('--model', default='model_best.pt', metavar='FILE',
                    help='model file (in checkpoint directory)')
parser.add_argument('--vocab', default='vocab.txt', metavar='FILE',
                    help='vocab file (in checkpoint directory)')

parser.add_argument('--m', type=int, default=1, metavar='N',
                    help='num of samples for monte carlo estimate of ppl')
parser.add_argument('--k', type=int, default=10, metavar='N',
                    help='num of MC estimates')

parser.add_argument('--batch_size', type=int, default=512, metavar='N',
                    help='batch size')
parser.add_argument('--seed', type=int, default=1111, metavar='N',
                    help='random seed')
parser.add_argument('--no_cuda', action='store_true',
                    help='disable CUDA')

def get_model(path, vocab, device):
    print('Load model from {}'.format(path))
    ckpt = torch.load(path)
    train_args = ckpt['args']
    model = LM(vocab, train_args).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model

def evaluate(model, device, batches, m):
    nll = []
    with torch.no_grad():
        for batch in batches:
            seq, n = map(lambda x: x.to(device), batch)
            nll += model.nll_mc(seq, n, m).tolist()
    return nll

def main():
    args = parser.parse_args()
    set_seed(args.seed)
    vocab = Vocab(os.path.join(args.checkpoint, args.vocab))
    cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    model = get_model(os.path.join(args.checkpoint, args.model), vocab, device)

    sents = load_data(args.data, model.args.add_eos, model.args.cat_sent, model.args.max_len)
    batches, _ = get_batches(sents, vocab, args.batch_size, same_len=True)
    nlls = [evaluate(model, device, batches, args.m) for _ in range(args.k)]

    nlls = np.array(nlls)
    mean = nlls.mean(axis=0)
    std = nlls.std(axis=0)
    r = std / mean
    print('mean:\t %.2f' % mean.mean())
    print('std:\t %.2f' % std.mean())
    print('ratio:\t %.2f%%' % (r.mean() * 100))

if __name__ == '__main__':
    main()
