import argparse
import os
import torch
import numpy as np

from vocab import Vocab
from model import LM
from utils import *
from batchify import get_batches
from train import evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', metavar='DIR', required=True,
                    help='checkpoint directory')
parser.add_argument('--model', default='model_best.pt', metavar='FILE',
                    help='model file (in checkpoint directory)')
parser.add_argument('--vocab', default='vocab.txt', metavar='FILE',
                    help='vocab file (in checkpoint directory)')
parser.add_argument('--output', default='output.txt', metavar='FILE',
                    help='output file (in checkpoint directory)')

parser.add_argument('--eval', default='', metavar='FILE',
                    help='data file to evaluate')
parser.add_argument('--sample', action='store_true',
                    help='sample sentences')
parser.add_argument('--expand', default='', metavar='FILE',
                    help='input file to expand')

parser.add_argument('--write_mid', action='store_true',
                    help='write intermediate partial sentences')
parser.add_argument('--n', type=int, default=1000, metavar='N',
                    help='num of sentences to generate')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')

parser.add_argument('--seed', type=int, default=1111, metavar='N',
                    help='random seed')
parser.add_argument('--no_cuda', action='store_true',
                    help='disable CUDA')
args = parser.parse_args()

def get_model(path):
    print('Load model from {}'.format(path))
    ckpt = torch.load(path)
    train_args = ckpt['args']
    model = LM(vocab, train_args).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model

def generate(seq=[]):
    seq = torch.LongTensor([vocab.bos] + seq + [vocab.eos]).to(device)
    sent_mid = [[vocab.idx2word[id] for id in seq[1:-1]]]
    while len(seq) < model.args.max_len:
        logits_c, logits_l = model(seq.unsqueeze(0))
        l = torch.multinomial(logits_l[0].exp(), num_samples=1)[0]
        c = torch.multinomial(logits_c[0, l].exp(), num_samples=1)
        if c[0].item() == vocab.eos:
            break
        seq = torch.cat((seq[:l+1], c, seq[l+1:]))
        sent_mid.append([vocab.idx2word[id] for id in seq[1:-1]])
    return sent_mid

def write(sents_mid):
    out_path = os.path.join(args.checkpoint, args.output)
    if args.write_mid:
        write_doc(sents_mid, out_path)
    else:
        write_sent([s[-1] for s in sents_mid], out_path)

if __name__ == '__main__':
    set_seed(args.seed)
    vocab = Vocab(os.path.join(args.checkpoint, args.vocab))
    cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    model = get_model(os.path.join(args.checkpoint, args.model))

    if args.eval:
        sents = load_sent(args.eval)
        n_words = sum(len(s) + 1 for s in sents)    # include <eos>
        batches, _ = get_batches(sents, vocab, args.batch_size)
        meter = evaluate(model, device, batches)
        print('NLL {:.2f}'.format(meter.avg))
        print('PPL {:.2f}'.format(np.exp(meter.avg * len(sents) / n_words)))

    if args.sample:
        sents = [generate() for _ in range(args.n)]
        write(sents)

    if args.expand:
        sents = load_sent(args.expand)
        sents = [[vocab.word2idx[w] if w in vocab.word2idx else vocab.unk
            for w in s] for s in sents]
        sents = [generate(s) for s in sents]
        write(sents)
