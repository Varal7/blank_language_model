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
parser.add_argument('--sample', type=int, default=0, metavar='N',
                    help='num of sentences to generate')
parser.add_argument('--fill', default='', metavar='FILE',
                    help='input file to expand')

parser.add_argument('--decode', default='greedy', metavar='M',
                    choices=['greedy', 'sample'],
                    help='greedy decoding or sampling')
parser.add_argument('--write_mid', action='store_true',
                    help='write intermediate partial sentences')

parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--seed', type=int, default=1111, metavar='N',
                    help='random seed')
parser.add_argument('--no_cuda', action='store_true',
                    help='disable CUDA')
args = parser.parse_args()

def select(logits):
    if args.decode == 'sample':
        return torch.multinomial(logits.exp(), num_samples=1)[0]
    else:
        return logits.argmax()

def generate(seq=[vocab.blank], blanks=[0]):
    seq = torch.LongTensor(seq).to(device)
    sent_mid = [[vocab.idx2word[id] for id in seq]]
    while len(blanks) > 0:
        logits_loc, logits_word_lb_rb = model(seq.unsqueeze(0), blanks)
        loc = select(logits_loc[0])
        word_lb_rb = select(logits_word_lb_rb[0, loc])
        word = int(word_lb_rb / 4)
        lb = int((word_lb_rb % 4) / 2)
        rb = word_lb_rb % 2
        blanks =
        seq = torch.cat((seq[:l+1], word, seq[l+1:]))
        sent_mid.append([vocab.idx2word[id] for id in seq])
    return sent_mid

if __name__ == '__main__':
    set_seed(args.seed)
    vocab = Vocab(os.path.join(args.checkpoint, args.vocab))
    cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    model = get_model(os.path.join(args.checkpoint, args.model), vocab, device)
    out_path = os.path.join(args.checkpoint, args.output)

    if args.eval:
        sents = load_sent(args.eval)
        n_words = sum(len(s) + 1 for s in sents)    # include <eos>
        batches, _ = get_batches(sents, vocab, args.batch_size)
        meter = evaluate(model, device, batches)
        print('NLL {:.2f}'.format(meter.avg))
        print('PPL {:.2f}'.format(np.exp(meter.avg * len(sents) / n_words)))

    if args.sample:
        sents = [generate() for _ in range(args.sample)]
        write_mid_or_last(sents, args.write_mid, out_path)

    if args.expand:
        inp_sents = load_sent(args.fill)
        sents, blanks = [], []
        for s in inp_sents:
            sent, blank = [], []
            for w in s:
                id = vocab.word2idx[w] if w in vocab.word2idx else vocab.unk
                if id == vocab.blank:
                    blank.append(len(sent))
                sent.append(id)
            sents.append(sent)
            blanks.append(blank)
        sents = [generate(s, b) for s, b in zip(sents, blanks)]
        write_mid_or_last(sents, args.write_mid, out_path)
