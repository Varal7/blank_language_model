import argparse
import os
import torch
import numpy as np

from vocab import Vocab
from model import LM
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--data', metavar='FILE', required=True,
                    help='path to data file')
parser.add_argument('--checkpoint', metavar='DIR', required=True,
                    help='checkpoint directory')
parser.add_argument('--model', default='model_best.pt', metavar='FILE',
                    help='model file (in checkpoint directory)')
parser.add_argument('--vocab', default='vocab.txt', metavar='FILE',
                    help='vocab file (in checkpoint directory)')
parser.add_argument('--output', default='output.txt', metavar='FILE',
                    help='output file (in checkpoint directory)')

parser.add_argument('--decode', default='greedy', metavar='M',
                    choices=['greedy', 'sample'],
                    help='greedy decoding or sampling')
parser.add_argument('--anywhere', action='store_true',
                    help='fill in anywhere, not only blanks')
parser.add_argument('--write_mid', action='store_true',
                    help='write intermediate partial sentences')

parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--seed', type=int, default=1111, metavar='N',
                    help='random seed')
parser.add_argument('--no_cuda', action='store_true',
                    help='disable CUDA')
args = parser.parse_args()

def fill(seq, blanks):
    seq = torch.LongTensor([vocab.bos] + seq + [vocab.eos]).to(device)
    sent_mid = [[vocab.idx2word[id] for id in seq[1:-1]]]
    while len(seq) < model.args.max_len:
        logits_c, logits_l = model(seq.unsqueeze(0))
        #l = torch.multinomial(logits_l[0].exp(), num_samples=1)[0]
        #c = torch.multinomial(logits_c[0, l].exp(), num_samples=1)
        l = logits_l[0].argmax()
        c = logits_c[0, l].argmax().unsqueeze(0)
        if c[0].item() == vocab.eos:
            break
        seq = torch.cat((seq[:l+1], c, seq[l+1:]))
        sent_mid.append([vocab.idx2word[id] for id in seq[1:-1]])
    return sent_mid

if __name__ == '__main__':
    set_seed(args.seed)
    vocab = Vocab(os.path.join(args.checkpoint, args.vocab))
    cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    model = get_model(os.path.join(args.checkpoint, args.model), vocab, device)
    out_path = os.path.join(args.checkpoint, args.output)

    sents = load_sent(args.data)
    output = []
    for s in sents:
        seq, blanks = [], []
        for w in s:
            id = vocab.word2idx[w] if w in vocab.word2idx else vocab.unk
            if id == vocab.blank:
                blanks.append(len(seq))
            else:
                seq.append(id)
        output.append(fill(seq, blanks))
    write_mid_or_last(output, args.write_mid, out_path)
