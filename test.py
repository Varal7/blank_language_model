import argparse
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F

from vocab import Vocab
from model import LM
from utils import set_seed, load_data, load_sent, strip_eos
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
                    help='input file to fill')

parser.add_argument('--n_mc', type=int, default=100, metavar='N',
                    help='num of samples for monte carlo estimate of ppl')
parser.add_argument('--decode', default='greedy', metavar='M',
                    choices=['greedy', 'sample'],
                    help='greedy decoding or sampling')
parser.add_argument('--write_mid', action='store_true',
                    help='write intermediate partial sentences')

#parser.add_argument('--batch_size', type=int, default=512, metavar='N',
#                    help='batch size')
parser.add_argument('--max_tok', type=int, default=40000, metavar='N',
                    help='max number of tokens per batch')
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

def select(logits, decode):
    if decode == 'sample':
        return torch.multinomial(logits.exp(), num_samples=1)[0]
    else:
        return logits.argmax()

def generate(seq, model, vocab, device, decode):
    seq = torch.LongTensor([vocab.first] + seq + [vocab.last]).to(device)
    sent_mid = [[vocab.idx2word[id] for id in seq[1:-1]]]
    while len(seq) <= model.args.max_len:
        output = model(seq.unsqueeze(0))
        features = model.pool_out(
            torch.cat((output[:, :-1, :], output[:, 1:, :]), dim=-1)
        )[0]

        logits_loc = model.loc(features).squeeze(-1)
        loc = select(logits_loc, decode)
        output_loc = features[loc]
        logits_word = model.word(output_loc) * model.x_logit_scale
        word = select(logits_word, decode)

        if loc == (logits_loc.size(0) - 1) and word.item() == vocab.eos:
            break

        seq = torch.cat((seq[:loc + 1], word.unsqueeze(0), seq[loc + 1:]))
        sent_mid.append([vocab.idx2word[id] for id in seq[1:-1]])
    return sent_mid

def write(file, sents, write_mid):
    sents = strip_eos(sents)
    if write_mid:
        for s in sents:
            file.write(' '.join(s) + '\n')
        file.write('\n')
    else:
        file.write(' '.join(sents[-1]) + '\n')
    file.flush()

def main():
    args = parser.parse_args()
    set_seed(args.seed)
    vocab = Vocab(os.path.join(args.checkpoint, args.vocab))
    cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    model = get_model(os.path.join(args.checkpoint, args.model), vocab, device)
    out_path = os.path.join(args.checkpoint, args.output)

    if os.path.exists(out_path):
        raise ValueError

    if args.eval:
        sents = load_data(args.eval, model.args.add_eos, model.args.cat_sent, model.args.max_len - 2)
        batches, _ = get_batches(sents, vocab, args.max_tok, same_len=True)
        meters = evaluate(model, device, batches, args.n_mc)
        with open(out_path, 'w') as w:
            for k, meter in meters.items():
                res = '{} {:.4f}'.format(k, meter.avg)
                print(res)
                w.write(res + "\n")

    if args.sample:
        with open(out_path + '.full', 'w') as f_full:
            for _ in tqdm(range(args.sample)):
                sent_mid = generate([], model, vocab, device, args.decode)
                write(f_full, sent_mid, args.write_mid)

    if args.fill:
        sents = load_sent(args.fill, model.args.add_eos)
        sents = [[vocab.word2idx[w] if w in vocab.word2idx else vocab.unk
            for w in s] for s in sents]
        with open(out_path + '.full', 'w') as f_full:
            for s in tqdm(sents):
                sent_mid = generate(s, model, vocab, device, args.decode)
                write(f_full, sent_mid, args.write_mid)

if __name__ == '__main__':
    main()
