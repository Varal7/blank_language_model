import argparse
import os
import torch
import numpy as np
from tqdm import tqdm

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

parser.add_argument('--force_legal', action='store_true',
                    help='If there are still slots, prevent stopping')

parser.add_argument('--force_insert', action='store_true',
                    help='If there are still slots and eos is chosen, pick first empty slot')


parser.add_argument('--batch_size', type=int, default=64, metavar='N',
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


def select(logits, decode):
    if decode == 'sample':
        return torch.multinomial(logits.exp(), num_samples=1)[0]
    else:
        return logits.argmax()

def write(file, sents, write_mid):
    sents = strip_eos(sents)
    if write_mid:
        for s in sents:
            file.write(' '.join(s) + '\n')
        file.write('\n')
    else:
        file.write(' '.join(sents[-1]) + '\n')
    file.flush()


def fill(seq, blanks, decode):
    seq = torch.LongTensor([vocab.bos] + seq + [vocab.eos]).to(device)
    sent_mid = [[vocab.idx2word[id] for id in seq[1:-1]]]
    mandatory_blanks = np.array([t for t in blanks])
    if len(blanks) > 0:
        while len(seq) < model.args.max_len:
            output = model(seq.unsqueeze(0))
            features = model.pool_out(
                torch.cat((output[:, :-1, :], output[:, 1:, :]), dim=-1)
            )[0]

            logits_loc = model.loc(features).squeeze(-1)

            # TODO mask
            if not (mandatory_blanks == None).all():
                logits_loc[np.array(blanks)[mandatory_blanks == None]] = float('-inf')

            p = select(logits_loc[blanks], decode)
            l = blanks[p]

            output_loc = features[l]
            logits_word = model.word(output_loc) * model.x_logit_scale

            # Can't finish sentence if not done
            if not all(e is None for e in mandatory_blanks):
                logits_word[vocab.eos] = float('-inf')

            word = select(logits_word, decode)

            if word.item() == vocab.eos:
                break

            blanks = blanks[:p+1] + [x+1 for x in blanks[p:]]
            mandatory_blanks = np.concatenate((
                mandatory_blanks[:p],
                np.array([None]),
                np.array([None]),
                [x+1 if x is not None else None for x in mandatory_blanks[p+1:]]
            ))
            seq = torch.cat((seq[:l+1], word.unsqueeze(0), seq[l+1:]))
            sent_mid.append([vocab.idx2word[id] for id in seq[1:-1]])
    return sent_mid

if __name__ == '__main__':
    args = parser.parse_args()
    set_seed(args.seed)
    vocab = Vocab(os.path.join(args.checkpoint, args.vocab))
    cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    model = get_model(os.path.join(args.checkpoint, args.model), vocab, device)
    out_path = os.path.join(args.checkpoint, args.output)

    sents = load_sent(args.data)
    output = []
    with open(out_path, 'w') as file:
        for s in tqdm(sents):
            seq, blanks = [], []
            for w in s:
                id = vocab.word2idx[w] if w in vocab.word2idx else vocab.unk
                if id == vocab.blank:
                    blanks.append(len(seq))
                else:
                    seq.append(id)
            if len(seq) not in blanks:
                blanks.append(len(seq)) # Always a blank at the end
            if args.anywhere:
                blanks = list(range(len(seq)+1))
            sent_mid = fill(seq, blanks, args.decode)
            write(file, sent_mid, args.write_mid)
