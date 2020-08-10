import argparse
import os
import torch
import pytorch_lightning as pl
import numpy as np
from tqdm import tqdm

from vocab import Vocab
from models import InsTLM
from utils import set_seed, get_last_model_path, Bunch
from dataset import load_sent

parser = argparse.ArgumentParser()
parser.add_argument('--data', metavar='FILE', required=True,
                    help='path to data file')
parser.add_argument('--checkpoint', metavar='DIR', required=True,
                    help='checkpoint directory')
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

#  parser.add_argument('--prevent_stopping_with_unfilled_spots', action='store_true',
#                      help='If there are still unfilled mandatory blanks, prevent stopping')

#  parser.add_argument('--pick_mandatory_first', action='store_true',
#                      help='If there are still undefilled mandatory blanks, start with those')

parser.add_argument('--append_blank_at_the_end', action='store_true',
                    help='Adds a blank at the end to allow the insT to end termination')

parser.add_argument('--constrained_length_single_blank', action='store_true',
                    help='In single blank setting, forces InsT to generate correct number of tokens')

parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--seed', type=int, default=1111, metavar='N',
                    help='random seed')
parser.add_argument('--no_cuda', action='store_true',
                    help='disable CUDA')

def get_args_model(path, vocab, args_override, device):
    print('Load model from {}'.format(path))
    ckpt = torch.load(path)

    # Handle models trained with old versions of lightning...
    if 'hyper_parameters' in ckpt:
        args = ckpt['hyper_parameters']
        args.update(args_override)
        model = InsTLM(vocab, args).to(device)
    else:
        args = (ckpt['hparams'])
        args.update(args_override)
        model = InsTLM(vocab, Bunch(args)).to(device)

    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return args, model


def select(logits, decode):
    if decode == 'sample':
        return torch.multinomial(logits.exp(), num_samples=1)[0]
    else:
        return logits.argmax()

def write(file, sents, write_mid):
    if write_mid:
        for s in sents:
            file.write(' '.join(s) + '\n')
        file.write('\n')
    else:
        file.write(' '.join(sents[-1]) + '\n')
    file.flush()


def fill(seq, blanks, mandatory_blanks, decode, count=-1):
    seq = torch.LongTensor([vocab.first] + seq + [vocab.last]).to(device)
    sent_mid = [[vocab.idx2word[id] for id in seq[1:-1]]]
    if len(blanks) > 0:
        while len(seq) < model.hparams.max_len:
            output = model.forward_encoder(seq.unsqueeze(0))
            features = model.pool_out(
                torch.cat((output[:, :-1, :], output[:, 1:, :]), dim=-1)
            )[0]

            logits_loc = model.loc(features).squeeze(-1)

            # If not all mandatory blanks are completed, start with those
            if not (mandatory_blanks == None).all():
                logits_loc[np.array(blanks)[mandatory_blanks == None]] = float('-inf')

            p = select(logits_loc[blanks], decode)
            l = blanks[p]

            output_loc = features[l]
            logits_word = model.word(output_loc) * model.x_logit_scale

            # Can't finish sentence if not all mandatory blanks are completed
            if not all(e is None for e in mandatory_blanks):
                logits_word[vocab.last] = float('-inf')

            # Can't finish sentence if blank not fully completed
            if count > 0:
                logits_word[vocab.last] = float('-inf')

            word = select(logits_word, decode)

            if word.item() == vocab.last:
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

            if count > 0:
                count -= 1
                if count == 0:
                    break

    return sent_mid

if __name__ == '__main__':
    args = parser.parse_args()
    cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    args.gpus = 1 if cuda else 0

    set_seed(args.seed)
    vocab = Vocab(os.path.join(args.checkpoint, args.vocab))

    epoch, path = get_last_model_path(args.checkpoint)
    args_override = args.__dict__
    model_args, model = get_args_model(path, vocab, args_override=args_override, device=device)
    model_args['epoch'] = epoch
    out_path = os.path.join(args.checkpoint, args.output)

    sents = load_sent(args.data, model.hparams.add_eos)
    output = []
    with open(out_path, 'w') as file:
        for s in tqdm(sents):
            seq, blanks = [], []
            count = -1
            for w in s:
                # <blank_x> is not in insT vocab but can be used at inference time
                if w.startswith("<blank"):
                    blanks.append(len(seq))
                    if args.constrained_length_single_blank:
                        count = int(w.split("_")[1][:-1])
                else:
                    id = vocab.word2idx[w] if w in vocab.word2idx else vocab.unk
                    seq.append(id)

            if args.anywhere:
                blanks = list(range(len(seq)+1))

            if args.constrained_length_single_blank and len(blanks) != 1:
                raise ValueError("Contrained length only works for a single blank setting")

            if args.constrained_length_single_blank and args.append_blank_at_the_end:
                raise ValueError("Constrainted length is not compatible with appending blank at the end")

            mandatory_blanks = np.array(blanks)

            if args.append_blank_at_the_end:
                if len(seq) not in blanks:
                    blanks.append(len(seq)) # Always a blank at the end for termination
                    mandatory_blanks = np.append(mandatory_blanks, None)

            sent_mid = fill(seq, blanks, mandatory_blanks, args.decode, count)
            write(file, sent_mid, args.write_mid)
