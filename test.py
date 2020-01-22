import argparse
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F

from vocab import Vocab
from model import LM, collect
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
parser.add_argument('--beam_size', type=int, default=1, metavar='N', help='Beam size')
parser.add_argument('--topk', type=int, default=None, metavar='N',
                    help='Restrict vocabulary to top topk words when doing beam search')
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

def new_arange(x, *size):
    """
    Return a Tensor of `size` filled with a range function on the device of x.
    If size is empty, using the size of the variable x.
    """
    if len(size) == 0:
        size = x.size()
    return torch.arange(size[-1], device=x.device).expand(*size).contiguous()

def get_index_from_mask(blank_mask):
    pad = -1
    blanks, blanks_rank = new_arange(blank_mask).masked_fill(~blank_mask, blank_mask.size(1)).sort(1)
    recover_blanks = blanks_rank.sort(1)[1]

    blanks.masked_fill_(blanks == blank_mask.size(1), pad)
    cut_off = blanks.ne(pad).sum(1)
    blanks = blanks[:, :cut_off.max()]

    recover_blanks.masked_fill_(recover_blanks >= cut_off.unsqueeze(1), pad)

    return blanks, recover_blanks

def beam_search(seq, model, vocab, device, beam_size, topk=None):
    if topk is None:
        topk = vocab.size

    full, fill = [], []
    t_seq = torch.LongTensor([seq]).to(device) # (t, L, d)
    t_scores = torch.FloatTensor([0.]).to(device)
    t_is_fill = t_seq.new_zeros(t_seq.size())


    fill.append([vocab.idx2word[id] for id, isf in zip(t_seq[0], t_is_fill[0]) if isf])
    full.append([vocab.idx2word[id] for id in t_seq[0]])

    while vocab.is_blank(t_seq).any() and len(t_seq[0]) <= model.args.max_len:

        # Batch t: one row per hypothesis

        # Get blanks
        t_blank_mask = vocab.is_blank(t_seq)
        t_blank, _ = get_index_from_mask(t_blank_mask)
        t_blank_non_pad = t_blank.ne(-1)

        with torch.no_grad():
            t_output = model(t_seq) # (t, L, d)

        t_output_blank = collect(t_output, t_blank) # (t, nb, d)

        with torch.no_grad():
            t_lprob_loc = model.loc(t_output_blank)

        # Batch b: one row per loc (b = t * nb)
        b2t = t_blank_non_pad.nonzero()[:, 0]
        b_loc = t_blank_non_pad.nonzero()[:, 1]
        b_score_loc = t_lprob_loc[t_blank_non_pad]
        b_seq = t_seq[b2t]
        b_is_fill = t_is_fill[b2t]

        # Select output at corresponding loc (one per row)
        _, _, hdim = t_output_blank.shape
        t_output_blank.view(-1)[t_blank_non_pad.unsqueeze(-1).expand(-1, -1, hdim).reshape(-1)].view(-1, hdim)
        b_output_loc = t_output_blank.view(-1)[t_blank_non_pad.unsqueeze(-1).expand(-1, -1, hdim).reshape(-1)].view(-1, hdim) # (b, d)

        # Get positions of blanks at each loc
        b_pos = t_blank.reshape(-1)[t_blank_non_pad.view(-1)]

        # Get logits of word

        with torch.no_grad():
            b_logits_word = model.word(b_output_loc) * model.x_logit_scale
        b_logits_word[:, vocab.blank] = float('-inf')    # never predict "<blank>"
        b_lprob_word = F.log_softmax(b_logits_word, -1) # (b, V)

        # Only keep topk best words
        b_lprob_word, b_top_word_indices = b_lprob_word.topk(topk, dim=1)

        # Concatenate output with word embedding
        b_size = len(b_output_loc)
        b_output_loc = b_output_loc.unsqueeze(1).expand(-1, topk, -1) # (b, k, d)
        b_embedding_weight = model.G.src_word_emb(b_top_word_indices) # (b, k, d)
        b_output_word = torch.cat((b_output_loc, b_embedding_weight), -1) # (b, k, 2d)

        # Get logits for lrb
        with torch.no_grad():
            b_logits_lrb = model.lrb(b_output_word)
        b_lprob_lrb = F.log_softmax(b_logits_lrb, -1) # (b, k, 4)


        #  Make joint prediction (pick top-k with k=beam_size), one per loc
        b_lprob_word_lrb = b_lprob_word.unsqueeze(-1) + b_lprob_lrb # (b, k, 4)
        b_flat_lprob_word_lrb = b_lprob_word_lrb.view(b_size, -1) # (b, 4k)
        b_lprob_word_lrb, b_word_lrb = torch.topk(b_flat_lprob_word_lrb, beam_size, -1) # (b, bs)

        # Revert to correct indices
        b_lrb = b_word_lrb % 4
        b_word_lrb = b_top_word_indices.gather(1, (b_word_lrb / 4)) * 4 + b_lrb

        # Add score of location
        b_lprob_loc_word_lrb = b_lprob_word_lrb + b_score_loc

        # Add initial score
        b_new_scores = b_lprob_loc_word_lrb + t_scores[b2t].unsqueeze(-1)

        # Pick top-k where k=beam_size, (once)
        bb_scores, bb_indices = b_new_scores.view(-1).topk(beam_size)

        # Batch bb: one row per new hypothesis
        # Translate everything into new batching

        bb2b = bb_indices // beam_size
        bb_word_lrb = b_word_lrb.view(-1)[bb_indices]
        bb_word = bb_word_lrb / 4
        bb_lrb = bb_word_lrb % 4

        # Fill-in and insert
        bb_lb = bb_lrb / 2
        bb_rb = bb_lrb % 2

        # TODO: can probably be made faster
        bb_ins = [torch.tensor(([vocab.blank] if lb else []) + [word] + ([vocab.blank] if rb else [])).to(device) for lb, rb, word in zip(bb_lb, bb_rb, bb_word)]
        bb_pos = b_pos[bb2b].to(device)
        bb_seq = [torch.cat([b_seq[bb2b_idx][:pos], ins, b_seq[bb2b_idx][pos+1:]]) for bb2b_idx, pos, ins in zip(bb2b, bb_pos, bb_ins)]
        bb_is_fill = [torch.cat((b_is_fill[bb2b_idx][:pos], torch.ones(len(ins)).long().to(device), b_is_fill[bb2b_idx][pos+1:]), 0) for bb2b_idx, ins, pos in zip(bb2b, bb_ins, bb_pos)]

        # Pad
        t_seq = torch.nn.utils.rnn.pad_sequence(bb_seq, batch_first=True, padding_value = vocab.pad)
        t_is_fill = torch.nn.utils.rnn.pad_sequence(bb_is_fill, batch_first=True, padding_value=0)

        # Remove extra padding
        cutoff = t_seq.ne(vocab.pad).sum(1).max()
        t_seq = t_seq[:, :cutoff]
        t_is_fill = t_is_fill[:, :cutoff]

        t_scores = bb_scores

        fill.append([vocab.idx2word[id] for id, isf in zip(t_seq[0], t_is_fill[0]) if isf])
        full.append([vocab.idx2word[id] for id in t_seq[0]])
    return fill, full


def generate(seq, model, vocab, device, decode):
    seq = torch.LongTensor(seq).to(device)
    blanks = [i for i, w in enumerate(seq) if w == vocab.blank]
    is_fill = [0] * len(seq)
    fill = [[vocab.idx2word[id] for id, isf in zip(seq, is_fill) if isf]]
    full = [[vocab.idx2word[id] for id in seq]]
    while len(blanks) > 0 and len(seq) <= model.args.max_len:
        output = model(seq.unsqueeze(0))[0]
        output_blank = output[blanks]
        loc = select(model.loc(output_blank).squeeze(-1), decode)
        output_loc = output_blank[loc]

        logits_word = model.word(output_loc) * model.x_logit_scale
        logits_word[vocab.blank] = float('-inf')    # never predict "<blank>"

        # joint word, lrb prediction
        lprob_word = F.log_softmax(logits_word, -1)
        output_word = torch.cat((output_loc.unsqueeze(0).expand(vocab.size, -1),
            model.G.src_word_emb.weight), -1)
        logits_lrb = model.lrb(output_word)
        lprob_lrb = F.log_softmax(logits_lrb, -1)
        lprob_word_lrb = lprob_word.unsqueeze(1) + lprob_lrb
        word_lrb = select(lprob_word_lrb.view(-1), decode)
        word, lrb = word_lrb / 4, word_lrb % 4

        # predict word first and then lrb
        #word = select(logits_word, decode)
        #output_word = torch.cat((output_loc, model.G.src_word_emb(word)), dim=-1)
        #lrb = select(model.lrb(output_word), decode)

        lb, rb = lrb / 2, lrb % 2
        ins = ([vocab.blank] if lb else []) + [word] + ([vocab.blank] if rb else [])
        ins = torch.LongTensor(ins).to(device)
        pos = blanks[loc]
        seq = torch.cat((seq[:pos], ins, seq[pos+1:]))
        blanks = [i for i, w in enumerate(seq) if w == vocab.blank]
        is_fill = is_fill[:pos] + [1] * len(ins) + is_fill[pos+1:]
        fill.append([vocab.idx2word[id] for id, isf in zip(seq, is_fill) if isf])
        full.append([vocab.idx2word[id] for id in seq])
    return fill, full

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

    if args.topk:
        assert args.beam_size > 1

    if args.eval:
        sents = load_data(args.eval, model.args.add_eos, model.args.cat_sent, model.args.max_len)
        batches, _ = get_batches(sents, vocab, args.max_tok, same_len=True)
        meters = evaluate(model, device, batches, args.n_mc)
        print(' '.join(['{} {:.2f},'.format(k, meter.avg)
            for k, meter in meters.items()]))

    if args.sample:
        with open(out_path + '.fill', 'w') as f_fill:
            with open(out_path + '.full', 'w') as f_full:
                for _ in tqdm(range(args.sample)):
                    fill, full = generate([vocab.blank], model, vocab, device, args.decode)
                    write(f_fill, fill, args.write_mid)
                    write(f_full, full, args.write_mid)

    if args.fill:
        sents = load_sent(args.fill, model.args.add_eos)
        sents = [[vocab.word2idx[w] if w in vocab.word2idx else vocab.unk
            for w in s] for s in sents]
        with open(out_path + '.fill', 'w') as f_fill:
            with open(out_path + '.full', 'w') as f_full:
                for s in tqdm(sents):
                    if args.beam_size == 1:
                        fill, full = generate(s, model, vocab, device, args.decode)
                    else:
                        assert args.decode == "greedy"
                        fill, full = beam_search(s, model, vocab, device, args.beam_size, args.topk)
                    write(f_fill, fill, args.write_mid)
                    write(f_full, full, args.write_mid)

if __name__ == '__main__':
    main()
