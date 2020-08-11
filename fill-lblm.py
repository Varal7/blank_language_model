import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from vocab import Vocab
from models import LBLM
from utils import set_seed, get_last_model_path, Bunch, collect, new_arange
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
parser.add_argument('--beam_size', type=int, default=1, metavar='N',
                    help='Beam size')
parser.add_argument('--write_mid', action='store_true',
                    help='write intermediate partial sentences')
parser.add_argument('--write_fill', action='store_true',
                    help='write only filled parts')

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
        model = LBLM(vocab, args).to(device)
    else:
        args = (ckpt['hparams'])
        args.update(args_override)
        model = LBLM(vocab, Bunch(args)).to(device)

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

def get_index_from_mask(blank_mask):
    pad = -1
    blanks, blanks_rank = new_arange(blank_mask).masked_fill(~blank_mask, blank_mask.size(1)).sort(1)
    recover_blanks = blanks_rank.sort(1)[1]

    blanks.masked_fill_(blanks == blank_mask.size(1), pad)
    cut_off = blanks.ne(pad).sum(1)
    blanks = blanks[:, :cut_off.max()]

    recover_blanks.masked_fill_(recover_blanks >= cut_off.unsqueeze(1), pad)

    return blanks, recover_blanks

def beam_search(seq, model, vocab, device, write_fill, beam_size):

    # Initialize
    res = []
    t_seq = torch.LongTensor([seq]).to(device)
    t_scores = torch.FloatTensor([0.]).to(device)
    t_is_fill = t_seq.new_zeros(t_seq.size())

    while vocab.is_blank(t_seq).any() and len(t_seq[0]) <= model.hparams.max_len:

        # Batch t: one row per hypothesis
        # Get blanks
        t_blank_mask = vocab.is_blank(t_seq)
        t_blank, _ = get_index_from_mask(t_blank_mask)
        t_blank_non_pad = t_blank.ne(-1)

        with torch.no_grad():
            t_output = model.forward_encoder(t_seq)

        t_output_blank = collect(t_output, t_blank)

        with torch.no_grad():
            t_lprob_loc = model.loc(t_output_blank)

        # Batch b: one row per loc
        b2t = t_blank_non_pad.nonzero()[:, 0]
        b_loc = t_blank_non_pad.nonzero()[:, 1]
        b_score_loc = t_lprob_loc[t_blank_non_pad]
        b_seq = t_seq[b2t]
        b_is_fill = t_is_fill[b2t]

        # Select output at corresponding loc (one per row)
        _, _, hdim = t_output_blank.shape
        t_output_blank.view(-1)[t_blank_non_pad.unsqueeze(-1).expand(-1, -1, hdim).reshape(-1)].view(-1, hdim)
        b_output_loc = t_output_blank.view(-1)[t_blank_non_pad.unsqueeze(-1).expand(-1, -1, hdim).reshape(-1)].view(-1, hdim)

        # Get values of blanks at each loc
        b_pos = t_blank.reshape(-1)[t_blank_non_pad.view(-1)]
        b_previous = b_seq.gather(1, b_pos.unsqueeze(1)).squeeze(-1)
        b_length_previous = b_previous - vocab.blanks[0]

        # Get logits of word
        with torch.no_grad():
            b_logits_word = model.word(b_output_loc) * model.x_logit_scale
        b_lprob_word = F.log_softmax(b_logits_word, -1)

        # Concatenate output with word embedding
        b_size = len(b_output_loc)
        b_output_word = torch.cat((b_output_loc.unsqueeze(1).expand(-1, vocab.size, -1),
            model.G.src_word_emb.weight.expand(b_size, -1, -1)), -1)

        # Get logits for lrb
        with torch.no_grad():
            b_logits_lrb = model.lrb(b_output_word)

        # Mask out illegal lrb outputs
        _, vocab_size, max_blank_len = b_logits_lrb.shape
        b_ta = b_length_previous.unsqueeze(-1).unsqueeze(-1).expand(-1, vocab_size, max_blank_len)
        b_ra = torch.arange(max_blank_len).unsqueeze(0).unsqueeze(0).expand(b_size, vocab_size, -1).to(device)
        b_logits_lrb.masked_fill_(b_ra >= b_ta, float('-inf'))
        b_lprob_lrb = F.log_softmax(b_logits_lrb, -1)


        #  Make joint prediction (pick top-k, k=beam_size), one per loc
        b_lprob_word_lrb = b_lprob_word.unsqueeze(-1) + b_lprob_lrb
        b_flat_lprob_word_lrb = b_lprob_word_lrb.view(b_size, -1)
        b_lprob_word_lrb, b_word_lrb = torch.topk(b_flat_lprob_word_lrb, beam_size, -1)

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
        bb_word = bb_word_lrb / max_blank_len
        bb_lrb = bb_word_lrb % max_blank_len
        bb_length_previous = b_length_previous[bb2b]

        # Fill-in and insert
        bb_lb = bb_lrb
        bb_rb = bb_length_previous - bb_lb - 1

        # TODO: can probably be made faster
        bb_ins = [torch.tensor(([vocab.blanks[lb]] if lb else []) + [word] + ([vocab.blanks[rb]] if rb else [])).to(device) for lb, rb, word in zip(bb_lb, bb_rb, bb_word)]
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

        cur = ([vocab.idx2word[id] for i, id in enumerate(t_seq[0]) if  t_is_fill[0][i] or (not write_fill and id != vocab.pad)])
        res.append(cur)

    return res



def generate(seq, model, vocab, device, decode, write_fill):
    seq = torch.LongTensor(seq).to(device)
    blanks = [i for i, w in enumerate(seq) if vocab.is_blank(w)]
    is_fill = [0] * len(seq)
    res = [[vocab.idx2word[id] for i, id in enumerate(seq) if is_fill[i] or not write_fill]]

    while len(blanks) > 0 and len(seq) <= model.hparams.max_len:
        output = model.forward_encoder(seq.unsqueeze(0))[0]
        output_blank = output[blanks]
        loc = select(model.loc(output_blank).squeeze(-1), decode)
        output_loc = output_blank[loc]

        previous = seq[blanks[loc]]
        length_previous = int(vocab.idx2word[seq[blanks[loc]]].replace("<blank_", "").replace(">", ""))


        # joint word, lrb prediction
        logits_word = model.word(output_loc) * model.x_logit_scale
        lprob_word = F.log_softmax(logits_word, -1)
        output_word = torch.cat((output_loc.unsqueeze(0).expand(vocab.size, -1),
            model.G.src_word_emb.weight), -1)
        logits_lrb = model.lrb(output_word)
        logits_lrb[:, length_previous:] = float('-inf')
        max_blank_len = logits_lrb.shape[1]
        lprob_lrb = F.log_softmax(logits_lrb, -1)
        lprob_word_lrb = lprob_word.unsqueeze(1) + lprob_lrb
        word_lrb = select(lprob_word_lrb.view(-1), decode)
        word, lrb = word_lrb / max_blank_len, word_lrb % max_blank_len

        # predict word first and then lrb
        #word = select(model.word(output_loc) * model.x_logit_scale, decode)
        #output_word = torch.cat((output_loc, model.G.src_word_emb(word)), dim=-1)
        #lrb = select(model.lrb(output_word), decode)

        lrb = min(lrb, length_previous - 1)
        lb = lrb
        rb = length_previous - lb - 1

        ins = ([vocab.blanks[lb]] if lb else []) + [word] + ([vocab.blanks[rb]] if rb else [])
        ins = torch.LongTensor(ins).to(device)
        pos = blanks[loc]
        seq = torch.cat((seq[:pos], ins, seq[pos+1:]))
        blanks = [i for i, w in enumerate(seq) if vocab.is_blank(w)]
        is_fill = is_fill[:pos] + [1] * len(ins) + is_fill[pos+1:]
        res.append([vocab.idx2word[id] for i, id in enumerate(seq) if is_fill[i] or not write_fill])
    return res


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

    sents = [[vocab.word2idx[w] if w in vocab.word2idx else vocab.unk
        for w in s] for s in sents]
    with open(out_path, 'w') as f:
        for s in tqdm(sents):
            if args.beam_size == 1:
                res = generate(s, model, vocab, device, args.decode, args.write_fill)
            else:
                res = beam_search(s, model, vocab, device, args.write_fill, args.beam_size)
            write(f, res, args.write_mid)
