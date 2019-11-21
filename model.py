import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from transformer.Models import Encoder
from transformer.Optim import *

def seq_cross_entropy(pred, gold, pad, smoothing):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    pred = pred.contiguous().view(-1, pred.size(2))
    gold = gold.contiguous().view(-1)

    if smoothing:
        import pdb; pdb.set_trace()
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(pad)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=pad)

    return loss

def get_canvas(seq, keep, __):
    bs, n = seq.size(0), seq.size(1)
    ___ = torch.zeros(bs, 1, dtype=torch.long, device=seq.device).fill_(__)
    canvas = torch.zeros(bs, 0, dtype=torch.long, device=seq.device)
    blanks, rest, loc, lb, rb = [], [], [], [], []
    i = 0
    while i < n:
        if i in keep:
            canvas = torch.cat((canvas, seq[:, i:i+1]), dim=1)
            i += 1
        else:
            a = []
            while i < n and i not in keep:
                rest.append(i)
                loc.append(len(blanks))
                a.append(1)
                i += 1
            lb += [0] + a[1:]
            rb += a[:-1] + [0]
            blanks.append(canvas.size(1))
            canvas = torch.cat((canvas, ___), dim=1)
    return canvas, blanks, rest, loc, lb, rb

class LM(nn.Module):
    """Language Model"""

    def __init__(self, vocab, args):
        super().__init__()
        self.vocab = vocab
        self.args = args

        self.log_factorial = [0]
        for i in range(1, args.max_len+1):
            self.log_factorial.append(self.log_factorial[-1] + np.log(i))

        self.G = Encoder(
            n_src_vocab=vocab.size, len_max_seq=args.max_len,
            d_word_vec=args.d_model, d_model=args.d_model, d_inner=args.d_inner_hid,
            n_layers=args.n_layers, n_head=args.n_head, d_k=args.d_k, d_v=args.d_v,
            dropout=args.dropout)

        self.tgt_word_prj = nn.Linear(args.d_model, vocab.size, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)
        self.loc = nn.Linear(args.d_model, 1, bias=False)
        self.lrb = nn.Linear(args.d_model*2, 4, bias=False)

        opt = optim.Adam(self.parameters(), betas=eval(args.adam_betas),
            eps=args.adam_eps, weight_decay=args.weight_decay)
        if args.lr_schedule == 'inverse_sqrt':
            self.opt = InverseSqrtScheduler(opt, args.lr, args.warmup_steps)
        elif args.lr_schedule == 'linear_decay':
            self.opt = LinearDecayScheduler(opt, args.lr, args.warmup_steps, args.train_steps)
        else:
            self.opt = LRScheduler(opt, args.lr)

    def forward(self, canvas, blanks):
        pos = torch.arange(canvas.size(1)).repeat(len(canvas), 1).to(canvas.device)
        output, *_ = self.G(canvas, pos)
        return output[:, blanks, :]

    def loss(self, seq):
        n = seq.size(1)
        k = np.random.randint(n)
        keep = sorted(np.random.permutation(n)[:k])
        canvas, blanks, rest, loc, lb, rb = get_canvas(seq, keep, self.vocab.blank)
        output = self(canvas, blanks)

        logits_loc = self.loc(output).squeeze(-1)
        loss_loc = -F.log_softmax(logits_loc, dim=1)[:, loc].mean()
        output_loc = output[:, loc, :]

        logits_word = self.tgt_word_prj(output_loc)
        loss_word = seq_cross_entropy(logits_word, seq[:, rest],
            self.vocab.pad, self.args.label_smoothing)
        output_loc_word = torch.cat((output_loc, self.G.src_word_emb(seq[:, rest])), dim=-1)

        logits_lrb = self.lrb(output_loc_word)
        lrb = (torch.tensor(lb) * 2 + torch.tensor(rb)).to(canvas.device)
        loss_lrb = F.cross_entropy(logits_lrb.view(-1, 4), lrb.repeat(len(canvas)))

        return (loss_loc+loss_word+loss_lrb) * n - self.log_factorial[n]
