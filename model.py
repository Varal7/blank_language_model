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
    blanks, rest, pos, l_, r_ = [], [], [], [], []
    i = 0
    while i < n:
        if i in keep:
            canvas = torch.cat((canvas, seq[:, i:i+1]), dim=1)
            i += 1
        else:
            blanks.append(canvas.size(1))
            a = []
            while i < n and i not in keep:
                rest.append(i)
                pos.append(canvas.size(1))
                a.append(1)
                i += 1
            l_ += [0] + a[1:]
            r_ += a[:-1] + [0]
            canvas = torch.cat((canvas, ___), dim=1)
    return canvas, blanks, rest, pos, l_, r_

class LM(nn.Module):
    """Language Model"""

    def __init__(self, vocab, args):
        super().__init__()
        self.vocab = vocab
        self.args = args

        self.log_factorial = [0]
        for i in range(1, args.max_len):
            self.log_factorial.append(self.log_factorial[-1] + np.log(i))

        self.G = Encoder(
            n_src_vocab=vocab.size, len_max_seq=args.max_len,
            d_word_vec=args.d_model, d_model=args.d_model, d_inner=args.d_inner_hid,
            n_layers=args.n_layers, n_head=args.n_head, d_k=args.d_k, d_v=args.d_v,
            dropout=args.dropout)

        self.tgt_word_prj = nn.Linear(2*args.d_model, vocab.size, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)

        if args.share_emb_prj_weight:
            self.tgt_word_prj.weight = self.G.src_word_emb.weight
            self.x_logit_scale = (args.d_model ** -0.5)
        else:
            self.x_logit_scale = 1.

        self.loc = nn.Linear(2*args.d_model, 1, bias=False)

        opt = optim.Adam(self.parameters(), betas=eval(args.adam_betas),
            eps=args.adam_eps, weight_decay=args.weight_decay)
        if args.lr_schedule == 'inverse_sqrt':
            self.opt = InverseSqrtScheduler(opt, args.lr, args.warmup_steps)
        elif args.lr_schedule == 'linear_decay':
            self.opt = LinearDecayScheduler(opt, args.lr, args.warmup_steps, args.train_steps)
        else:
            self.opt = LRScheduler(opt, args.lr)

    def forward(self, canvas, blanks):
        pos = torch.arange(canvas.size(1)).repeat(canvas.size(0), 1).to(seq.device)
        output, *_ = self.G(canvas, pos)

        slot = torch.cat((output[:, :-1, :], output[:, 1:, :]), dim=-1)
        logits_c = self.tgt_word_prj(slot) * self.x_logit_scale
        logits_l = self.loc(slot).squeeze(-1)
        return logits_c, logits_l

    def loss(self, seq):
        n = seq.size(1)
        k = np.random.randint(n)
        keep = sorted(np.random.permutation(n)[:k])
        canvas, blanks, rest, pos, l_, r_ = get_canvas(seq, keep, self.vocab.blank)

        import pdb; pdb.set_trace()
        logits_c, logits_l = self(canvas, blanks)
        lp_c = -seq_cross_entropy(logits_c[:, slot, :], seq[:, rest],
            self.vocab.pad, self.args.label_smoothing)
        lp_l = F.log_softmax(logits_l, dim=1)[:, slot].mean()

        return -self.log_prb(seq, canvas, rest, slot)*(n+1) - self.log_factorial[n]
