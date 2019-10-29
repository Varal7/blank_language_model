import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from transformer.Models import Encoder
from transformer.Optim import *

def loss_rec(pred, gold, pad, smoothing):
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

    def forward(self, seq):
        pos = torch.arange(seq.size(1)).repeat(seq.size(0), 1).to(seq.device)
        output, *_ = self.G(seq, pos)
        slot = torch.cat((output[:, :-1, :], output[:, 1:, :]), dim=-1)
        logits_c = self.tgt_word_prj(slot) * self.x_logit_scale
        logits_l = self.loc(slot).squeeze(-1)
        return logits_c, logits_l

    def log_prb(self, seq, canvas, rest, slot):
        logits_c, logits_l = self(seq[:, canvas])
        lp_c = -loss_rec(logits_c[:, slot, :], seq[:, rest],
            self.vocab.pad, self.args.label_smoothing)
        lp_l = F.log_softmax(logits_l, dim=1)[:, slot].mean()
        return lp_c + lp_l

    def loss(self, seq):
        n = seq.size(1) - 2
        t = np.random.randint(n+1)
        order = np.random.permutation(n) + 1
        canvas = [0] + sorted(order[:t]) + [n+1]
        if t < n:
            rest = sorted(order[t:])
            slot = []
            j = 0
            for i in rest:
                while canvas[j] < i:
                    j += 1
                slot.append(j-1)
        else:
            rest = [n+1]
            slot = [n]

        return -self.log_prb(seq, canvas, rest, slot)*(n+1) - self.log_factorial[n]
