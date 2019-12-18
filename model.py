import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.cpp_extension import load

from transformer.Models import Encoder
from transformer.Optim import *
from meters import StopwatchMeter

get_canvas_cpp = load(name="get_canvas_cpp", sources=["get_canvas.cpp"])

def seq_cross_entropy(pred, gold, pad, smoothing=False):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold_shape = gold.shape
    pred = pred.view(-1, pred.size(-1))
    gold = gold.view(-1)

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
        loss = F.cross_entropy(pred, gold, ignore_index=pad, reduction='none')

    return loss.view(gold_shape)

def to_tensor(x, pad_id, device):
    max_len = max([len(xi) for xi in x])
    x_ = [xi + [pad_id] * (max_len - len(xi)) for xi in x]
    return torch.tensor(x_).to(device)

def sample_permutation(seq, pad_id):
    score = torch.rand_like(seq.float())
    score.masked_fill_(seq == pad_id, 1) # always put pads last
    indices = score.argsort()
    rank = torch.zeros_like(seq)
    rank[torch.arange(len(seq)).unsqueeze(1), indices] = \
        torch.arange(seq.size(1), device=seq.device)
    return rank

def get_canvas(seq, keep, n, vocab):
    res = get_canvas_cpp.get_canvas(seq.tolist(), keep.tolist(), n.tolist(), vocab.blank)
    pad = [vocab.pad, -1, -1, -1, -1, -1]
    for i in range(len(res)):
        res[i] = to_tensor(res[i], pad[i], seq.device)
    return res

def collect(input, index, padding_idx=0):
    """
    Performs a batched index select where index is given for each example
    Args:
        input: tensor of shape (B, T_1, dim_2, dim_3, ...)
        index: tensor of shape (B, T_2)
    Returns:
        tensor of shape (B, T_2, dim_2, dim_3, ...)
    """
    # Add a column of padding_idx at index 0 (of dim 1)
    view = list(input.shape)
    view[1] = 1
    padding_column = input.new_ones(view) * padding_idx
    input = torch.cat([padding_column, input], 1)

    # Expand index to compatible size for gather
    for i in range(2, len(input.shape)):
        index = index.unsqueeze(i)

    view[0] = -1
    view[1] = -1
    index = index.expand(view)
    return torch.gather(input, 1, index + 1)


class LM(nn.Module):
    """Language Model"""

    def __init__(self, vocab, args):
        super().__init__()
        self.vocab = vocab
        self.args = args

        self.G = Encoder(
            n_src_vocab=vocab.size, len_max_seq=args.max_len,
            d_word_vec=args.d_model, d_model=args.d_model, d_inner=args.d_inner_hid,
            n_layers=args.n_layers, n_head=args.n_head, d_k=args.d_k, d_v=args.d_v,
            dropout=args.dropout)

        self.word = nn.Linear(args.d_model, vocab.size, bias=False)
        nn.init.xavier_normal_(self.word.weight)
        self.x_logit_scale = 1.
        if args.share_emb_prj_weight:
            self.word.weight = self.G.src_word_emb.weight
            self.x_logit_scale = (args.d_model ** -0.5)

        self.loc = nn.Linear(args.d_model, 1)
        self.lrb = nn.Sequential(nn.Linear(args.d_model*2, args.d_model*2),
            nn.ReLU(), nn.Linear(args.d_model*2, 4))

        opt = optim.Adam(self.parameters(), betas=eval(args.adam_betas),
            eps=args.adam_eps, weight_decay=args.weight_decay)
        if args.lr_schedule == 'inverse_sqrt':
            self.opt = InverseSqrtScheduler(opt, args.lr, args.warmup_steps)
        elif args.lr_schedule == 'linear_decay':
            self.opt = LinearDecayScheduler(opt, args.lr, args.warmup_steps, args.train_steps)
        else:
            self.opt = LRScheduler(opt, args.lr)

    def forward(self, canvas):
        pos = (1 + torch.arange(canvas.size(1))).repeat(len(canvas), 1)
        pos[canvas == self.vocab.pad] = 0
        output, *_ = self.G(canvas, pos.to(canvas.device))
        return output

    def get_loss(self, seq, canvas, blanks, rest, loc, lb, rb):
        count = (rest != -1).sum(1)
        output = self(canvas)
        output_blank = collect(output, blanks)

        logits_loc = self.loc(output_blank).squeeze(-1)
        logits_loc[blanks == -1] = float('-inf')
        nll_loc = -F.log_softmax(logits_loc, 1)
        loss_loc = collect(nll_loc, loc)
        loss_loc = loss_loc.sum(1) / count.float()
        output_loc = collect(output_blank, loc)

        logits_word = self.word(output_loc) * self.x_logit_scale
        target = collect(seq, rest, self.vocab.pad)
        loss_word = seq_cross_entropy(logits_word, target, self.vocab.pad,
            self.args.label_smoothing)
        loss_word = loss_word.sum(1) / count.float()
        output_word = torch.cat((output_loc, self.G.src_word_emb(target)), -1)

        logits_lrb = self.lrb(output_word)
        loss_lrb = seq_cross_entropy(logits_lrb, lb * 2 + rb, -3)
        loss_lrb = loss_lrb.sum(1) / count.float()

        return loss_loc, loss_word, loss_lrb

    def losses(self, seq, n):
        k = (torch.rand_like(n.float()) * n.float()).long() # sample k from 0 to n-1
        rank = sample_permutation(seq, self.vocab.pad)
        keep = (rank < k.unsqueeze(1))
        canvas, blanks, rest, loc, lb, rb = get_canvas(seq, keep, n, self.vocab)
        loss_loc, loss_word, loss_lrb = self.get_loss(seq, canvas, blanks, rest, loc, lb, rb)
        nll_lb = (loss_loc + loss_word + loss_lrb) * n.float() - (n + 1).float().lgamma()
        return {'loss' : nll_lb.sum() / n.sum(),
                'loc'  : loss_loc.mean(),
                'word' : loss_word.mean(),
                'lrb'  : loss_lrb.mean()
               }

    def nll_mc(self, seq, n, m):
        """Compute negative log-likelihood by monte carlo estimate
           m: number of samples to take
        """
        a = []
        for _ in range(m):
            rank = sample_permutation(seq, self.vocab.pad)
            for k in range(seq.size(1)):
                if k == 25:
                    import pdb; pdb.set_trace()
                keep = (rank < k)
                canvas, blanks, rest, loc, lb, rb = get_canvas(seq, keep, n, self.vocab)
                k_th = (rank == k).nonzero(as_tuple=True)[1]
                x, y = (rest == k_th.unsqueeze(1)).nonzero(as_tuple=True)
                rest = rest[x, y].unsqueeze(1)
                loc = loc[x, y].unsqueeze(1)
                lb = lb[x, y].unsqueeze(1)
                rb = rb[x, y].unsqueeze(1)
