import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . lm import LM
from . torch_utils import get_canvas, sample_permutation, seq_cross_entropy, collect, batch_randint
from vocab import Vocab


class BLM(LM):
    """Blank Language Model"""

    def __init__(self, hparams):
        super().__init__(hparams)

        self.lrb = nn.Sequential(
            nn.Linear(hparams.d_model * 2, hparams.d_model * 2),
            nn.ReLU(),
            nn.Linear(hparams.d_model * 2, 4)
        )

    def get_loss(self, seq, canvas, blanks, rest, loc, lb, rb):
        count = (rest != -1).sum(1)
        output = self.forward_encoder(canvas)
        output_blank = collect(output, blanks)

        logits_loc = self.loc(output_blank).squeeze(-1)
        logits_loc[blanks == -1] = float('-inf')
        nll_loc = -F.log_softmax(logits_loc, 1)
        loss_loc = collect(nll_loc, loc)
        loss_loc = loss_loc.sum(1) / count.float()
        output_loc = collect(output_blank, loc)

        logits_word = self.word(output_loc) * self.x_logit_scale
        target = collect(seq, rest, Vocab.pad)
        loss_word = seq_cross_entropy(logits_word, target, Vocab.pad)
        loss_word = loss_word.sum(1) / count.float()
        output_word = torch.cat((output_loc, self.enc.src_word_emb(target)), -1)

        logits_lrb = self.lrb(output_word)
        loss_lrb = seq_cross_entropy(logits_lrb, lb * 2 + rb, -3)
        loss_lrb = loss_lrb.sum(1) / count.float()

        return loss_loc, loss_word, loss_lrb

    def losses(self, seq, n, n_real):
        """
        Args:
            n: number of BPE tokens
            n_real: number of real words (for reporting PPL)
        """
        k = batch_randint(0, n - 1)
        rank = sample_permutation(seq)
        keep = (rank < k.unsqueeze(1))
        canvas, blanks, rest, loc, lb, rb = get_canvas(seq, keep, n)
        loss_loc, loss_word, loss_lrb = self.get_loss(seq, canvas, blanks, rest, loc, lb, rb)
        nll_lb = (loss_loc + loss_word + loss_lrb) * n.float() - (n + 1).float().lgamma()
        return {'loss': nll_lb.sum() / n_real.sum(),
                'loc': loss_loc.mean(),
                'word': loss_word.mean(),
                'lrb': loss_lrb.mean()
                }

    def nll_mc(self, seq, n, m):
        """
        Compute negative log-likelihood by monte carlo estimate
        Args:
            m: number of samples to take

        Note: sentences in the batch must have the same length
        """
        a = []
        for _ in range(m):
            rank = sample_permutation(seq)
            logp = 0.
            for k in range(seq.size(1)):
                keep = (rank < k)
                canvas, blanks, rest, loc, lb, rb = get_canvas(seq, keep, n)
                k_th = (rank == k).nonzero(as_tuple=True)[1]
                x, y = (rest == k_th.unsqueeze(1)).nonzero(as_tuple=True)
                assert torch.all(x == torch.arange(len(seq), device=seq.device))
                rest, loc, lb, rb = [t[x, y].unsqueeze(1) for t in [rest, loc, lb, rb]]
                loss_loc, loss_word, loss_lrb = self.get_loss(seq, canvas, blanks, rest, loc, lb, rb)
                logp -= loss_loc + loss_word + loss_lrb
            a.append(logp.unsqueeze(1))
        return np.log(m) - (n + 1).float().lgamma() - torch.logsumexp(torch.cat(a, 1), 1)
