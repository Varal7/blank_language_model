import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . lm import LM
from . torch_utils import get_ins_canvas, sample_permutation, seq_cross_entropy, collect, batch_randint, new_arange
from vocab import Vocab


class InsTLM(LM):
    """Insertion Transformer Language Model"""

    def __init__(self, hparams):
        super().__init__(hparams)
        hparams = self.hparams  # a['key'] (if so) -> a.key

        self.pool_out = nn.Linear(2 * hparams.d_model, hparams.d_model)

    def get_loss(self, seq, canvas, rest, loc, mask):
        count = (rest != -1).sum(1)
        output = self.forward_encoder(canvas)
        features = self.pool_out(torch.cat((output[:, :-1, :], output[:, 1:, :]), dim=-1))
        logits_loc = self.loc(features).squeeze(-1)
        logits_loc[~mask] = float('-inf')
        nll_loc = -F.log_softmax(logits_loc, 1)
        loss_loc = collect(nll_loc, loc)
        loss_loc = loss_loc.sum(1) / count.float()
        output_loc = collect(features, loc)

        logits_word = self.word(output_loc) * self.x_logit_scale
        target = collect(seq, rest, Vocab.pad)
        loss_word = seq_cross_entropy(logits_word, target, Vocab.pad)
        loss_word = loss_word.sum(1) / count.float()
        # output_word = torch.cat((output_loc, self.enc.src_word_emb(target)), -1)
        return loss_loc, loss_word

    def losses(self, seq, n, n_real):
        """
        Args:
            n: number of BPE tokens
            n_real: number of real words (for reporting PPL)
        """
        m = (seq == Vocab.missing).sum(1)
        k = batch_randint(m, n)
        rank = sample_permutation(seq)
        keep = (rank < (k + 2).unsqueeze(1))    # keep <first>, <last> and k tokens with k >= m
        canvas, rest, loc = get_ins_canvas(seq, keep, n)

        # canvas has <first> + k tokens + <last>, so k + 1 slots
        mask = (new_arange(canvas) < (k + 1).unsqueeze(1))[:, :-1]  # mask for logits_loc
        loss_loc, loss_word = self.get_loss(seq, canvas, rest, loc, mask)
        nll_lb = (loss_loc + loss_word) * (n - m + 1).float() - (n - m + 1).float().lgamma()
        return {'loss': nll_lb.sum() / n_real.sum(),
                'loc': loss_loc.mean(),
                'word': loss_word.mean(),
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
            for k in range(2, seq.size(1) + 1):     # k from 2 to n + 2
                keep = (rank < k)
                canvas, rest, loc = get_ins_canvas(seq, keep, n)
                if k == seq.size(1):
                    pass    # rest and loc are already correct
                else:
                    k_th = (rank == k).nonzero(as_tuple=True)[1]    # First token not kept
                    x, y = (rest == k_th.unsqueeze(1)).nonzero(as_tuple=True)
                    assert len(seq) == len(x)
                    assert torch.all(x == torch.arange(len(seq), device=seq.device))
                    rest, loc = [t[x, y].unsqueeze(1) for t in [rest, loc]]
                mask = (new_arange(canvas) < (k - 1))[:, :-1]   # mask for logits_loc
                loss_loc, loss_word = self.get_loss(seq, canvas, rest, loc, mask)
                logp -= loss_loc + loss_word
            a.append(logp.unsqueeze(1))
        return np.log(m) - (n + 1).float().lgamma() - torch.logsumexp(torch.cat(a, 1), 1)
