import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . lm import LM
from utils import get_known_length_canvas, sample_permutation, seq_cross_entropy, collect, batch_randint


class LBLM(LM):
    """Length-aware Blank Language Model"""

    def __init__(self, vocab, hparams):
        super().__init__(vocab, hparams)

        self.lrb = nn.Sequential(
            nn.Linear(hparams.d_model * 2, hparams.d_model * 2),
            nn.ReLU(),
            nn.Linear(hparams.d_model * 2, hparams.max_len)
        )

    def get_loss(self, seq, canvas, blanks, rest, loc, lb):
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
        target = collect(seq, rest, self.vocab.pad)
        loss_word = seq_cross_entropy(logits_word, target, self.vocab.pad)
        loss_word = loss_word.sum(1) / count.float()
        output_word = torch.cat((output_loc, self.enc.src_word_emb(target)), -1)

        logits_lrb = self.lrb(output_word)

        # Mask out illegal blank options
        blank_0 = self.vocab.word2idx['<blank_0>']
        length = collect(canvas, blanks) - blank_0
        length_loc = collect(length, loc, -1)
        bs, seq_len = length_loc.shape
        ta = length_loc.unsqueeze(-1).repeat(1, 1, self.hparams.max_len)
        ra = torch.arange(self.hparams.max_len).unsqueeze(0).unsqueeze(0).repeat(bs, seq_len, 1).to(ta.device)
        mask = (ra >= ta)
        logits_lrb.masked_fill_(mask, float('-inf'))

        loss_lrb = seq_cross_entropy(logits_lrb, lb, -1)
        loss_lrb = loss_lrb.sum(1) / count.float()

        return loss_loc, loss_word, loss_lrb

    def losses(self, seq, n, n_real):
        """
        Args:
            n: number of BPE tokens
            n_real: number of real words (for reporting PPL)
        """
        m = (seq == self.vocab.missing).sum(1)
        # m = torch.max(m, n - 10)
        k = batch_randint(m, n - 1)
        rank = sample_permutation(seq, self.vocab)
        keep = (rank < k.unsqueeze(1))
        canvas, blanks, rest, loc, lb = get_known_length_canvas(seq, keep, n, self.vocab)
        loss_loc, loss_word, loss_lrb = self.get_loss(seq, canvas, blanks, rest, loc, lb)
        nll_lb = (loss_loc + loss_word + loss_lrb) * (n - m).float() - (n - m + 1).float().lgamma()
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
        raise NotImplementedError
