import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . lm import LM
from . torch_utils import get_known_length_canvas, sample_permutation, seq_cross_entropy, collect, batch_randint, select
from vocab import Vocab


class LBLM(LM):
    """Length-aware Blank Language Model"""

    def __init__(self, hparams):
        super().__init__(hparams)
        hparams = self.hparams  # a['key'] (if so) -> a.key

        self.lrb = nn.Sequential(
            nn.Linear(hparams.d_model * 2, hparams.d_model * 2),
            nn.ReLU(),
            nn.Linear(hparams.d_model * 2, hparams.max_len)
        )

    def blank_indices(self):
        return Vocab.blank_0 + np.arange(self.hparams.max_len)

    def init_canvas(self):
        return np.random.choice(self.blank_indices()[1:])   # no blank_0

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
        target = collect(seq, rest, Vocab.pad)
        loss_word = seq_cross_entropy(logits_word, target, Vocab.pad)
        loss_word = loss_word.sum(1) / count.float()
        output_word = torch.cat((output_loc, self.enc.src_word_emb(target)), -1)

        logits_lrb = self.lrb(output_word)

        # mask out illegal blank options
        length = collect(canvas, blanks) - Vocab.blank_0
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
        m = (seq == Vocab.missing).sum(1)
        # m = torch.max(m, n - 10)
        k = batch_randint(m, n - 1)
        rank = sample_permutation(seq)
        keep = (rank < k.unsqueeze(1))
        canvas, blanks, rest, loc, lb = get_known_length_canvas(seq, keep, n)
        loss_loc, loss_word, loss_lrb = self.get_loss(seq, canvas, blanks, rest, loc, lb)
        nll_lb = (loss_loc + loss_word + loss_lrb) * (n - m).float() - (n - m + 1).float().lgamma()
        return {'loss': nll_lb.sum() / n_real.sum(),
                'loc': loss_loc.mean(),
                'word': loss_word.mean(),
                'lrb': loss_lrb.mean()
                }

    # lower than real perplexity since conditioned on length
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
                canvas, blanks, rest, loc, lb = get_known_length_canvas(seq, keep, n)
                k_th = (rank == k).nonzero(as_tuple=True)[1]
                x, y = (rest == k_th.unsqueeze(1)).nonzero(as_tuple=True)
                assert torch.all(x == torch.arange(len(seq), device=seq.device))
                rest, loc, lb = [t[x, y].unsqueeze(1) for t in [rest, loc, lb]]
                loss_loc, loss_word, loss_lrb = self.get_loss(seq, canvas, blanks, rest, loc, lb)
                logp -= loss_loc + loss_word + loss_lrb
            a.append(logp.unsqueeze(1))
        return np.log(m) - (n + 1).float().lgamma() - torch.logsumexp(torch.cat(a, 1), 1)

    def generate(self, seq, decode, device):
        seq = torch.LongTensor(seq).to(device)
        blanks = [i for i, w in enumerate(seq) if w.item() in self.blank_indices()]
        is_fill = [0] * len(seq)
        fill = [[id for id, isf in zip(seq, is_fill) if isf]]
        full = [seq]
        while len(blanks) > 0 and len(seq) <= self.hparams.max_len:
            output = self.forward_encoder(seq.unsqueeze(0))[0]
            output_blank = output[blanks]
            loc = select(self.loc(output_blank).squeeze(-1), decode)
            output_loc = output_blank[loc]

            length_previous = seq[blanks[loc]] - Vocab.blank_0

            logits_word = self.word(output_loc) * self.x_logit_scale
            logits_word[self.blank_indices()] = float('-inf')    # never predict <blank_t>

            # joint word, lrb prediction
            lprob_word = F.log_softmax(logits_word, -1)
            output_word = torch.cat((output_loc.unsqueeze(0).expand(self.hparams.vocab_size, -1),
                                     self.enc.src_word_emb.weight), -1)
            logits_lrb = self.lrb(output_word)
            logits_lrb[:, length_previous:] = float('-inf')     # mask out illegal blank options
            max_blank_len = logits_lrb.shape[1]
            lprob_lrb = F.log_softmax(logits_lrb, -1)
            lprob_word_lrb = lprob_word.unsqueeze(1) + lprob_lrb
            word_lrb = select(lprob_word_lrb.view(-1), decode)
            word, lrb = word_lrb // max_blank_len, word_lrb % max_blank_len

            lb = lrb
            rb = length_previous - lb - 1

            ins = ([Vocab.blank_0 + lb] if lb else []) + [word] + ([Vocab.blank_0 + rb] if rb else [])
            ins = torch.LongTensor(ins).to(device)
            pos = blanks[loc]
            seq = torch.cat((seq[:pos], ins, seq[pos + 1:]))
            blanks = [i for i, w in enumerate(seq) if w.item() in self.blank_indices()]
            is_fill = is_fill[:pos] + [1] * len(ins) + is_fill[pos + 1:]
            fill.append([id for id, isf in zip(seq, is_fill) if isf])
            full.append(seq)
        return fill, full
