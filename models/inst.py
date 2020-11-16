import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . lm import LM
from . torch_utils import get_ins_canvas, sample_permutation, seq_cross_entropy, collect, batch_randint, new_arange, select
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
        k = batch_randint(0, n)
        rank = sample_permutation(seq)
        keep = (rank < (k + 2).unsqueeze(1))    # keep <first> and <last> in addition
        canvas, rest, loc = get_ins_canvas(seq, keep, n)

        # canvas has <first> + k tokens + <last>, so k + 1 slots
        mask = (new_arange(canvas) < (k + 1).unsqueeze(1))[:, :-1]  # mask for logits_loc
        loss_loc, loss_word = self.get_loss(seq, canvas, rest, loc, mask)
        nll_lb = (loss_loc + loss_word) * (n + 1).float() - (n + 1).float().lgamma()
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

    def generate(self, seq, blanks, decode, device, force_insert=False, prioritize_unfilled=False):
        seq = torch.LongTensor([Vocab.first] + seq + [Vocab.last]).to(device)
        is_fill = [0] * len(seq)
        fill = [[]]
        full = [seq[1:-1]]
        mandatory_blanks = np.array(blanks)
        if len(blanks) > 0:
            while len(seq) < self.hparams.max_len:
                output = self.forward_encoder(seq.unsqueeze(0))
                features = self.pool_out(
                    torch.cat((output[:, :-1, :], output[:, 1:, :]), dim=-1)
                )[0]

                logits_loc = self.loc(features).squeeze(-1)

                all_filled = (mandatory_blanks == None).all()
                can_stop = not force_insert or all_filled
                end_slot = len(seq) - 2

                if prioritize_unfilled and not all_filled:
                    logits_loc[np.array(blanks)[mandatory_blanks == None]] = float('-inf')

                if end_slot not in blanks and can_stop:   # enable end slot for termination
                    blanks_end = blanks + [end_slot]
                    loc = select(logits_loc[blanks_end], decode)
                    pos = blanks_end[loc]
                else:
                    loc = select(logits_loc[blanks], decode)
                    pos = blanks[loc]

                output_loc = features[pos]
                logits_word = self.word(output_loc) * self.x_logit_scale

                if pos == end_slot:
                    if end_slot not in blanks:  # end slot is added artificially, so no words allowed there
                        break
                    elif not can_stop:
                        logits_word[Vocab.last] = float('-inf')

                word = select(logits_word, decode)

                if pos == end_slot and word.item() == Vocab.last:
                    break

                blanks = blanks[:loc + 1] + [x + 1 for x in blanks[loc:]]
                mandatory_blanks = np.concatenate((
                    mandatory_blanks[:loc],
                    np.array([None]),
                    np.array([None]),
                    [x + 1 if x is not None else None for x in mandatory_blanks[loc + 1:]]
                ))
                seq = torch.cat((seq[:pos + 1], word.unsqueeze(0), seq[pos + 1:]))
                is_fill = is_fill[:pos + 1] + [1] + is_fill[pos + 1:]
                fill.append([id for id, isf in zip(seq, is_fill) if isf])
                full.append(seq[1:-1])
        return fill, full
