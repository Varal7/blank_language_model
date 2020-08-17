import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from transformer.Models import Encoder
from optimizer import config_opt_schedule
from utils import get_known_length_canvas, sample_permutation, seq_cross_entropy, collect, batch_randint


class LBLM(pl.LightningModule):
    """Length-aware Blank Language Model"""

    def __init__(self, vocab, hparams):
        super().__init__()
        self.vocab = vocab
        self.hparams = hparams

        self.G = Encoder(
            n_src_vocab=vocab.size, len_max_seq=hparams.max_len,
            d_word_vec=hparams.d_model, d_model=hparams.d_model,
            d_inner=hparams.d_inner_hid, d_k=hparams.d_k, d_v=hparams.d_v,
            n_layers=hparams.n_layers, n_head=hparams.n_head,
            dropout=hparams.dropout)

        self.word = nn.Linear(hparams.d_model, vocab.size, bias=False)
        nn.init.xavier_normal_(self.word.weight)
        self.x_logit_scale = 1.
        if hparams.share_emb_prj_weight:
            self.word.weight = self.G.src_word_emb.weight
            self.x_logit_scale = (hparams.d_model ** -0.5)

        self.loc = nn.Linear(hparams.d_model, 1)
        self.lrb = nn.Sequential(
            nn.Linear(hparams.d_model * 2, hparams.d_model * 2),
            nn.ReLU(),
            nn.Linear(hparams.d_model * 2, hparams.max_len)
        )

    def configure_optimizers(self):
        return config_opt_schedule(self.parameters(), self.hparams)

    def training_step(self, batch, batch_idx):
        seq, n, n_real = map(lambda x: x.squeeze(0), batch)
        losses = self('losses', seq, n, n_real)
        return {**losses, 'log': {**losses}}

    def eval_step(self, batch, batch_idx):
        seq, n, n_real = map(lambda x: x.squeeze(0), batch)
        # This is not actually compute n_mc, but uses the args.n_mc argument to
        # compute an average of the loss across multiple k / sigma
        nlls = []
        for _ in range(max(1, self.hparams.n_mc)):
            losses = self('losses', seq, n, n_real)
            nll = losses['loss'] * n_real.sum()
            nlls.append(nll)
        nll = torch.tensor(nlls).mean()
        n_words = n_real.sum()
        return {**losses, 'n_words': n_words, 'nll': nll}

    def eval_epoch_end(self, outputs):
        # n_words and nll are batch/dataset sum, other losses are mean
        losses = {}
        for key in outputs[0].keys():
            if key not in ['n_words', 'nll']:
                losses[key] = torch.stack([x[key] for x in outputs]).mean()
        nll = torch.stack([x['nll'] for x in outputs]).sum()
        n_words = torch.stack([x['n_words'] for x in outputs]).sum()
        ppl = torch.exp(nll / n_words)
        return {**losses, 'nll': nll, 'n_words': n_words, 'ppl': ppl}

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        logs = self.eval_epoch_end(outputs)
        val_logs = {'val_' + k: v for k, v in logs.items()}
        return {'val_loss': logs['loss'], 'log': val_logs}

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        logs = self.eval_epoch_end(outputs)
        test_logs = {'test_' + k: v for k, v in logs.items()}
        return {'test_loss': logs['loss'], 'log': test_logs}

    def forward_encoder(self, canvas):
        pos = (1 + torch.arange(canvas.size(1))).repeat(len(canvas), 1)
        pos[canvas == self.vocab.pad] = 0
        output, *_ = self.G(canvas, pos.to(canvas.device))
        return output

    def forward(self, action, *args):
        if action == 'nll_mc':
            return self.nll_mc(*args)
        elif action == 'losses':
            return self.losses(*args)
        raise NotImplementedError

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
        output_word = torch.cat((output_loc, self.G.src_word_emb(target)), -1)

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
