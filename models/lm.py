import torch
import torch.nn as nn
import pytorch_lightning as pl

from transformer.Models import Encoder
from optimizer import config_opt_schedule
from vocab import Vocab


class LM(pl.LightningModule):
    """Language Model Container Class"""

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.enc = Encoder(
            n_src_vocab=hparams.vocab_size, len_max_seq=hparams.max_len,
            d_word_vec=hparams.d_model, d_model=hparams.d_model,
            d_inner=hparams.d_inner_hid, d_k=hparams.d_k, d_v=hparams.d_v,
            n_layers=hparams.n_layers, n_head=hparams.n_head,
            dropout=hparams.dropout)

        self.word = nn.Linear(hparams.d_model, hparams.vocab_size, bias=False)
        nn.init.xavier_normal_(self.word.weight)
        self.x_logit_scale = 1.
        if hparams.share_emb_prj_weight:
            self.word.weight = self.enc.src_word_emb.weight
            self.x_logit_scale = (hparams.d_model ** -0.5)

        self.loc = nn.Linear(hparams.d_model, 1)

    def configure_optimizers(self):
        return config_opt_schedule(self.parameters(), self.hparams)

    def training_step(self, batch, batch_idx):
        seq, n, n_real = map(lambda x: x.squeeze(0), batch)
        losses = self('losses', seq, n, n_real)
        return {**losses, 'log': {**losses}}

    def eval_step(self, batch, batch_idx):
        seq, n, n_real = map(lambda x: x.squeeze(0), batch)
        losses = self('losses', seq, n, n_real)
        if self.hparams.n_mc > 0:
            nll = self('nll_mc', seq, n, self.hparams.n_mc).sum()
        else:
            nll = losses['loss'] * n_real.sum()
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
        pos[canvas == Vocab.pad] = 0
        output, *_ = self.enc(canvas, pos.to(canvas.device))
        return output

    def forward(self, action, *args):
        if action == 'nll_mc':
            return self.nll_mc(*args)
        elif action == 'losses':
            return self.losses(*args)
        raise NotImplementedError
