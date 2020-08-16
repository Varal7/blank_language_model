import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from transformer.Models import Encoder
from optimizer import config_opt_schedule
from utils import get_ins_canvas, sample_permutation, \
                  seq_cross_entropy, collect, batch_randint, new_arange


class InsTLM(pl.LightningModule):
    """Insertion Transformer Language Model"""

    def __init__(self, vocab, hparams):
        super().__init__()
        self.vocab = vocab
        self.hparams = hparams

        self.G = Encoder(
            n_src_vocab=vocab.size, len_max_seq=hparams.max_len + 2,
            d_word_vec=hparams.d_model, d_model=hparams.d_model,
            d_inner=hparams.d_inner_hid, d_k=hparams.d_k, d_v=hparams.d_v,
            n_layers=hparams.n_layers, n_head=hparams.n_head,
            dropout=hparams.dropout)

        self.pool_out = nn.Linear(2 * hparams.d_model, hparams.d_model)

        self.word = nn.Linear(hparams.d_model, vocab.size, bias=False)
        nn.init.xavier_normal_(self.word.weight)
        self.x_logit_scale = 1.
        if hparams.share_emb_prj_weight:
            self.word.weight = self.G.src_word_emb.weight
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
        val_logs = {'val_'+k: v for k, v in logs.items()}
        return {'val_loss': logs['loss'], 'log': val_logs}

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        logs = self.eval_epoch_end(outputs)
        test_logs = {'test_'+k: v for k, v in logs.items()}
        return {'test_loss': logs['loss'], 'log': test_logs}

    def forward_encoder(self, canvas):
        pos = (1 + torch.arange(canvas.size(1))).repeat(len(canvas), 1)
        pos[canvas == self.vocab.pad] = 0
        output, *_ = self.G(canvas, pos.to(canvas.device))
        return output

    def forward(self, action, *args):
        if action == "nll_mc":
            return self.nll_mc(*args)
        elif action == "losses":
            return self.losses(*args)
        raise NotImplementedError

    def get_loss(self, seq, canvas, rest, loc, mask):
        count = (rest != -1).sum(1)
        output = self.forward_encoder(canvas)
        features = self.pool_out(
                torch.cat((output[:, :-1, :], output[:, 1:, :]), dim=-1)
        )
        logits_loc = self.loc(features).squeeze(-1)
        logits_loc[~mask] = float('-inf')
        nll_loc = -F.log_softmax(logits_loc, 1)
        loss_loc = collect(nll_loc, loc)
        loss_loc = loss_loc.sum(1) / count.float()
        output_loc = collect(features, loc)

        logits_word = self.word(output_loc) * self.x_logit_scale
        target = collect(seq, rest, self.vocab.pad)
        loss_word = seq_cross_entropy(logits_word, target, self.vocab.pad)
        loss_word = loss_word.sum(1) / count.float()
        # output_word = torch.cat((output_loc, self.G.src_word_emb(target)), -1)
        return loss_loc, loss_word


    def losses(self, seq, n, n_real):
        """
        seq is a tensor of tokens in batch, starting with <first>, endling with <last> and optionally including <eos>
            <first> tok tok ... tok <last> <pad> <pad>
        n is the number of BPE tokens
        n_real is the number of real words
        """
        m = (seq == self.vocab.missing).sum(1)
        #  k = (torch.rand_like(n.float()) * (n + 1).float()).long() # sample k from 0 to n
        k = batch_randint(m, n)

        rank = sample_permutation(seq, self.vocab)
        keep = (rank < (k + 2).unsqueeze(1)) # keep <first>, <last> and k tokens with k >= m
        canvas, rest, loc = get_ins_canvas(seq, keep, n, self.vocab)


        # canvas has <first> + k tokens + <last>, so k + 1 slots
        mask = (new_arange(canvas) < (k + 1).unsqueeze(1))[:, :-1] # mask for logits_loc
        loss_loc, loss_word = self.get_loss(seq, canvas, rest, loc, mask)
        nll_lb = (loss_loc + loss_word) * (n - m + 1).float() - (n - m + 1).float().lgamma()
        return {'loss' : nll_lb.sum() / n_real.sum(),
                'loc'  : loss_loc.mean(),
                'word' : loss_word.mean(),
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
            rank = sample_permutation(seq, self.vocab)
            logp = 0.
            for k in range(2, seq.size(1) + 1): # k from 2 to n + 2
                keep = (rank < k)
                canvas, rest, loc = get_ins_canvas(seq, keep, n, self.vocab)
                if k == seq.size(1):
                    pass # rest and loc are already correct
                else:
                    k_th = (rank == k).nonzero(as_tuple=True)[1] # First token not kept
                    x, y = (rest == k_th.unsqueeze(1)).nonzero(as_tuple=True)
                    assert len(seq) == len(x)
                    assert torch.all(x == torch.arange(len(seq), device=seq.device))
                    rest, loc = [t[x, y].unsqueeze(1) for t in [rest, loc]]
                mask = (new_arange(canvas) < (k - 1))[:, :-1] # mask for logits_loc
                loss_loc, loss_word = self.get_loss(seq, canvas, rest, loc, mask)
                logp -= loss_loc + loss_word
            a.append(logp.unsqueeze(1))
        return np.log(m) - (n + 1).float().lgamma() - torch.logsumexp(torch.cat(a, 1), 1)
