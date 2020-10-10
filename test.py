import argparse
import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from vocab import Vocab
from utils import strip_eos, load_model
from dataset import load_data, get_eval_dataloader, load_sent


def select(logits, decode):
    if decode == 'sample':
        return torch.multinomial(logits.exp(), num_samples=1)[0]
    else:
        return logits.argmax()


def generate(seq, model, device, decode):
    seq = torch.LongTensor(seq).to(device)
    blanks = [i for i, w in enumerate(seq) if w == Vocab.blank]
    is_fill = [0] * len(seq)
    fill = [[id for id, isf in zip(seq, is_fill) if isf]]
    full = [seq]
    while len(blanks) > 0 and len(seq) <= model.hparams.max_len:
        output = model.forward_encoder(seq.unsqueeze(0))[0]
        output_blank = output[blanks]
        loc = select(model.loc(output_blank).squeeze(-1), decode)
        output_loc = output_blank[loc]

        logits_word = model.word(output_loc) * model.x_logit_scale
        logits_word[Vocab.blank] = float('-inf')    # never predict <blank>

        # joint word, lrb prediction
        lprob_word = F.log_softmax(logits_word, -1)
        output_word = torch.cat((output_loc.unsqueeze(0).expand(model.hparams.vocab_size, -1),
                                 model.enc.src_word_emb.weight), -1)
        logits_lrb = model.lrb(output_word)
        lprob_lrb = F.log_softmax(logits_lrb, -1)
        lprob_word_lrb = lprob_word.unsqueeze(1) + lprob_lrb
        word_lrb = select(lprob_word_lrb.view(-1), decode)
        word, lrb = word_lrb // 4, word_lrb % 4

        # predict word first and then lrb
        # word = select(logits_word, decode)
        # output_word = torch.cat((output_loc, model.enc.src_word_emb(word)), dim=-1)
        # lrb = select(model.lrb(output_word), decode)

        lb, rb = lrb // 2, lrb % 2
        ins = ([Vocab.blank] if lb else []) + [word] + ([Vocab.blank] if rb else [])
        ins = torch.LongTensor(ins).to(device)
        pos = blanks[loc]
        seq = torch.cat((seq[:pos], ins, seq[pos + 1:]))
        blanks = [i for i, w in enumerate(seq) if w == Vocab.blank]
        is_fill = is_fill[:pos] + [1] * len(ins) + is_fill[pos + 1:]
        fill.append([id for id, isf in zip(seq, is_fill) if isf])
        full.append(seq)
    return fill, full


def makedir(path):
    dir = os.path.dirname(path)
    if dir:
        os.makedirs(dir, exist_ok=True)


def write(file, sents, write_mid):
    sents = strip_eos(sents)
    if write_mid:
        for s in sents:
            file.write(' '.join(s) + '\n')
        file.write('\n')
    else:
        file.write(' '.join(sents[-1]) + '\n')
    file.flush()


def main(args):
    pl.seed_everything(args.seed)

    model = load_model(args.checkpoint).to(device)
    model.eval()
    vocab = Vocab(os.path.join(model.hparams.root_dir, 'vocab.txt'))

    if args.eval:
        data = load_data(args.eval, model.hparams.add_eos, model.hparams.cat_sent, model.hparams.max_len)
        dl = get_eval_dataloader(
            data, vocab, args.max_tok,
            data_workers=args.data_workers,
            model_type=model.hparams.model_type)
        trainer = pl.Trainer(
            gpus=args.gpus,
            amp_level=args.fp16_opt_level,
            precision=16 if args.fp16 else 32,
            default_root_dir='testing_logs')
        trainer.test(model, test_dataloaders=dl)

    if args.output:
        output = os.path.join(os.path.dirname(os.path.dirname(args.checkpoint)), 'outputs/', args.output)
        makedir(output)

    if args.sample:
        with open(output, 'w') as f:
            for i in tqdm(range(args.sample)):
                _, full = generate([Vocab.blank], model, device, args.decode)
                full = [[vocab.idx2word[id] for id in ids] for ids in full]
                write(f, full, args.write_mid)

    if args.fill:
        sents = load_sent(args.fill, model.hparams.add_eos)
        sents = [[vocab.word_to_idx(w) for w in s] for s in sents]
        with open(output + '.fill', 'w') as f_fill:
            with open(output + '.full', 'w') as f_full:
                for s in tqdm(sents):
                    fill, full = generate(s, model, device, args.decode)
                    fill = [[vocab.idx2word[id] for id in ids] for ids in fill]
                    full = [[vocab.idx2word[id] for id in ids] for ids in full]
                    write(f_fill, fill, args.write_mid)
                    write(f_full, full, args.write_mid)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint', required=True,
                        help='path to checkpoint')

    parser.add_argument('--eval', default='',
                        help='data file to evaluate')
    parser.add_argument('--n_mc', type=int, default=100,
                        help='num of samples for monte carlo estimate of ppl')
    parser.add_argument('--max_tok', type=int, default=40000,
                        help='max number of tokens per batch')

    parser.add_argument('--output', default='',
                        help='output file')
    parser.add_argument('--sample', type=int, default=0,
                        help='num of sentences to generate')
    parser.add_argument('--fill', default='',
                        help='input file to fill')
    parser.add_argument('--decode', default='greedy',
                        choices=['greedy', 'sample'],
                        help='greedy decoding or sampling')
    parser.add_argument('--write_mid', action='store_true',
                        help='write intermediate partial sentences')

    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--data_workers', type=int, default=8,
                        help='data workers')
    parser.add_argument('--no_cuda', action='store_true',
                        help='disable CUDA')
    parser.add_argument('--fp16', action='store_true',
                        help='whether to use 16-bit (mixed) precision '
                             '(through NVIDIA apex) instead of 32-bit')
    parser.add_argument('--fp16_opt_level', default='O1',
                        help="for fp16: Apex AMP optimization level selected "
                             "in ['O0', 'O1', 'O2', and 'O3']. see details at "
                             "https://nvidia.github.io/apex/amp.html")

    args = parser.parse_args()

    cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    args.gpus = 1 if cuda else 0

    main(args)
