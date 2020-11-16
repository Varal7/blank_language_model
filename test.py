import argparse
import os
from tqdm import tqdm
import torch
import pytorch_lightning as pl

from vocab import Vocab
from utils import load_data, load_sent, load_model, makedir, write
from dataset import get_eval_dataloader


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
        model.hparams.n_mc = args.n_mc
        trainer.test(model, test_dataloaders=dl)

    if args.output:
        output = os.path.join(os.path.dirname(os.path.dirname(args.checkpoint)), 'outputs/', args.output)
        makedir(output)

    if args.sample:
        with open(output, 'w') as f:
            for i in tqdm(range(args.sample)):
                if model.hparams.model_type == 'inst':
                    _, full = model.generate([], [0], args.decode, device)
                else:
                    _, full = model.generate([model.init_canvas()], args.decode, device)

                full = [[vocab.idx2word[id] for id in ids] for ids in full]
                write(f, full, args.write_mid)

    if args.fill:
        sents = load_sent(args.fill, model.hparams.add_eos)
        sents = [[vocab.word_to_idx(w) for w in s] for s in sents]
        with open(output + '.fill', 'w') as f_fill:
            with open(output + '.full', 'w') as f_full:
                for s in tqdm(sents):
                    if model.hparams.model_type == 'inst':
                        seq, blanks = [], []
                        for w in s:
                            if w == vocab.blank:
                                blanks.append(len(seq))
                            else:
                                seq.append(w)
                        if args.anywhere:
                            blanks = list(range(len(seq) + 1))
                        fill, full = model.generate(seq, blanks, args.decode, device,
                                                    args.force_insert, args.prioritize_unfilled)
                    else:
                        fill, full = model.generate(s, args.decode, device)

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

    # Specific to InsT
    parser.add_argument('--anywhere', action='store_true',
                        help='fill in anywhere, not only blanks')
    parser.add_argument('--force_insert', action='store_true',
                        help='disable termination unless all slots are filled')
    parser.add_argument('--prioritize_unfilled', action='store_true',
                        help='prioritize unfilled slots if any')

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
