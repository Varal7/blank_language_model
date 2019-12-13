import random
import numpy as np
import torch

from model import LM

def set_seed(seed):     # set the random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def strip_eos(sents):
    return [sent[:sent.index('<eos>')] if '<eos>' in sent else sent
        for sent in sents]

def merge(sents, max_len):
    data = []
    i = 0
    while i < len(sents):
        sent = sents[i]
        i += 1
        while i < len(sents) and len(sent) + len(sents[i]) < max_len:
            sent += ['<eos>'] + sents[i]
            i += 1
        data.append(sent)
    return data

def load_sent(path, multisent=-1):
    sents = []
    with open(path) as f:
        for line in f:
            sents.append(line.split())
    if multisent == -1:
        return sents
    return merge(sents, multisent)

def write_sent(sents, path):
    with open(path, 'w') as f:
        for s in sents:
            f.write(' '.join(s) + '\n')

def write_doc(docs, path):
    with open(path, 'w') as f:
        for d in docs:
            for s in d:
                f.write(' '.join(s) + '\n')
            f.write('\n')

def write_mid_or_last(sents_mid, write_mid, path):
    if write_mid:
        write_doc(sents_mid, path)
    else:
        sents_last = [s[-1] for s in sents_mid]
        write_sent(sents_last, path)

def logging(s, path, print_=True):
    if print_:
        print(s)
    if path:
        with open(path, 'a+') as f:
            f.write(s + '\n')

def get_model(path, vocab, device):
    print('Load model from {}'.format(path))
    ckpt = torch.load(path)
    train_args = ckpt['args']
    model = LM(vocab, train_args).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model
