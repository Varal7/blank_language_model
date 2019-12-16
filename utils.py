import random
import numpy as np
import torch

def set_seed(seed):     # set the random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def strip_eos(sents):
    return [sent[:sent.index('<eos>')] if '<eos>' in sent else sent
        for sent in sents]

def load_sent(path):
    sents = []
    with open(path) as f:
        for line in f:
            sents.append(line.split() + ['<eos>'])
    return sents

def load_data(path, doc, seq_len):
    sents = load_sent(path)
    if doc:
        d = [w for s in sents for w in s]
        sents = [d[i: i+seq_len] for i in range(0, len(d), seq_len)]
    return sents

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

def logging(s, path, print_=True):
    if print_:
        print(s)
    if path:
        with open(path, 'a+') as f:
            f.write(s + '\n')
