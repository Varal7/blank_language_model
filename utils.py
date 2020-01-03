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

def load_sent(path, add_eos=False):
    sents = []
    with open(path) as f:
        for line in f:
            s = line.split()
            if add_eos:
                s += ['<eos>']
            sents.append(s)
    return sents

def load_data(path, add_eos=False, cat_sent=False, max_len=50):
    sents = load_sent(path, add_eos)
    if cat_sent:
        d = [w for s in sents for w in s]
        sents = [d[i: i+max_len] for i in range(0, len(d), max_len)]
    else:
        n = len(sents)
        sents = [s for s in sents if len(s) <= max_len]
        print('# discarded sents:', n - len(sents))
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
