import os
import yaml

from models import get_model_class


def strip_eos(sents):
    return [sent[:sent.index('<eos>')] if '<eos>' in sent else sent
            for sent in sents]


def makedir(path):
    dir = os.path.dirname(path)
    if dir:
        os.makedirs(dir, exist_ok=True)


def repeat(f, x, n):
    for i in range(n):
        x = f(x)
    return x


def get_hparams(checkpoint):
    hparams_file = os.path.join(os.path.dirname(os.path.dirname(checkpoint)), 'hparams.yaml')
    with open(hparams_file) as stream:
        return yaml.safe_load(stream)


def load_model(checkpoint):
    hparams = get_hparams(checkpoint)
    model = get_model_class(hparams['model_type']).load_from_checkpoint(checkpoint)
    model.hparams.root_dir = repeat(lambda x: os.path.dirname(x), checkpoint, 4)
    return model


def load_sent(path, add_eos=False):
    sents = []
    with open(path) as f:
        for line in f:
            s = line.split()
            if add_eos:
                s += ['<eos>']
            sents.append(s)
    return sents


def load_data(path, add_eos=False, cat_sent=False, max_len=512):
    if not add_eos:
        print('WARNING: You should always use add_eos to get comparable PPL on'
              'language modeling tasks')

    sents = load_sent(path, add_eos)
    if cat_sent:
        if not add_eos:
            raise ValueError('Using cat_sent without add_eos')
        d = [w for s in sents for w in s]
        data = [d[i: i + max_len] for i in range(0, len(d), max_len)]
    else:
        print('# truncated sentences:',
              sum(1 for s in sents if len(s) > max_len))
        data = [s[:max_len] for s in sents]
    return data


def write(file, sents, write_mid):
    sents = strip_eos(sents)
    if write_mid:
        for s in sents:
            file.write(' '.join(s) + '\n')
        file.write('\n')
    else:
        file.write(' '.join(sents[-1]) + '\n')
    file.flush()
