import os
import yaml

from models import get_model_class


def get_hparams(checkpoint):
    hparams_file = os.path.join(os.path.dirname(os.path.dirname(checkpoint)), 'hparams.yaml')
    with open(hparams_file) as stream:
        return yaml.safe_load(stream)


def load_model(checkpoint):
    hparams = get_hparams(checkpoint)
    model = get_model_class(hparams['model_type']).load_from_checkpoint(checkpoint)
    return model


def strip_eos(sents):
    return [sent[:sent.index('<eos>')] if '<eos>' in sent else sent
            for sent in sents]


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


class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)
