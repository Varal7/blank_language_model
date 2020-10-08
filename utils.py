import os
import re


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


def parse_epoch(filename):
    m = re.search("_ckpt_epoch_(.*).ckpt", filename)
    if m is None:
        return -1
    return int(m.group(1))


def get_last_model_path(dir):
    last, epoch = sorted([(filename, parse_epoch(filename)) for filename in os.listdir(dir)], key=lambda x: -x[1])[0]
    path = os.path.join(dir, last)
    return epoch, path
