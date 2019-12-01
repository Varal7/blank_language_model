import sys
import os

def load_sent(path):
    sents = []
    with open(path) as f:
        for line in f:
            sents.append(line.split())
    return sents

dir = sys.argv[1]
for file in ['train.txt', 'valid.txt', 'test.txt']:
    sents = load_sent(os.path.join(dir, file))
    lens = [len(s) for s in sents]
    print('\n%s' % file)
    print('num of sents:\t %d' % len(sents))
    print('max length:\t %d' % max(lens))
    print('avg length:\t %.2f' % (sum(lens) / len(lens)))
