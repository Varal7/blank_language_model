import sys
import os
from collections import Counter
from nltk import sent_tokenize

def write(data, path):
    with open(path, 'w') as f:
        for x in data:
            f.write(' '.join(x) + '\n')

def segmentation(path):
    data = []
    with open(path) as f:
        for line in f:
            sents = sent_tokenize(line)
            for s in sents:
                data.append(s.split())
    return data

dir = sys.argv[1]
for file in ['train.txt', 'valid.txt', 'test.txt']:
    data = segmentation(os.path.join(dir, file))
    write(data, os.path.join(dir+'-sent', file))
