import sys
import gzip
from nltk import word_tokenize

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)

file = sys.argv[1]
max_len = int(sys.argv[2])
with open(file + '.maxlen%d.txt' % max_len, 'w') as f:
    for data in parse(file + '.json.gz'):
        text = word_tokenize(data['reviewText'])
        if len(text) <= max_len:
            f.write(' '.join(text) + '\n')
