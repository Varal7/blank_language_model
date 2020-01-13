import sys
import gzip
from nltk import word_tokenize

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)

file = sys.argv[1]
with open(file + '.txt', 'w') as f:
    for data in parse(file + '.json.gz'):
        text = word_tokenize(data['reviewText'])
        f.write(' '.join(text) + '\n')
