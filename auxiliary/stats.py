import sys
import os
from collections import Counter

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
    cnt = Counter([w for s in sents for w in s])

    print('\n%s' % file)
    print('num of sents:      %d' % len(sents))
    print('num of tokens:     %d' % sum(lens))
    print('max length:        %d' % max(lens))
    print('avg length:        %.2f' % (sum(lens) / len(lens)))
    print('vocab size (>=1):  %d' % len(cnt))
    print('vocab size (>=5):  %d' % sum([cnt[w] >= 5 for w in cnt]))
