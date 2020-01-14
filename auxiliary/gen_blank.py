import sys
import random
from itertools import accumulate

def load_sent(path):
    sents = []
    with open(path) as f:
        for line in f:
            sents.append(line.split())
    return sents

def write_sent(sents, path):
    with open(path, 'w') as f:
        for s in sents:
            f.write(' '.join(s) + '\n')

def process(sent, keep):
    blank, fill = [], []
    for w, k in zip(sent, keep):
        if k:
            blank.append(w)
            if not (len(fill) > 0 and fill[-1] == '<sep>'):
                fill.append('<sep>')
        else:
            if not (len(blank) > 0 and blank[-1] == '<blank>'):
                blank.append('<blank>')
            fill.append(w)
    if len(fill) > 0 and fill[0] == '<sep>':
        fill = fill[1:]
    if len(fill) > 0 and fill[-1] == '<sep>':
        fill = fill[:-1]
    return blank, fill

def partition(m, k):
    ''' randomly partition m into k positive integers '''
    assert m >= k
    a = list(range(1, m))
    random.shuffle(a)
    b = [0] + sorted(a[:k-1]) + [m]
    return [b[i+1] - b[i] for i in range(k)]

def fitin(n, l):
    ''' randomly fit blanks of length l into a sentence of length n '''
    s = partition(n+2-sum(l), len(l)+1)
    keep = [True] * s[0]
    for p, q in zip(l, s[1:]):
        keep += [False] * p + [True] * q
    return keep[1: -1]

def mask_nblanks_ratio(sents, k, r, times, path):
    blank, fill, full = [], [], []
    for _ in range(times):
        for s in sents:
            n = len(s)
            m = int(n*r)
            if m >= k and m+k-1 <= n:
                l = partition(m, k)
                keep = fitin(n, l)
                b, f = process(s, keep)
                blank.append(b)
                fill.append(f)
                full.append(s)
    write_sent(blank, path + '.blank')
    write_sent(fill, path + '.fill')
    write_sent(full, path + '.full')

def mask_nblanks_maxlen(sents, k, max_l, times, path):
    blank, fill, full = [], [], []
    for _ in range(times):
        for s in sents:
            n = len(s)
            l = [random.randint(1, max_l) for _ in range(k)]
            if sum(l)+k-1 <= n:
                keep = fitin(n, l)
                b, f = process(s, keep)
                blank.append(b)
                fill.append(f)
                full.append(s)
    write_sent(blank, path + '.blank')
    write_sent(fill, path + '.fill')
    write_sent(full, path + '.full')

def mask_uni_len(sents, times, path):
    blank, fill = [], []
    for _ in range(times):
        for sent in sents:
            n = len(sent)
            k = random.randint(0, n)
            order = list(range(n))
            random.shuffle(order)
            keep = [order[i] < k for i in range(n)]
            b, f = process(sent, keep)
            blank.append(b)
            fill.append(f)
    write_sent(blank, path + '.blank')
    write_sent(fill, path + '.fill')

def main():
    random.seed(1)

    path = sys.argv[1]
    sents = load_sent(path + '.txt')

    times = 1
    if len(sys.argv) > 2:
        times = int(sys.argv[2])
        path += '.times%d' % times

    #for k in [1, 2, 3]:
    #    for r in [0.25, 0.50, 0.75]:
    #        mask_nblanks_ratio(sents, k, r, times, path + '.blank%d.maskratio%.2f' % (k, r))

    for k in [1, 2, 3]:
        for l in [1, 5, 10]:
            mask_nblanks_maxlen(sents, k, l, times, path + '.blank%d.maxlen%d' % (k, l))

    #mask_uni_len(sents, times, path + '.uni_len')

if __name__ == '__main__':
    main()
