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

def split(m, k):
    ''' randomly generate k positive integers that sum up to m '''
    assert m >= k
    a = list(range(1, m))
    random.shuffle(a)
    b = [0] + sorted(a[:k-1]) + [m]
    return [b[i+1] - b[i] for i in range(k)]

def mask_nblanks_ratio(sents, k, r, times, path):
    blank, fill = [], []
    for _ in range(times):
        for sent in sents:
            n = len(sent)
            m = int(n*r)
            if m >= k and m+k-1 <= n:
                l = split(m, k)
                ls = list(accumulate(l[::-1]))[::-1]
                keep = [True] * n
                i = 0
                for j in range(k):
                    q = n - ls[j] - (k-j-1)
                    p = random.randint(i, q)
                    for x in range(l[j]):
                        keep[p+x] = False
                    i = p+l[j]+1
                b, f = process(sent, keep)
                blank.append(b)
                fill.append(f)
    write_sent(blank, path + '.blank')
    write_sent(fill, path + '.fill')

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

    for k in [1, 2, 3]:
        for r in [0.25, 0.50, 0.75]:
            mask_nblanks_ratio(sents, k, r, times, path + '.blank%d.maskratio%.2f' % (k, r))

    mask_uni_len(sents, times, path + '.uni_len')

if __name__ == '__main__':
    main()
