import sys
import random

def load_sent(path):
    sents = []
    with open(path) as f:
        for line in f:
            sents.append(line.split())
    return sents

def write(sents, path):
    with open(path, 'w') as f:
        for s in sents:
            f.write(' '.join(s) + '\n')

def shatter(sent):
    k = random.randint(0, len(sent))
    order = list(range(len(sent)))
    random.shuffle(order)
    keep = order[:k]

    drop, blank, fill = [], [], []
    for i, w in enumerate(sent):
        if i in keep:
            drop.append(w)
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

    return drop, blank, fill

if __name__ == '__main__':
    path = sys.argv[1]
    sents = load_sent(path + '.txt')
    random.seed(1)

    n = 1
    if len(sys.argv) > 2:
        n = int(sys.argv[2])
        path += str(n)

    drop, blank, fill = [], [], []
    for i in range(n):
        for sent in sents:
            d, b, f = shatter(sent)
            drop.append(d)
            blank.append(b)
            fill.append(f)

    #write(drop, path + '.drop')
    write(blank, path + '.blank')
    write(fill, path + '.fill')
