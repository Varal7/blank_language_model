import sys
import random

def load_sent(path):
    sents = []
    with open(path) as f:
        for line in f:
            sents.append(line.split())
    return sents

path = sys.argv[1]
sents = load_sent(path + '.txt')
with open(path + '.blank', 'w') as fb:
    with open(path + '.fill', 'w') as ff:
        for s in sents:
            k = random.randint(0, len(s))
            order = list(range(len(s)))
            random.shuffle(order)
            keep = order[:k]

            blank, fill = [], []
            for i, w in enumerate(s):
                if i in keep:
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

            fb.write(' '.join(blank) + '\n')
            ff.write(' '.join(fill) + '\n')
