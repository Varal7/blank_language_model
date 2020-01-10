import sys
import random

from gen_blank import load_sent, write_sent, process

if __name__ == '__main__':
    random.seed(1)

    path = sys.argv[1]
    sents = load_sent(path + '.txt')

    n = 1
    if len(sys.argv) > 2:
        n = int(sys.argv[2])
        path += '.' + str(n) + 'x'

    blank, fill = [], []
    for i in range(n):
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
