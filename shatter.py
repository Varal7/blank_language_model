import sys
import random
from utils import load_sent

path = sys.argv[1]
sents = load_sent(path)
with open(path + '.shatter', 'w') as f:
    for s in sents:
        k = random.randint(1, len(s))
        order = list(range(len(s)))
        random.shuffle(order)
        keep = order[:k]
        s_ = [w for i, w in enumerate(s) if i in keep]
        f.write(' '.join(s_) + '\n')
