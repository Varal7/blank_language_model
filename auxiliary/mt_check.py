import sys
import re

def load(path):
    with open(path) as f:
        lines = f.readlines()
    return [l.strip() for l in lines]

blank = load(sys.argv[1])
full = load(sys.argv[2])
err = 0

for b, f in zip(blank, full):
    b_ = re.split('<blank>', b)
    f_ = f
    for x in b_:
        k = f_.find(x)
        if k == -1:
            print('error:\n%s\n%s\n' % (b, f))
            err += 1
            break
        else:
            f_ = f_[k+len(x):]
print('error: %d/%d' % (err, len(blank)))
