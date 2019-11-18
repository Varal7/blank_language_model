import sys
import re

def load(path):
    with open(path) as f:
        lines = f.readlines()
    return [l.strip() for l in lines]

blank = load(sys.argv[1])
fill = load(sys.argv[2])
err = 0
with open(sys.argv[2] + '.merge', 'w') as fout:
    for b, f in zip(blank, fill):
        b_ = re.split('<blank>', b)
        f_ = re.split(' <sep> ', f) if f != '' else []
        if len(b_) != len(f_) + 1:
            #print('error:\n%s\n%s\n' % (b, f))
            err += 1
        s = b_[0]
        for i in range(len(b_)-1):
            if i < len(f_):
                s += f_[i]
            s += b_[i+1]
        fout.write(s + '\n')
print('error: %d/%d' % (err, len(blank)))
