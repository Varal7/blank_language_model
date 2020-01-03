import re

def convert(inp, out):
    with open(inp) as finp:
        with open(out, 'w') as fout:
            for line in finp:
                line = line.replace('[', '').replace(']', '')
                #line = re.sub('\-+', '-', line)
                for c in line:
                    if c == ' ':
                        fout.write('_')
                    elif c == '-':
                        fout.write('<missing>')
                    else:
                        fout.write(c)

                    if c != '\n':
                        fout.write(' ')

for x in ['train', 'valid', 'test']:
    convert(x + '.ori.txt', x + '.txt')
