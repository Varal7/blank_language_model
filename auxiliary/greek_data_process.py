import re

def convert_train(inp, out):
    with open(inp) as finp:
        with open(out, 'w') as fout:
            for line in finp:
                line = line.replace('[', '').replace(']', '')
                #line = re.sub('\-+', '-', line)
                for c in line:
                    if c == ' ':
                        fout.write('<space>')
                    elif c == '-':
                        fout.write('<missing>')
                    else:
                        fout.write(c)

                    if c != '\n':
                        fout.write(' ')

def convert_test(inp, out):
    with open(inp) as finp:
        with open(out, 'w') as fout:
            for line in finp:
                line = re.sub('_+', '_', line)
                for c in line:
                    if c == ' ':
                        fout.write('<space>')
                    elif c == '-':
                        fout.write('<missing>')
                    elif c == '_':
                        fout.write('<blank>')
                    elif c == '.':
                        fout.write('<eos>')
                    else:
                        fout.write(c)

                    if c != '\n':
                        fout.write(' ')

#for x in ['train', 'valid', 'test']:
#    convert_train(x + '.ori.txt', x + '.txt')

convert_test('test.x', 'test.blank')
