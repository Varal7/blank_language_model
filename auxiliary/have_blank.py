import sys

inp_file = sys.argv[1]
out_file = sys.argv[2]
with open(inp_file) as f:
    inp = f.readlines()
with open(out_file) as f:
    out = f.readlines()

inp_b, out_b = [], []
for i, o in zip(inp, out):
    if '<blank>' in o:
        inp_b.append(i)
        out_b.append(o)

print('%d/%d have <blank>' % (len(out_b), len(out)))
with open(out_file + '.have_blank', 'w') as f:
    for i, o in zip(inp_b, out_b):
        f.write(i)
        f.write(o)
        f.write('\n')
