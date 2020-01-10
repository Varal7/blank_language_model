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

inp_file += '.fail'
out_file += '.fail'
with open(inp_file, 'w') as f:
    f.writelines(inp_b)
with open(out_file, 'w') as f:
    f.writelines(out_b)
