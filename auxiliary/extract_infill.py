import argparse
import unicodedata

def assert_pdb(cond):
    if not cond:
        import pdb; pdb.set_trace()

parser = argparse.ArgumentParser()

parser.add_argument('--blank', metavar='FILE', required=True, help='Blank file with lengths')
parser.add_argument('--full', metavar='FILE', required=True, help='Predictions')

args = parser.parse_args()

with open(args.blank) as f:
    blank = f.readlines()

with open(args.full) as f:
    full = f.readlines()

for line_blank, line_full in zip(blank, full):
    i_blank = 0
    i_full = 0
    words_blank = line_blank.strip().split()
    words_full = line_full.strip().split()
    while i_blank < len(words_blank):
        word_blank = words_blank[i_blank]
        if not word_blank.startswith("<blank"):
            assert_pdb( words_full[i_full] == word_blank or words_full[i_full] == "<unk>")

            i_full += 1
            i_blank += 1
            continue
        count = int(word_blank.split("_")[1][:-1])
        for j in range(count):
            print(words_full[i_full + j], end=" ")
        i_full += count
        i_blank += 1
    print()

