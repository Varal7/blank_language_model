import argparse
import unicodedata

parser = argparse.ArgumentParser()

parser.add_argument('--gold', metavar='FILE', required=True, help='Gold')
parser.add_argument('--pred', metavar='FILE', required=True, help='Predictions')
parser.add_argument('--strip_accent', action="store_true", help='whether to strip accents')
parser.add_argument('--first', type=int, default=None, help='Evaluate only first [FIRST] examples')

args = parser.parse_args()

def strip_accents(s):
  return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def equals(a, b, strip_accent):
    if strip_accent:
        return strip_accents(a) == strip_accents(b)
    return a == b


with open(args.gold) as f:
    gold = f.readlines()

gold = [[s.replace(" ", "<space>") for s in p[:-1]] for p in gold]

with open(args.pred) as f:
    preds = f.readlines()

preds = [t[:-1].split() for t in preds]

if args.first is not None:
    preds = preds[:args.first]

count = 0
correct = 0
count_not_space = 0
correct_not_space = 0
if len(gold) != len(preds):
    print("Warning! Evaluating only on frist {}/{} preds".format(len(preds), len(gold)))
for g, p in zip(gold, preds):
    if not (len(g) == len(p)):
        print(len(g), len(p))
    for cg, cp in zip(g, p):
        if equals(cg, cp, args.strip_accent):
            correct += 1
            if cg != "<space>":
                correct_not_space += 1
        count += 1
        if cg != "<space>":
            count_not_space += 1

err = 1 - (correct / count)
err_not_space = 1 - (correct_not_space/ count_not_space)
print("Error rate: {} | Correct: {} | Total: {}".format(err, correct, count))
print("Error rate (not <space>): {} | Correct (not <space>) : {} | Total (not <space>): {}".format(err_not_space, correct_not_space, count_not_space))


