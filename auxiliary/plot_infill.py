import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

n_blanks = [1, 2, 3, 4, 5]
max_len = [5, 10]
models = ['No Infill', 'Blank Model']
markers = {5: 'o', 10: 's'}
colors = {'No Infill': 'tab:orange', 'Blank Model': 'tab:green'}

bleu = {}
for l in max_len:
    bleu[l] = {}

bleu[5]['No Infill'] = [94.31,	88.6,	82.99,	77.26,	71.93]
bleu[10]['No Infill'] = [90.95,	81.83,	72.98,	65.36,	58.75]

bleu[5]['Blank Model'] = [95.12,	90.29,	85.61,	80.77,	76.41]
bleu[10]['Blank Model'] = [91.66,	83.43,	75.88,	69.26,	63.51]

for l in max_len:
    for m in models:
        plt.plot(n_blanks, bleu[l][m], marker=markers[l], color=colors[m], label='max_len %d, %s' % (l, m))

plt.legend(prop={'size': 12})
plt.xlabel('#blanks', fontsize=14)
plt.ylabel('BLEU', fontsize=14)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.show()
