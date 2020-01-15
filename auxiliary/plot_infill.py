import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

mask_ratio = [10, 20, 30, 40, 50]

bleu = {}
bleu['No Infill'] = [75.34,	55.89,	39.41,	26.43,	16.2]
bleu['Blank Model'] = [86.51,	73.15,	59.6,	46.79,	34.78]

for m in ['No Infill', 'Blank Model']:
    plt.plot(mask_ratio, bleu[m], '-o', label=m)

plt.xticks(mask_ratio)
plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
#plt.gca().xaxis.set_major_locator(mtick.MaxNLocator(integer=True))

plt.legend(prop={'size': 12})
plt.xlabel('Mask Ratio', fontsize=14)
plt.ylabel('BLEU', fontsize=14)
plt.show()
