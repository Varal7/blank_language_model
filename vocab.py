from collections import Counter

class Vocab(object):
    def __init__(self, path):
        self.word2idx = {}
        self.idx2word = []

        with open(path) as f:
            for line in f:
                w = line.split()[0]
                self.word2idx[w] = len(self.word2idx)
                self.idx2word.append(w)
        self.size = len(self.word2idx)

        self.pad = self.word2idx['<pad>']
        self.unk = self.word2idx['<unk>']
        self.bos = self.word2idx['<bos>']
        self.eos = self.word2idx['<eos>']
        self.blank = self.word2idx['<blank>']

    @staticmethod
    def build(sents, path, size):
        voc = ['<pad>', '<unk>', '<bos>', '<eos>', '<blank>']
        occ = [0, 0, 0, 0, 0]

        cnt = Counter([w for s in sents for w in s])
        for i, v in enumerate(voc):
            if v in cnt:
                occ[i] = cnt[v]
                del cnt[v]
        for v, o in cnt.most_common(size):
            voc.append(v)
            occ.append(o)
        for v, o in cnt.most_common()[size:]:
            occ[1] += o

        with open(path, 'w') as f:
            for v, o in zip(voc, occ):
                f.write('{}\t{}\n'.format(v, o))
