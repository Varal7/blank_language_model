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
        self.first = self.word2idx['<first>']
        self.last = self.word2idx['<last>']
        self.missing = self.word2idx['<missing>']
        self.eos = self.word2idx['<eos>']
        self.blank = self.word2idx['<blank>']
        self.blanks = [idx for idx in range(self.size) if self.idx2word[idx][:6] == "<blank"]

    @staticmethod
    def build(sents, path, size, max_blank_length=None):
        voc = ['<pad>', '<unk>', '<first>', '<last>', '<eos>', '<blank>', '<missing>']
        if max_blank_length is not None:
            voc += ['<blank_{}>'.format(i) for i in range(0, max_blank_length)]
        occ = [0 for _ in voc]

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

    def is_blank(self, candidate):
        return candidate == self.blank | self.is_l_lblank(candidate)

    def is_l_lblank(self, candidate):
        return len(self.blanks) > 0 & (self.blanks[0] <= candidate) & (candidate <= self.blanks[-1])

    def get_blank_length(self, idx):
        return idx - self.blanks[0]
