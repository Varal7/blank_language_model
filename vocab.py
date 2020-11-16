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

    pad, unk, first, last, eos, blank, blank_0 = range(7)

    @staticmethod
    def build(sents, path, size, max_blank_len=None):
        voc = ['<pad>', '<unk>', '<first>', '<last>', '<eos>', '<blank>', '<blank_0>']
        if max_blank_len:
            voc += ['<blank_{}>'.format(i) for i in range(1, max_blank_len)]
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
            occ[Vocab.unk] += o

        with open(path, 'w') as f:
            for v, o in zip(voc, occ):
                f.write('{}\t{}\n'.format(v, o))

    def word_to_idx(self, word):
        return self.word2idx[word] if word in self.word2idx else Vocab.unk
