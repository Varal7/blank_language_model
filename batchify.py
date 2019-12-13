import torch

def get_batch(x, vocab):
    seq = []
    n = [len(s) for s in x]
    max_len = max(n)
    for s, l in zip(x, n):
        s_idx = [vocab.word2idx[w] if w in vocab.word2idx else vocab.unk for w in s]
        seq.append(s_idx + [vocab.pad] * (max_len - l))
    return torch.LongTensor(seq), torch.LongTensor(n)

def get_batches(data, vocab, batch_size):
    order = range(len(data))
    z = sorted(zip(order, data), key=lambda i: len(i[1]))
    order, data = zip(*z)

    batches = [get_batch(data[i: i+batch_size], vocab) \
        for i in range(0, len(data), batch_size)]
    return batches, order

'''
def get_batches(data, vocab, max_tokens):
    order = range(len(data))
    z = sorted(zip(order, data), key=lambda i: len(i[1]))
    order, data = zip(*z)

    batches = []
    i = 0
    while i < len(data):
        j = i
        while j < len(data) and (len(data[j])+2) * (j-i+1) <= max_tokens:
            j += 1
        batches.append(get_batch(data[i:j], vocab))
        i = j
    return batches, order
'''
