import torch

def get_batch(x, vocab):
    batch_seq = []
    for s in x:
        s_idx = [vocab.word2idx[w] if w in vocab.word2idx else vocab.unk for w in s]
        batch_seq.append([vocab.bos] + s_idx + [vocab.eos])
    return torch.LongTensor(batch_seq)

def get_batches(data, vocab, batch_size):
    order = range(len(data))
    z = sorted(zip(order, data), key=lambda i: len(i[1]))
    order, data = zip(*z)

    batches = []
    i = 0
    while i < len(data):
        j = i
        while j < min(len(data), i+batch_size) and len(data[j]) == len(data[i]):
            j += 1
        batches.append(get_batch(data[i: j], vocab))
        i = j
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
        while j < len(data) and len(data[j]) == len(data[i]) \
            and (len(data[j])+2) * (j-i+1) <= max_tokens:
            j += 1
        batches.append(get_batch(data[i:j], vocab))
        i = j
    return batches, order
'''
