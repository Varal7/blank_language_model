import torch

def get_batch(x, vocab):
    seq = []
    n = [len(s) for s in x]
    n_real = []
    max_len = max(n)
    for s, l in zip(x, n):
        s_idx = [vocab.word2idx[w] if w in vocab.word2idx else vocab.unk for w in s]
        n_real.append(l - sum(1 for t in s if t.endswith("@@")))
        seq.append([vocab.first] + s_idx + [vocab.last] + [vocab.pad] * (max_len - l))
    return torch.LongTensor(seq), torch.LongTensor(n), torch.LongTensor(n_real)

def get_batches(data, vocab, max_tokens, same_len=False):
    order = range(len(data))
    z = sorted(zip(order, data), key=lambda i: -len(i[1]))
    order, data = zip(*z)

    batches = []
    i = 0
    while i < len(data):
        j = i
        while j < len(data) and (len(data[j]) + 2) * (j-i+1) <= max_tokens and \
            (not same_len or len(data[j]) == len(data[i])):
            j += 1
        batches.append(get_batch(data[i: j], vocab))
        i = j
    return batches, order

def load_sent(path, add_eos=False):
    sents = []
    with open(path) as f:
        for line in f:
            s = line.split()
            if add_eos:
                s += ['<eos>']
            sents.append(s)
    return sents

def load_data(path, add_eos=False, cat_sent=False, max_len=512):
    if not add_eos:
        print("WARNING: You should always use add_eos to get comparable ppl on LM tasks")

    sents = load_sent(path, add_eos)
    if cat_sent:
        if not add_eos:
            raise ValueError("Using cat_sent without add_eos")
        d = [w for s in sents for w in s]
        sents = [d[i: i+max_len] for i in range(0, len(d), max_len)]
    else:
        n = len(sents)
        sents = [s for s in sents if len(s) <= max_len]
        print('# discarded sents:', n - len(sents))
    return sents

class LMDataset(torch.utils.data.Dataset):
    def __init__(self, batches):
        self.batches = batches

    def __getitem__(self, idx):
        return self.batches[idx]

    def __len__(self):
        return len(self.batches)


def get_train_dataloader(train_sents, vocab, max_tok, data_workers=8):
    train_batches, _ = get_batches(train_sents, vocab, max_tok)
    print("Number of train batches: {}".format(len(train_batches)))
    train_ds = LMDataset(train_batches)
    train_dl = torch.utils.data.DataLoader(train_ds, num_workers=data_workers, shuffle=True, pin_memory=True)
    return train_dl

def get_eval_dataloader(val_sents, vocab, max_tok, data_workers=8):
    val_batches, _ = get_batches(val_sents, vocab, max_tok, same_len=True)
    print("Number of eval batches: {}".format(len(val_batches)))
    val_ds = LMDataset(val_batches)
    val_dl = torch.utils.data.DataLoader(val_ds, num_workers=data_workers, pin_memory=True)
    return val_dl
