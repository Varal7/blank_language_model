import torch

from vocab import Vocab


def get_batch(x, vocab, append_at_ends=False):
    seq = []
    n = [len(s) for s in x]
    n_real = []
    max_len = max(n)
    for s, l in zip(x, n):
        # Combine BPE tokens to count the number of words
        n_real.append(l - sum(1 for t in s if t.endswith("@@")))

        s_idx = [vocab.word_to_idx(w) for w in s]
        if append_at_ends:
            s_idx = [Vocab.first] + s_idx + [Vocab.last]
        seq.append(s_idx + [Vocab.pad] * (max_len - l))
    return torch.LongTensor(seq), torch.LongTensor(n), torch.LongTensor(n_real)


def get_batches(data, vocab, max_tok, append_at_ends=False, same_len=False):
    offset = 2 if append_at_ends else 0

    order = range(len(data))
    z = sorted(zip(order, data), key=lambda i: len(i[1]), reverse=True)
    order, data = zip(*z)

    batches = []
    i = 0
    while i < len(data):
        j = i
        while j < len(data) and (len(data[i]) + offset) * (j-i+1) <= max_tok \
                and (not same_len or len(data[j]) == len(data[i])):
            j += 1
        batches.append(get_batch(data[i: j], vocab, append_at_ends))
        i = j
    return batches, order


class LMDataset(torch.utils.data.Dataset):
    def __init__(self, batches):
        self.batches = batches

    def __getitem__(self, idx):
        return self.batches[idx]

    def __len__(self):
        return len(self.batches)


def get_train_dataloader(train_data, vocab, max_tok, data_workers=8,
                         model_type=None):
    train_batches, _ = get_batches(train_data, vocab, max_tok,
                                   append_at_ends=(model_type == 'inst'))
    print("Number of train batches: {}".format(len(train_batches)))
    train_ds = LMDataset(train_batches)
    return torch.utils.data.DataLoader(train_ds, num_workers=data_workers,
                                       shuffle=True, pin_memory=True)


def get_eval_dataloader(val_data, vocab, max_tok, data_workers=8,
                        model_type=None):
    val_batches, _ = get_batches(val_data, vocab, max_tok, same_len=True,
                                 append_at_ends=(model_type == 'inst'))
    print("Number of eval batches: {}".format(len(val_batches)))
    val_ds = LMDataset(val_batches)
    return torch.utils.data.DataLoader(val_ds, num_workers=data_workers,
                                       pin_memory=True)
