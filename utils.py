import random
import numpy as np
import torch
import math
import torch.nn.functional as F

from torch.utils.cpp_extension import load

get_canvas_cpp = load(name="canvas", sources=["get_canvas.cpp"])

def set_seed(seed):     # set the random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def strip_eos(sents):
    return [sent[:sent.index('<eos>')] if '<eos>' in sent else sent
        for sent in sents]

def new_arange(x, *size):
    """
    Return a Tensor of `size` filled with a range function on the device of x.
    If size is empty, using the size of the variable x.
    """
    if len(size) == 0:
        size = x.size()
    return torch.arange(size[-1], device=x.device).expand(*size).contiguous()

def seq_cross_entropy(pred, gold, pad):
    ''' Calculate cross entropy loss'''

    gold_shape = gold.shape
    pred = pred.view(-1, pred.size(-1))
    gold = gold.view(-1)
    loss = F.cross_entropy(pred, gold, ignore_index=pad, reduction='none')

    return loss.view(gold_shape)


def to_tensor(x, pad_id, device):
    max_len = max([len(xi) for xi in x])
    x_ = [xi + [pad_id] * (max_len - len(xi)) for xi in x]
    return torch.tensor(x_).to(device)


def sample_permutation(seq, vocab):
    score = torch.rand_like(seq.float())
    score.masked_fill_(seq == vocab.pad, 1)  # always put pads last
    score.masked_fill_(seq == vocab.first, -1)  # always keep <first>
    score.masked_fill_(seq == vocab.last, -1)  # always keep <last>
    indices = score.argsort()
    rank = torch.zeros_like(seq)
    rank[torch.arange(len(seq)).unsqueeze(1), indices] = \
        torch.arange(seq.size(1), device=seq.device)
    return rank

def get_ins_canvas(seq, keep, n, vocab):
    """Returns canvas, rest, loc"""
    res = get_canvas_cpp.get_insertion_canvas(seq.tolist(), keep.tolist(), n.tolist())
    pad = [vocab.pad, -1, -1]
    for i in range(len(res)):
        res[i] = to_tensor(res[i], pad[i], seq.device)
    return res


def collect(input, index, padding_idx=0):
    """
    Performs a batched index select where index is given for each example
    Args:
        input: tensor of shape (B, T_1, dim_2, dim_3, ...)
        index: tensor of shape (B, T_2)
    Returns:
        tensor of shape (B, T_2, dim_2, dim_3, ...)
    """
    # Add a column of padding_idx at index 0 (of dim 1)
    view = list(input.shape)
    view[1] = 1
    padding_column = input.new_ones(view) * padding_idx
    input = torch.cat([padding_column, input], 1)

    # Expand index to compatible size for gather
    for i in range(2, len(input.shape)):
        index = index.unsqueeze(i)

    view[0] = -1
    view[1] = -1
    index = index.expand(view)
    return torch.gather(input, 1, index + 1)

def write_sent(sents, path):
    with open(path, 'w') as f:
        for s in sents:
            f.write(' '.join(s) + '\n')

def write_doc(docs, path):
    with open(path, 'w') as f:
        for d in docs:
            for s in d:
                f.write(' '.join(s) + '\n')
            f.write('\n')

class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)
