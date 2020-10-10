import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

from vocab import Vocab

get_canvas_cpp = load(name='canvas', sources=['models/get_canvas.cpp'])


def select(logits, decode):
    if decode == 'sample':
        return torch.multinomial(logits.exp(), num_samples=1)[0]
    else:
        return logits.argmax()


def seq_cross_entropy(pred, gold, pad):
    gold_shape = gold.shape
    pred = pred.view(-1, pred.size(-1))
    gold = gold.view(-1)
    loss = F.cross_entropy(pred, gold, ignore_index=pad, reduction='none')
    return loss.view(gold_shape)


def new_arange(x, *size):
    """
    Return a Tensor of `size` filled with a range function on the device of x
    If `size` is empty, using the size of the variable x
    """
    if len(size) == 0:
        size = x.size()
    return torch.arange(size[-1], device=x.device).expand(*size).contiguous()


def batch_randint(start, batch_end):
    """
    Sample k from start to end (both inclusive)
    Return the same shape as batch_end
    """
    return start + (torch.rand_like(batch_end.float()) * (batch_end - start + 1).float()).long()


def sample_permutation(seq):
    score = torch.rand_like(seq.float())
    score.masked_fill_(seq == Vocab.pad, 1)         # always put pads last
    score.masked_fill_(seq == Vocab.first, -1)      # always keep <first>
    score.masked_fill_(seq == Vocab.last, -1)       # always keep <last>
    score.masked_fill_(seq == Vocab.missing, -1)    # always keep missings
    indices = score.argsort()
    rank = torch.zeros_like(seq)
    rank[torch.arange(len(seq)).unsqueeze(1), indices] = \
        torch.arange(seq.size(1), device=seq.device)
    return rank


def collect(input, index, padding_idx=0):
    """
    Perform a batched index select where index is given for each example
    Args:
        input: tensor of shape (B, T_1, dim_2, dim_3, ...)
        index: tensor of shape (B, T_2)
    Return:
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


def to_tensor(x, pad_id, device):
    max_len = max([len(xi) for xi in x])
    x_ = [xi + [pad_id] * (max_len - len(xi)) for xi in x]
    return torch.tensor(x_).to(device)


def get_canvas(seq, keep, n):
    """
    Args:
        seq: original (batched) sequence of tokens
        keep: mask over seq indicating whether to keep each token
        n: number of tokens
    Return:
        canvas: replace consecutive masked tokens in seq by the <blank> token
        blanks: indices of <blank> tokens in canvas
        rest: indices of masked tokens in seq, these are the tokens to predict
        loc: indices of how rest relates to blanks
        lb: whether to create a left blank for predicting each token in rest
        rb: whether to create a right blank for predicting each token in rest
        (rest, loc, lb, rb have the same shape)
    """
    res = get_canvas_cpp.get_canvas(seq.tolist(), keep.tolist(), n.tolist(), Vocab.blank)
    pad = [Vocab.pad, -1, -1, -1, -1, -1]
    return [to_tensor(r, p, seq.device) for r, p in zip(res, pad)]


def get_known_length_canvas(seq, keep, n):
    """
    Return:
        canvas: replace consecutive masked tokens in seq by the <blank_t> token
        blanks: indices of <blank_t> tokens in canvas
        rest: indices of masked tokens in seq, these are the tokens to predict
        loc: indices of how rest relates to blanks
        lb: length of the new left blank for predicting each token in rest
        (rest, loc, lb have the same shape)
    """
    res = get_canvas_cpp.get_known_length_canvas(seq.tolist(), keep.tolist(), n.tolist(), Vocab.blank_0)
    pad = [Vocab.pad, -1, -1, -1, -1]
    return [to_tensor(r, p, seq.device) for r, p in zip(res, pad)]


def get_ins_canvas(seq, keep, n):
    """
    Return:
        canvas: remove masked tokens in seq
        rest: indices of masked tokens in seq, these are the tokens to predict
        loc: indices of how rest relates to canvas
        (rest, loc have the same shape)
    """
    res = get_canvas_cpp.get_insertion_canvas(seq.tolist(), keep.tolist(), n.tolist())
    pad = [Vocab.pad, -1, -1]
    return [to_tensor(r, p, seq.device) for r, p in zip(res, pad)]
