import unittest

import torch
from torch.utils.cpp_extension import load

batched_get_canvas = load(name="get_canvas_cpp", sources=["get_canvas.cpp"])


def get_canvas(seq, keep, n):
    return [x[0] for x in  batched_get_canvas.get_insertion_canvas([seq.tolist()], [keep.tolist()], [n])]


class TestMakeInsertionBlanks(unittest.TestCase):
    def test_get_canvas_one(self):
        seq       = torch.tensor([0, 100, 101, 102, 103, 104, 105, 106, 107, 2, 1])
        keep_mask = torch.tensor([1,   1,   0,   1,   1,   0,   0,   1,   0, 1, 0]).bool()
        n = 8
        canvas, rest, loc = get_canvas(seq, keep_mask, n)

        exp_canvas = [0, 100, 102, 103, 106, 2]
        exp_rest   = [2, 5, 6, 8]
        exp_loc    = [1, 3, 3, 4]

        self.assertEqual(exp_canvas, canvas)
        self.assertEqual(exp_rest, rest)
        self.assertEqual(exp_loc, loc)


    def test_get_canvas_two(self):
        seq       = torch.tensor([0, 100, 101, 102, 103, 104, 105, 106, 107, 2, 1])
        keep_mask = torch.tensor([1,   1,   0,   1,   1,   0,   0,   1,   1, 1, 0]).bool()
        n = 8
        canvas, rest, loc = get_canvas(seq, keep_mask, n)

        exp_canvas = [0, 100, 102, 103, 106, 107, 2]
        exp_rest   = [2, 5, 6]
        exp_loc    = [1, 3, 3]

        self.assertEqual(exp_canvas, canvas)
        self.assertEqual(exp_rest, rest)
        self.assertEqual(exp_loc, loc)

    def test_get_canvas_three(self):
        seq       = torch.tensor([0, 200, 201, 202, 203, 204, 205, 206, 207, 2, 1])
        keep_mask = torch.tensor([1,   0,   1,   0,   1,   0,   1,   0,   1, 1, 0]).bool()
        n = 8
        canvas, rest, loc = get_canvas(seq, keep_mask, n)

        exp_canvas = [0, 201, 203, 205, 207, 2]
        exp_rest   = [1, 3, 5, 7]
        exp_loc    = [0, 1, 2, 3]

        self.assertEqual(exp_canvas, canvas)
        self.assertEqual(exp_rest, rest)
        self.assertEqual(exp_loc, loc)

    def test_get_canvas_all(self):
        seq       = torch.tensor([0, 200, 201, 202, 203, 204, 205, 206, 207, 2, 1])
        keep_mask = torch.tensor([1,   1,   1,   1,   1,   1,   1,   1,   1, 1, 0]).bool()
        n = 8
        canvas, rest, loc = get_canvas(seq, keep_mask, n)

        exp_canvas = [0, 200, 201, 202, 203, 204, 205, 206, 207, 2]
        exp_rest   = [9]
        exp_loc    = [8]

        self.assertEqual(exp_canvas, canvas)
        self.assertEqual(exp_rest, rest)
        self.assertEqual(exp_loc, loc)



if __name__ == "__main__":
    unittest.main()
