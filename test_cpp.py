import unittest

import torch
from torch.utils.cpp_extension import load

batched_get_canvas = load(name="canvas", sources=["get_canvas.cpp"])


def get_insertion_canvas(seq, keep, n):
    return [x[0] for x in  batched_get_canvas.get_insertion_canvas([seq.tolist()], [keep.tolist()], [n])]

def get_known_length_canvas(seq, keep, n, blank_id):
    return [x[0] for x in batched_get_canvas.get_known_length_canvas([seq.tolist()], [keep.tolist()], [n], blank_id)]

def get_canvas(seq, keep, n, blank_id):
    return [x[0] for x in  batched_get_canvas.get_canvas([seq.tolist()], [keep.tolist()], [n], blank_id)]


class TestMakeBlanks(unittest.TestCase):
    def test_get_canvas_one(self):
        seq       = torch.tensor([0, 100, 101, 102, 103, 104, 105, 106, 107, 2, 1])
        keep_mask = torch.tensor([1,   1,   0,   1,   1,   0,   0,   1,   0, 1, 0]).bool()
        n = 10
        canvas, blanks, rest, loc, lb, rb = get_canvas(seq, keep_mask, n, 4)

        exp_canvas = [0, 100, 4, 102, 103, 4, 106, 4, 2]
        exp_blanks = [2, 5, 7]
        exp_rest   = [2, 5, 6, 8]
        exp_loc    = [0, 1, 1, 2]
        exp_lb     = [0, 0, 1, 0]
        exp_rb     = [0, 1, 0, 0]

        self.assertEqual(exp_canvas, canvas)
        self.assertEqual(exp_blanks, blanks)
        self.assertEqual(exp_rest, rest)
        self.assertEqual(exp_loc, loc)
        self.assertEqual(exp_lb, lb)
        self.assertEqual(exp_rb, rb)


    def test_get_canvas_two(self):
        seq       = torch.tensor([0, 100, 101, 102, 103, 104, 105, 106, 107, 2, 1])
        keep_mask = torch.tensor([1,   1,   0,   1,   1,   0,   0,   1,   1, 1, 0]).bool()
        n = 10
        canvas, blanks, rest, loc, lb, rb = get_canvas(seq, keep_mask, n, 4)

        exp_canvas = [0, 100, 4, 102, 103, 4, 106, 107, 2]
        exp_blanks = [2, 5]
        exp_rest   = [2, 5, 6]
        exp_loc    = [0, 1, 1]
        exp_lb     = [0, 0, 1]
        exp_rb     = [0, 1, 0]

        self.assertEqual(exp_canvas, canvas)
        self.assertEqual(exp_blanks, blanks)
        self.assertEqual(exp_rest, rest)
        self.assertEqual(exp_loc, loc)
        self.assertEqual(exp_lb, lb)
        self.assertEqual(exp_rb, rb)

    def test_get_canvas_three(self):
        seq       = torch.tensor([0, 200, 201, 202, 203, 204, 205, 206, 207, 2, 1])
        keep_mask = torch.tensor([1,   0,   1,   0,   1,   0,   1,   0,   1, 1, 0]).bool()
        n = 10
        canvas, blanks, rest, loc, lb, rb = get_canvas(seq, keep_mask, n, 4)

        exp_canvas = [0, 4, 201, 4, 203, 4, 205, 4, 207, 2]
        exp_blanks = [1, 3, 5, 7]
        exp_rest   = [1, 3, 5, 7]
        exp_loc    = [0, 1, 2, 3]
        exp_lb     = [0, 0, 0, 0]
        exp_rb     = [0, 0, 0, 0]

        self.assertEqual(exp_canvas, canvas)
        self.assertEqual(exp_blanks, blanks)
        self.assertEqual(exp_rest, rest)
        self.assertEqual(exp_loc, loc)
        self.assertEqual(exp_lb, lb)
        self.assertEqual(exp_rb, rb)


class TestMakeFixedLengthBlanks(unittest.TestCase):
    def test_get_canvas_one(self):
        seq       = torch.tensor([0, 100, 101, 102, 103, 104, 105, 106, 107, 2, 1])
        keep_mask = torch.tensor([1,   1,   0,   1,   1,   0,   0,   1,   0, 1, 0]).bool()
        n = 10
        blank_0 = 3
        canvas, blanks, rest, loc, lb = get_known_length_canvas(seq, keep_mask, n, blank_0)

        exp_canvas = [0, 100, 4, 102, 103, 5, 106, 4, 2]
        exp_blanks = [2, 5, 7]
        exp_rest   = [2, 5, 6, 8]
        exp_loc    = [0, 1, 1, 2]
        exp_lb     = [0, 0, 1, 0]

        self.assertEqual(exp_canvas, canvas)
        self.assertEqual(exp_blanks, blanks)
        self.assertEqual(exp_rest, rest)
        self.assertEqual(exp_loc, loc)
        self.assertEqual(exp_lb, lb)


    def test_get_canvas_two(self):
        seq       = torch.tensor([0, 100, 101, 102, 103, 104, 105, 106, 107, 2, 1])
        keep_mask = torch.tensor([1,   1,   0,   1,   1,   0,   0,   1,   1, 1, 0]).bool()
        n = 10
        blank_0 = 3
        canvas, blanks, rest, loc, lb = get_known_length_canvas(seq, keep_mask, n, blank_0)

        exp_canvas = [0, 100, 4, 102, 103, 5, 106, 107, 2]
        exp_blanks = [2, 5]
        exp_rest   = [2, 5, 6]
        exp_loc    = [0, 1, 1]
        exp_lb     = [0, 0, 1]

        self.assertEqual(exp_canvas, canvas)
        self.assertEqual(exp_blanks, blanks)
        self.assertEqual(exp_rest, rest)
        self.assertEqual(exp_loc, loc)
        self.assertEqual(exp_lb, lb)

    def test_get_canvas_three(self):
        seq       = torch.tensor([0, 200, 201, 202, 203, 204, 205, 206, 207, 2, 1])
        keep_mask = torch.tensor([1,   0,   1,   0,   1,   0,   1,   0,   1, 1, 0]).bool()
        n = 10
        blank_0 = 3
        canvas, blanks, rest, loc, lb = get_known_length_canvas(seq, keep_mask, n, blank_0)

        exp_canvas = [0, 4, 201, 4, 203, 4, 205, 4, 207, 2]
        exp_blanks = [1, 3, 5, 7]
        exp_rest   = [1, 3, 5, 7]
        exp_loc    = [0, 1, 2, 3]
        exp_lb     = [0, 0, 0, 0]

        self.assertEqual(exp_canvas, canvas)
        self.assertEqual(exp_blanks, blanks)
        self.assertEqual(exp_rest, rest)
        self.assertEqual(exp_loc, loc)
        self.assertEqual(exp_lb, lb)


    def test_get_canvas_four(self):
        seq       = torch.tensor([0, 200, 201, 202, 203, 204, 205, 206, 207, 2, 1])
        keep_mask = torch.tensor([1,   0,   0,   0,   0,   0,   1,   0,   1, 1, 0]).bool()
        n = 10
        blank_0 = 3
        canvas, blanks, rest, loc, lb = get_known_length_canvas(seq, keep_mask, n, blank_0)

        exp_canvas = [0, 8, 205, 4, 207, 2]
        exp_blanks = [1, 3]
        exp_rest   = [1, 2, 3, 4, 5, 7]
        exp_loc    = [0, 0, 0, 0, 0, 1]
        exp_lb     = [0, 1, 2, 3, 4, 0]

        self.assertEqual(exp_canvas, canvas)
        self.assertEqual(exp_blanks, blanks)
        self.assertEqual(exp_rest, rest)
        self.assertEqual(exp_loc, loc)
        self.assertEqual(exp_lb, lb)


class TestMakeInsertionBlanks(unittest.TestCase):
    def test_get_canvas_one(self):
        seq       = torch.tensor([0, 100, 101, 102, 103, 104, 105, 106, 107, 2, 1])
        keep_mask = torch.tensor([1,   1,   0,   1,   1,   0,   0,   1,   0, 1, 0]).bool()
        n = 8
        canvas, rest, loc = get_insertion_canvas(seq, keep_mask, n)

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
        canvas, rest, loc = get_insertion_canvas(seq, keep_mask, n)

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
        canvas, rest, loc = get_insertion_canvas(seq, keep_mask, n)

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
        canvas, rest, loc = get_insertion_canvas(seq, keep_mask, n)

        exp_canvas = [0, 200, 201, 202, 203, 204, 205, 206, 207, 2]
        exp_rest   = [9]
        exp_loc    = [8]

        self.assertEqual(exp_canvas, canvas)
        self.assertEqual(exp_rest, rest)
        self.assertEqual(exp_loc, loc)



if __name__ == "__main__":
    unittest.main()
