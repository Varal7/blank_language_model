#include <torch/extension.h>
#include <stdlib.h>
#include <vector>

using namespace ::std;

vector<vector<uint32_t>> get_canvas (
    vector<uint32_t>& seq, vector<bool>& keep, uint32_t n, uint32_t blank_tok) {
  vector<uint32_t> canvas, blanks, rest, loc, lb, rb;
  for (uint32_t i = 0; i < n; ) {
    if (keep[i]) {
      canvas.push_back(seq[i]);
      i++;
    } else {
      lb.push_back(0);
      while (i < n && !keep[i]) {
        rest.push_back(i);
        loc.push_back(blanks.size());
        lb.push_back(1);
        rb.push_back(1);
        i++;
      }
      lb.pop_back();
      rb.pop_back();
      rb.push_back(0);
      blanks.push_back(canvas.size());
      canvas.push_back(blank_tok);
    }
  }
  return {canvas, blanks, rest, loc, lb, rb};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("get_canvas", &get_canvas, "get_canvas");
}
