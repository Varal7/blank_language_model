#include <torch/extension.h>
#include <stdlib.h>
#include <vector>

using namespace std;

vector<vector<vector<uint32_t>>> get_canvas(
    vector<vector<uint32_t>>& seq,
    vector<vector<bool>>& keep,
    vector<uint32_t> n,
    uint32_t blank_tok) {
  vector<vector<uint32_t>> batch_canvas, batch_blanks, batch_rest, batch_loc, batch_lb, batch_rb;
  for (uint32_t b = 0; b < seq.size(); b++) {
    vector<uint32_t> canvas, blanks, rest, loc, lb, rb;
    for (uint32_t i = 0; i < n[b]; ) {
      if (keep[b][i]) {
        canvas.push_back(seq[b][i]);
        i++;
      } else {
        lb.push_back(0);
        while (i < n[b] && !keep[b][i]) {
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
    batch_canvas.push_back(canvas);
    batch_blanks.push_back(blanks);
    batch_rest.push_back(rest);
    batch_loc.push_back(loc);
    batch_lb.push_back(lb);
    batch_rb.push_back(rb);
  }
  return {batch_canvas, batch_blanks, batch_rest, batch_loc, batch_lb, batch_rb};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("get_canvas", &get_canvas, "get_canvas");
}
