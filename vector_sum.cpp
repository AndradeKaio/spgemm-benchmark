#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <iostream>
#include <vector>

// Sum two vectors stored in compressed form using AVX-512 expand + add.
// Each vector is represented by:
//   - vals: packed values (length = popcount(mask))
//   - mask: bitmask describing target lanes (LSB = lane 0)
// The result is written as a dense vector (expanded).

using val_t = float;

static inline int popcount_u32(uint32_t x) { return __builtin_popcount(x); }

void sum_compressed_vectors_avx512(const val_t *a_vals, uint32_t a_mask,
                                   const val_t *b_vals, uint32_t b_mask,
                                   val_t *out_dense,
                                   int lanes // must be <= 16 for AVX-512 fp32
) {
  // Preconditions:
  // - lanes <= 16
  // - a_vals has popcount(a_mask) elements
  // - b_vals has popcount(b_mask) elements

  // Load compressed values into registers
  __m512 a_comp = _mm512_setzero_ps();
  __m512 b_comp = _mm512_setzero_ps();

  // Load only the needed number of elements (scalar loads are fine here)
  int a_nnz = popcount_u32(a_mask);
  int b_nnz = popcount_u32(b_mask);

  // Copy compressed values into the low lanes of a_comp / b_comp
  // (expand will consume them in order)
  std::memcpy(&a_comp, a_vals, a_nnz * sizeof(val_t));
  std::memcpy(&b_comp, b_vals, b_nnz * sizeof(val_t));

  // Expand according to masks
  __mmask16 ma = (__mmask16)a_mask;
  __mmask16 mb = (__mmask16)b_mask;

  __m512 a_exp = _mm512_maskz_expand_ps(ma, a_comp);
  __m512 b_exp = _mm512_maskz_expand_ps(mb, b_comp);

  // Add
  __m512 c = _mm512_add_ps(a_exp, b_exp);

  // Store only the requested number of lanes
  // (store all 16 lanes if you want a full vector)
  __mmask16 mout = (__mmask16)((1u << lanes) - 1u);
  _mm512_mask_storeu_ps(out_dense, mout, c);
}

int main() {
  // Example from the discussion
  // A: vals = [1,2,3,4], mask = 1010101b
  // B: vals = [4,5,6],   mask = 1100001b

  std::vector<val_t> a_vals = {1.f, 2.f, 3.f, 4.f};
  std::vector<val_t> b_vals = {4.f, 5.f, 6.f};

  uint32_t a_mask = 0b1010101; // lanes 0,2,4,6
  uint32_t b_mask = 0b1100001; // lanes 0,1,6

  val_t out[16] = {0};

  sum_compressed_vectors_avx512(a_vals.data(), a_mask, b_vals.data(), b_mask,
                                out, 7);

  std::cout << "Result: ";
  for (int i = 0; i < 7; ++i) {
    std::cout << out[i] << " ";
  }
  std::cout << "\n";

  return 0;
}
