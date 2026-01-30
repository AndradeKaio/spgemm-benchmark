// AVX-512 Outer-product SpGEMM: PAD vs EXPAND vs EXPAND_MASK
// --------------------------------------------------
// Three kernels:
//   1) PAD          : B rows fully padded (dense), pure AVX-512 FMAs
//   2) EXPAND       : B rows index-compressed, expanded ONCE per row
//   3) EXPAND_MASK  : B rows mask-compressed, expanded via AVX-512 expand
//
// Dense C output, outer-product traversal.
// Research / benchmarking oriented.
//
// Compile (Skylake+):
//   g++ -O3 -std=c++17 -march=skylake-avx512 -fopenmp spgemm_avx512.cpp -o
//   spgemm
// --------------------------------------------------
#include <algorithm>
#include <cassert>
#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <omp.h>
#include <sstream>
#include <string>
#include <time.h>
#include <vector>

using val_t = double;
constexpr int W = 8; // AVX-512 width (fp32)

// ------------------------
// COO / CSC
// ------------------------
struct COO {
  int i, j;
  val_t v;
};

struct CSC {
  int nrows, ncols;
  std::vector<int> col_ptr;
  std::vector<int> row_idx;
  std::vector<val_t> vals;
};

struct CSR {
  int nrows, ncols;
  std::vector<int> row_ptr;
  std::vector<int> col_idx;
  std::vector<val_t> vals;
};

struct BChunk {
  int j;          // starting column (multiple of W)
  __mmask16 mask; // active lanes
  int vpos;       // offset into packed vals
};

struct BRowChunked {
  std::vector<val_t> vals;    // packed nonzeros
  std::vector<BChunk> chunks; // only non-empty chunks
};

CSR coo_to_csr(int nrows, int ncols, const std::vector<COO> &coo) {
  CSR A{nrows, ncols};

  A.row_ptr.assign(nrows + 1, 0);

  for (const auto &e : coo)
    A.row_ptr[e.i + 1]++;

  for (int r = 0; r < nrows; ++r)
    A.row_ptr[r + 1] += A.row_ptr[r];

  int nnz = coo.size();
  A.col_idx.resize(nnz);
  A.vals.resize(nnz);

  std::vector<int> next = A.row_ptr;

  for (const auto &e : coo) {
    int p = next[e.i]++;
    A.col_idx[p] = e.j;
    A.vals[p] = e.v;
  }

  return A;
}

CSC coo_to_csc(int nrows, int ncols, const std::vector<COO> &coo) {
  CSC A{nrows, ncols};
  A.col_ptr.assign(ncols + 1, 0);
  for (auto &e : coo)
    A.col_ptr[e.j + 1]++;
  for (int c = 0; c < ncols; ++c)
    A.col_ptr[c + 1] += A.col_ptr[c];

  int nnz = coo.size();
  A.row_idx.resize(nnz);
  A.vals.resize(nnz);
  std::vector<int> next = A.col_ptr;

  for (auto &e : coo) {
    int p = next[e.j]++;
    A.row_idx[p] = e.i;
    A.vals[p] = e.v;
  }
  return A;
}

// ------------------------
// B row storage formats
// ------------------------
// Index-based (scalar expand-once)
struct BRowCompressed {
  std::vector<int> idx;
  std::vector<val_t> vals;
};

// Mask-based (AVX-512 expand)
struct BRowMasked {
  std::vector<val_t> vals;     // packed nonzeros
  std::vector<__mmask8> masks; // one mask per W-wide chunk
};

// Fully padded
struct BRowPadded {
  std::vector<val_t> dense;
};

size_t get_csc_mem_bytes(const CSC matrix) {
  size_t pos_bytes = (size_t)(matrix.ncols + 1) * sizeof(int);
  size_t idx_bytes = (size_t)matrix.vals.size() * sizeof(int);
  size_t val_bytes = (size_t)matrix.vals.size() * sizeof(val_t);
  return pos_bytes + idx_bytes + val_bytes;
}

size_t get_masked_mem_bytes(const std::vector<BRowMasked> matrix) {
  size_t val_bytes = 0;
  size_t mask_bytes = 0;
  for (auto row : matrix) {
    val_bytes += (size_t)row.vals.size() * sizeof(val_t);
    mask_bytes += (size_t)row.masks.size() * sizeof(__mmask8);
  }
  return val_bytes + mask_bytes;
}

size_t dense_mem_bytes(int N) { return (size_t)N * N * sizeof(val_t); }

std::vector<COO> read_mtx(const std::string &filename, int &nrows, int &ncols) {
  std::vector<COO> coordinates;
  std::ifstream file(filename);

  if (!file.is_open()) {
    std::cerr << "Error: Could not open file " << filename << std::endl;
    return coordinates;
  }

  std::string line;
  int entries = 0;
  bool isSymmetric = false;

  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '%')
      continue;

    std::istringstream iss(line);
    iss >> nrows >> ncols >> entries;
    break;
  }

  if (nrows == 0 || ncols == 0) {
    std::cerr << "Error: Invalid MTX file format" << std::endl;
    return coordinates;
  }

  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '%')
      continue;

    std::istringstream iss(line);
    int row, col;
    val_t value;
    iss >> row >> col >> value;

    coordinates.push_back({row - 1, col - 1, value});
  }

  return coordinates;
}
// ------------------------
// TNS loader (1-based)
// ------------------------
std::vector<COO> load_tns(const std::string &path, int &nrows, int &ncols) {
  std::ifstream f(path);
  assert(f && "failed to open file");
  std::vector<COO> coo;
  int i, j;
  val_t v;
  nrows = ncols = 0;
  while (f >> i >> j >> v) {
    --i;
    --j;
    coo.push_back({i, j, v});
    nrows = std::max(nrows, i + 1);
    ncols = std::max(ncols, j + 1);
  }
  return coo;
}

// ------------------------
// Build B rows (index-based)
// ------------------------
std::vector<BRowCompressed> build_B_rows(int nrows,
                                         const std::vector<COO> &coo) {
  std::vector<BRowCompressed> rows(nrows);
  for (auto &e : coo) {
    rows[e.i].idx.push_back(e.j);
    rows[e.i].vals.push_back(e.v);
  }
  return rows;
}

// ------------------------
// Build B rows (mask-based)
// ------------------------
std::vector<BRowMasked> build_B_rows_masked(int nrows, int ncols,
                                            const std::vector<COO> &coo) {
  int nchunks = (ncols + W - 1) / W;
  std::vector<BRowMasked> rows(nrows);

  std::vector<std::vector<std::pair<int, val_t>>> tmp(nrows);
  for (auto &e : coo)
    tmp[e.i].push_back({e.j, e.v});

  for (int r = 0; r < nrows; ++r) {
    rows[r].masks.assign(nchunks, 0);
    std::sort(tmp[r].begin(), tmp[r].end(),
              [](auto &a, auto &b) { return a.first < b.first; });

    for (auto &[j, v] : tmp[r]) {
      int chunk = j / W;
      int lane = j % W;
      rows[r].masks[chunk] |= (__mmask8(1) << lane);
      rows[r].vals.push_back(v);
    }

#ifndef NDEBUG
    int pc = 0;
    for (auto m : rows[r].masks)
      pc += _mm_popcnt_u32(m);
    assert(pc == (int)rows[r].vals.size());
#endif
  }
  return rows;
}

std::vector<BRowChunked>
build_B_rows_chunked_from_coo(const std::vector<COO> &coo, int nrows,
                              int ncols) {
  // bucket entries per row
  std::vector<std::vector<COO>> rows(nrows);
  for (const auto &e : coo)
    rows[e.i].push_back(e);
  std::vector<BRowChunked> out(nrows);

  for (int r = 0; r < nrows; ++r) {
    auto &dst = out[r];
    auto &row = rows[r];

    if (row.empty())
      continue;

    // sort by column (required)
    std::sort(row.begin(), row.end(),
              [](const COO &a, const COO &b) { return a.j < b.j; });

    int cur_chunk = -1;
    __mmask16 mask = 0;
    int vpos = 0;
    for (const auto &e : row) {
      int chunk = e.j / W;
      int lane = e.j % W;

      if (chunk != cur_chunk) {
        if (mask != 0) {
          dst.chunks.push_back({cur_chunk * W, mask, vpos});
          vpos += _mm_popcnt_u32(mask);
        }
        cur_chunk = chunk;
        mask = 0;
      }

      mask |= (__mmask16(1) << lane);
      dst.vals.push_back(e.v);
    }

    // flush last chunk
    if (mask != 0) {
      dst.chunks.push_back({cur_chunk * W, mask, vpos});
      vpos += _mm_popcnt_u32(mask);
    }

    // safety padding to allow wide loads
    dst.vals.resize(dst.vals.size() + W, val_t(0));
  }

  return out;
}

CSR spgemm_expand_chunked_rowlocal_avx512(const CSR &A,
                                          const std::vector<BRowChunked> &B,
                                          int ncols) {
  int padded_cols = ((ncols + W - 1) / W) * W;

  CSR C;
  C.nrows = A.nrows;
  C.ncols = ncols;
  C.row_ptr.resize(A.nrows + 1);

#pragma omp parallel
  {
    // thread-local dense buffer
    std::vector<val_t> dense_row(padded_cols, 0.0);

    // thread-local output
    std::vector<int> local_cols;
    std::vector<val_t> local_vals;

#pragma omp for schedule(static)
    for (int i = 0; i < A.nrows; ++i) {
      std::fill(dense_row.begin(), dense_row.end(), 0.0);

      // compute row i
      for (int p = A.row_ptr[i]; p < A.row_ptr[i + 1]; ++p) {
        int k = A.col_idx[p];
        val_t aval = A.vals[p];

        const auto &brow = B[k];
        __m512d a = _mm512_set1_pd(aval);

        for (const auto &ch : brow.chunks) {
          __m512d packed = _mm512_loadu_pd(brow.vals.data() + ch.vpos);
          __m512d b = _mm512_maskz_expand_pd(ch.mask, packed);
          __m512d c = _mm512_loadu_pd(dense_row.data() + ch.j);
          c = _mm512_fmadd_pd(a, b, c);
          _mm512_storeu_pd(dense_row.data() + ch.j, c);
        }
      }

      // compress row i
      int nnz = 0;
      for (int j = 0; j < ncols; ++j) {
        val_t v = dense_row[j];
        if (v != 0.0) {
          local_cols.push_back(j);
          local_vals.push_back(v);
          nnz++;
        }
      }

      C.row_ptr[i + 1] = nnz;
    }

#pragma omp critical
    {
      int base = C.col_idx.size();
      C.col_idx.insert(C.col_idx.end(), local_cols.begin(), local_cols.end());
      C.vals.insert(C.vals.end(), local_vals.begin(), local_vals.end());
    }
  }

  for (int i = 0; i < C.nrows; ++i)
    C.row_ptr[i + 1] += C.row_ptr[i];

  return C;
}

// ------------------------
// PAD preparation
// ------------------------
std::vector<BRowPadded> pad_B_rows(const std::vector<BRowCompressed> &B,
                                   int ncols) {
  int padded_cols = ((ncols + W - 1) / W) * W;
  std::vector<BRowPadded> out(B.size());
  for (size_t r = 0; r < B.size(); ++r) {
    out[r].dense.assign(padded_cols, 0.f);
    for (size_t k = 0; k < B[r].idx.size(); ++k)
      out[r].dense[B[r].idx[k]] = B[r].vals[k];
  }
  return out;
}

// ------------------------
// Expand once (scalar)
// ------------------------
void expand_row_once(const BRowCompressed &row, val_t *dense, int ncols) {
  std::fill(dense, dense + ncols, 0.f);
  for (size_t k = 0; k < row.idx.size(); ++k)
    dense[row.idx[k]] = row.vals[k];
}

// ------------------------
// AVX-512 kernels
// ------------------------

std::vector<val_t>
spgemm_pad_avx512(const CSC &A, const std::vector<BRowPadded> &B, int ncols) {
  int padded_cols = ((ncols + W - 1) / W) * W;
  std::vector<val_t> C(A.nrows * padded_cols, 0.f);

#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int nt = omp_get_num_threads();
    int rows_per = (A.nrows + nt - 1) / nt;
    int i0 = tid * rows_per;
    int i1 = std::min(A.nrows, i0 + rows_per);

    for (int k = 0; k < A.ncols; ++k) {
      const val_t *Brow = B[k].dense.data();
      for (int p = A.col_ptr[k]; p < A.col_ptr[k + 1]; ++p) {
        int i = A.row_idx[p];
        if (i < i0 || i >= i1)
          continue;

        __m512d a = _mm512_set1_pd(A.vals[p]);
        val_t *Crow = &C[i * padded_cols];
        for (int j = 0; j < padded_cols; j += W) {
          __m512d b = _mm512_loadu_pd(Brow + j);
          __m512d c = _mm512_loadu_pd(Crow + j);
          c = _mm512_fmadd_pd(a, b, c);
          _mm512_storeu_pd(Crow + j, c);
        }
      }
    }
  }
  return C;
}

std::vector<val_t>
spgemm_expand_once_avx512(const CSC &A, const std::vector<BRowCompressed> &B,
                          int ncols) {
  int padded_cols = ((ncols + W - 1) / W) * W;
  std::vector<val_t> C(A.nrows * padded_cols, 0.f);

#pragma omp parallel
  {
    std::vector<val_t> Brow_dense(padded_cols);
    int tid = omp_get_thread_num();
    int nt = omp_get_num_threads();
    int rows_per = (A.nrows + nt - 1) / nt;
    int i0 = tid * rows_per;
    int i1 = std::min(A.nrows, i0 + rows_per);

    for (int k = 0; k < A.ncols; ++k) {
      expand_row_once(B[k], Brow_dense.data(), ncols);
      std::fill(Brow_dense.begin() + ncols, Brow_dense.end(), 0.f);

      for (int p = A.col_ptr[k]; p < A.col_ptr[k + 1]; ++p) {
        int i = A.row_idx[p];
        if (i < i0 || i >= i1)
          continue;

        __m512d a = _mm512_set1_pd(A.vals[p]);
        val_t *Crow = &C[i * padded_cols];
        for (int j = 0; j < padded_cols; j += W) {
          __m512d b = _mm512_loadu_pd(Brow_dense.data() + j);
          __m512d c = _mm512_loadu_pd(Crow + j);
          c = _mm512_fmadd_pd(a, b, c);
          _mm512_storeu_pd(Crow + j, c);
        }
      }
    }
  }
  return C;
}

std::vector<val_t>
spgemm_expand_chunked_avx512(const CSC &A, const std::vector<BRowChunked> &B,
                             int ncols) {
  int padded_cols = ((ncols + W - 1) / W) * W;
  std::vector<val_t> C(A.nrows * padded_cols, 0.0);

#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int nt = omp_get_num_threads();
    int rows_per = (A.nrows + nt - 1) / nt;
    int i0 = tid * rows_per;
    int i1 = std::min(A.nrows, i0 + rows_per);

    for (int k = 0; k < A.ncols; ++k) {
      const auto &brow = B[k];

      for (int p = A.col_ptr[k]; p < A.col_ptr[k + 1]; ++p) {
        int i = A.row_idx[p];
        if (i < i0 || i >= i1)
          continue;

        __m512d a = _mm512_set1_pd(A.vals[p]);
        val_t *Crow = &C[i * padded_cols];

        for (const auto &ch : brow.chunks) {
          __m512d packed = _mm512_loadu_pd(brow.vals.data() + ch.vpos);

          __m512d b = _mm512_maskz_expand_pd(ch.mask, packed);

          __m512d c = _mm512_loadu_pd(Crow + ch.j);

          c = _mm512_fmadd_pd(a, b, c);
          _mm512_storeu_pd(Crow + ch.j, c);
        }
      }
    }
  }

  return C;
}

std::vector<val_t> spgemm_expand_mask_avx512(const CSC &A,
                                             const std::vector<BRowMasked> &B,
                                             int ncols) {
  int padded_cols = ((ncols + W - 1) / W) * W;
  std::vector<val_t> C(A.nrows * padded_cols, 0.f);

#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int nt = omp_get_num_threads();
    int rows_per = (A.nrows + nt - 1) / nt;
    int i0 = tid * rows_per;
    int i1 = std::min(A.nrows, i0 + rows_per);

    for (int k = 0; k < A.ncols; ++k) {
      const auto &row = B[k];
      for (int p = A.col_ptr[k]; p < A.col_ptr[k + 1]; ++p) {
        int i = A.row_idx[p];
        if (i < i0 || i >= i1)
          continue;

        __m512d a = _mm512_set1_pd(A.vals[p]);
        val_t *Crow = &C[i * padded_cols];

        int vpos = 0;
        for (int j = 0; j < padded_cols; j += W) {
          __mmask8 m = row.masks[j / W];
          __m512d packed = _mm512_loadu_pd(row.vals.data() + vpos);
          __m512d b = _mm512_maskz_expand_pd(m, packed);
          vpos += _mm_popcnt_u64(m);

          __m512d c = _mm512_loadu_pd(Crow + j);
          c = _mm512_fmadd_pd(a, b, c);
          _mm512_storeu_pd(Crow + j, c);
        }
      }
    }
  }
  return C;
}

std::vector<COO> spgemm_expand_chunked_extract_compress(
    const CSC &A, const std::vector<BRowChunked> &B, int ncols) {
  constexpr int W = 8;
  int padded_cols = ((ncols + W - 1) / W) * W;

  std::vector<COO> out;

#pragma omp parallel
  {
    std::vector<COO> local;
    std::vector<val_t> acc(padded_cols, 0.0);
    std::vector<uint8_t> acc_mask(padded_cols / W, 0);

    int tid = omp_get_thread_num();
    int nt = omp_get_num_threads();
    int rows_per = (A.nrows + nt - 1) / nt;
    int i0 = tid * rows_per;
    int i1 = std::min(A.nrows, i0 + rows_per);

    for (int i = i0; i < i1; ++i) {
      std::fill(acc.begin(), acc.end(), 0.0);
      std::fill(acc_mask.begin(), acc_mask.end(), 0);

      for (int k = 0; k < A.ncols; ++k) {
        for (int p = A.col_ptr[k]; p < A.col_ptr[k + 1]; ++p) {
          if (A.row_idx[p] != i)
            continue;

          __m512d a = _mm512_set1_pd(A.vals[p]);
          const auto &brow = B[k];

          for (const auto &ch : brow.chunks) {
            __m512d packed = _mm512_loadu_pd(brow.vals.data() + ch.vpos);
            __m512d b = _mm512_maskz_expand_pd(ch.mask, packed);

            __m512d c = _mm512_loadu_pd(acc.data() + ch.j);
            c = _mm512_fmadd_pd(a, b, c);
            _mm512_storeu_pd(acc.data() + ch.j, c);

            acc_mask[ch.j / W] = 1;
          }
        }
      }

      // === Vectorized extraction ===
      for (int cj = 0; cj < padded_cols; cj += W) {
        if (!acc_mask[cj / W])
          continue;

        __m512d v = _mm512_loadu_pd(acc.data() + cj);
        __mmask8 nz = _mm512_cmp_pd_mask(v, _mm512_setzero_pd(), _CMP_NEQ_OQ);
        if (!nz)
          continue;

        __m512d packed = _mm512_maskz_compress_pd(nz, v);
        int cnt = _mm_popcnt_u32(nz);

        alignas(64) val_t tmp[W];
        _mm512_store_pd(tmp, packed);

        for (int t = 0; t < cnt; ++t) {
          int lane = _tzcnt_u32(nz);
          nz &= nz - 1;
          local.push_back({i, cj + lane, tmp[t]});
        }
      }
    }

#pragma omp critical
    out.insert(out.end(), local.begin(), local.end());
  }

  return out;
}

double now_ms() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

void save_tns(const std::string &file, size_t N, const std::vector<double> &C) {
  std::ofstream out(file);
  for (size_t i = 0; i < N; i++)
    for (size_t j = 0; j < N; j++)
      if (C[i * N + j] != 0.0f)
        out << i + 1 << " " << j + 1 << " " << C[i * N + j] << "\n";
}

// ------------------------
// Main
// ------------------------
int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "usage: ./spgemm A.tns B.tns pad|expand|expand_mask\n";
    return 1;
  }

  int Ar, Ac, Br, Bc;
  std::string file_name = argv[1];
  /*auto Acoo = load_tns(argv[1], Ar, Ac);*/
  /*auto Bcoo = load_tns(argv[2], Br, Bc);*/
  auto Acoo = read_mtx(argv[1], Ar, Ac);
  auto Bcoo = Acoo;
  Br = Ar;
  Bc = Ac;

  assert(Ac == Br);

  CSC A = coo_to_csc(Ar, Ac, Acoo);
  CSR Acsr = coo_to_csr(Ar, Ar, Acoo);
  auto Brows = build_B_rows(Br, Bcoo);

  std::vector<val_t> C;
  std::string mode = argv[3];

  /*std::string mode = "expand_mask";*/
  if (mode == "pad") {
    auto Bpad = pad_B_rows(Brows, Bc);
    auto t0 = now_ms();
    C = spgemm_pad_avx512(A, Bpad, Bc);
    auto t1 = now_ms();
    std::cout << Ar << "," << Ar << "," << (t1 - t0) << "," << file_name << ","
              << file_name << std::endl;

  } else if (mode == "expand") {
    auto t0 = now_ms();
    C = spgemm_expand_once_avx512(A, Brows, Bc);
    auto t1 = now_ms();
    std::cout << Ar << "," << Ar << "," << (t1 - t0) << "," << file_name << ","
              << file_name << std::endl;

  } else if (mode == "expand_mask") {
    auto t0 = now_ms();
    auto Bmask = build_B_rows_masked(Br, Bc, Bcoo);
    C = spgemm_expand_mask_avx512(A, Bmask, Bc);
    auto t1 = now_ms();
    std::cout << Ar << "," << Ar << "," << (t1 - t0) << "," << file_name << ","
              << file_name;
    // memory data
    size_t C_mem = dense_mem_bytes(Ar);
    size_t A_mem = get_csc_mem_bytes(A);
    size_t B_mem = get_masked_mem_bytes(Bmask);

    std::cout << Ar << "," << Ar << "," << (C_mem + A_mem + B_mem) << ","
              << file_name;

  } else if (mode == "active_chunks") {
    auto t0 = now_ms();
    auto Brows = build_B_rows_chunked_from_coo(Bcoo, Br, Bc);
    C = spgemm_expand_chunked_avx512(A, Brows, Bc);
    auto t1 = now_ms();
    std::cout << Ar << "," << Ar << "," << (t1 - t0) << "," << file_name << ","
              << file_name;
    // memory data
    /*size_t C_mem = dense_mem_bytes(Ar);*/
    /*size_t A_mem = get_csc_mem_bytes(A);*/
    /*size_t B_mem = get_masked_mem_bytes(Bmask);*/

    /*std::cout << Ar << "," << Ar << "," << (C_mem + A_mem + B_mem) << ","*/
    /*          << file_name;*/
  } else if (mode == "csr") {
    auto t0 = now_ms();
    auto Brows = build_B_rows_chunked_from_coo(Bcoo, Br, Bc);
    CSR Ccsr = spgemm_expand_chunked_rowlocal_avx512(Acsr, Brows, Bc);
    auto t1 = now_ms();
    std::cout << Ar << "," << Ar << "," << (t1 - t0) << "," << file_name << ","
              << file_name;

  } else {
    std::cerr << "unknown mode: " << mode << '\n';
    return 1;
  }

  // save_tns(file_name + "_result.tns", Ar, C);
}
