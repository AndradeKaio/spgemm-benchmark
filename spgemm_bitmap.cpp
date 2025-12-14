#include <bitset>
#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include <time.h>
#include <omp.h>

constexpr size_t MAXN = 4096;

double now_ms() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/******************************************************
 * Load A as row-major bitsets + flat values
 ******************************************************/
void load_tns_rowmajor(const std::string& file, size_t N,
                       std::vector<std::bitset<MAXN>>& bits,
                       std::vector<double>& vals)
{
    bits.assign(N, std::bitset<MAXN>());
    vals.assign(N*N, 0.0f);

    std::ifstream in(file);
    if (!in) { std::cerr<<"Cannot open "<<file<<"\n"; exit(1); }

    int i,j; double v;
    while (in >> i >> j >> v) {
        i--; j--;
        bits[i].set(j);
        vals[i*N + j] = v;
    }
}

/******************************************************
 * Load B but construct *column-major* bitsets + values
 ******************************************************/
void load_tns_colmajor(const std::string& file, size_t N,
                       std::vector<std::bitset<MAXN>>& bits,
                       std::vector<double>& vals)
{
    bits.assign(N, std::bitset<MAXN>());    // bitset for each column
    vals.assign(N*N, 0.0f);                 // still store row-major values

    std::ifstream in(file);
    if (!in) { std::cerr<<"Cannot open "<<file<<"\n"; exit(1); }

    int i,j; double v;
    while (in >> i >> j >> v) {
        i--; j--;
        bits[j].set(i);      // <-- transpose pattern
        vals[i*N + j] = v;   // numeric stays row-major
    }
}
void spgemm_sym(size_t N,
            const std::vector<std::bitset<MAXN>>& A_bits,
            const std::vector<std::bitset<MAXN>>& B_bits_col,
            std::vector<std::bitset<MAXN>>& C)
{


#pragma omp parallel for schedule(runtime)
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
		std::bitset<MAXN> r = A_bits[i] & B_bits_col[j];
		C[i] = C[i] & r;
        }
    }
}

/******************************************************
 * SpGEMM with correct A.row & B.col intersection
 ******************************************************/
void spgemm(size_t N,
            const std::vector<std::bitset<MAXN>>& A_bits,
            const std::vector<double>& A_vals,
            const std::vector<std::bitset<MAXN>>& B_bits_col,
            const std::vector<double>& B_vals,
            std::vector<double>& C_vals)
{
    C_vals.assign(N*N, 0.0f);


#pragma omp parallel for schedule(runtime)
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {

            std::bitset<MAXN> mask = A_bits[i] & B_bits_col[j];

            for (size_t k = mask._Find_first();
                 k < N;
                 k = mask._Find_next(k))
            {
                // correct numeric indexing
                C_vals[i*N + j] += A_vals[i*N + k] * B_vals[k*N + j];
            }
        }
    }
}

/******************************************************
 * Save result
 ******************************************************/
void save_tns(const std::string& file, size_t N,
              const std::vector<double>& C)
{
    std::ofstream out(file);
    for (size_t i=0;i<N;i++)
        for (size_t j=0;j<N;j++)
            if (C[i*N+j] != 0.0f)
                out << i+1 << " " << j+1 << " " << C[i*N+j] << "\n";
}

/******************************************************
 * Main
 ******************************************************/
int main(int argc, char** argv)
{
    if (argc < 4) {
        std::cerr<<"Usage: "<<argv[0]<<" A.tns B.tns N\n";
        return 1;
    }

    std::string Afile = argv[1];
    std::string Bfile = argv[2];
    size_t N = std::stoul(argv[3]);

    std::vector<std::bitset<MAXN>> A_bits, B_bits_col, C_bits;
    std::vector<double> A_vals, B_vals, C_vals;

    load_tns_rowmajor(Afile, N, A_bits, A_vals);
    load_tns_colmajor(Bfile, N, B_bits_col, B_vals);

    auto t0 = now_ms();
    spgemm(N, A_bits, A_vals, B_bits_col, B_vals, C_vals);
    //spgemm_sym(N, A_bits, B_bits_col, C_bits);
    auto t1 = now_ms();
    //printf("%d,%d,%.6f, %s, %s", N, N, (t1 - t0), Afile, Bfile);
    std::cout << N << "," << N << "," << (t1-t0) << "," << Afile << "," << Bfile << std::endl;
    //size_t mem_bitsets = A_bits.size() * (MAXN / 8);
    //size_t mem_vals    = A_vals.size() * sizeof(double);
    //size_t b_mem_bitsets = B_bits_col.size() * (MAXN / 8);
    //size_t b_mem_vals    = B_vals.size() * sizeof(double);
    //size_t c_mem_bitsets = C_bits.size() * (MAXN / 8);
    //size_t c_mem_vals    = C_vals.size() * sizeof(double);
    //std::cout << N <<"," << N << ","
   //       << (mem_bitsets + mem_vals + b_mem_bitsets + b_mem_vals + c_mem_bitsets + c_mem_vals) << "," << Afile << std::endl;

    //save_tns(Bfile + "_bitset_result.tns", N, C_vals);
    return 0;
}

