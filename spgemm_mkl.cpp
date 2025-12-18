#include <mkl.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <chrono>
#include <algorithm>

struct COO {
    int i, j;
    double v;
};

std::vector<COO> load_tns(const char* path, int& nrows, int& ncols) {
    std::ifstream f(path);
    if (!f) { std::cerr << "cannot open " << path << "\n"; exit(1); }

    std::vector<COO> coo;
    int i, j;
    double v;
    nrows = ncols = 0;

    while (f >> i >> j >> v) {
        --i; --j;
        coo.push_back({i,j,v});
        nrows = std::max(nrows, i+1);
        ncols = std::max(ncols, j+1);
    }
    return coo;
}

void coo_to_csr(int nrows, int ncols,
                const std::vector<COO>& coo,
                std::vector<MKL_INT>& rowptr,
                std::vector<MKL_INT>& colidx,
                std::vector<double>& vals)
{
    rowptr.assign(nrows+1, 0);
    for (auto& e : coo) rowptr[e.i+1]++;
    for (int i=0;i<nrows;i++) rowptr[i+1]+=rowptr[i];

    colidx.resize(coo.size());
    vals.resize(coo.size());
    std::vector<int> next = rowptr;

    for (auto& e : coo) {
        int p = next[e.i]++;
        colidx[p] = e.j;
        vals[p] = e.v;
    }
}
void write_csr_to_tns(const char* path, sparse_matrix_t C) {
    sparse_index_base_t indexing;
    MKL_INT rows, cols;
    MKL_INT *rowptr, *rowptr_end, *colidx;
    double *vals;

    mkl_sparse_d_export_csr(C, &indexing,
                            &rows, &cols,
                            &rowptr, &rowptr_end,
                            &colidx, &vals);

    std::ofstream f(path);
    if (!f) {
        std::cerr << "cannot open output file\n";
        return;
    }

    for (MKL_INT i = 0; i < rows; ++i) {
        for (MKL_INT p = rowptr[i]; p < rowptr_end[i]; ++p) {
            // +1 because TNS is 1-based
            f << (i + 1) << " "
              << (colidx[p] + 1) << " "
              << vals[p] << "\n";
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: ./mkl_spgemm A.tns B.tns\n";
        return 1;
    }

    int Ar, Ac, Br, Bc;
    auto Acoo = load_tns(argv[1], Ar, Ac);
    auto Bcoo = load_tns(argv[2], Br, Bc);
    if (Ac != Br) { std::cerr << "dimension mismatch\n"; return 1; }

    std::vector<MKL_INT> Arp, Aci, Brp, Bci;
    std::vector<double> Aval, Bval;

    coo_to_csr(Ar, Ac, Acoo, Arp, Aci, Aval);
    coo_to_csr(Br, Bc, Bcoo, Brp, Bci, Bval);

    sparse_matrix_t A, B, C;
    mkl_sparse_d_create_csr(&A, SPARSE_INDEX_BASE_ZERO,
                            Ar, Ac, Arp.data(), Arp.data()+1,
                            Aci.data(), Aval.data());

    mkl_sparse_d_create_csr(&B, SPARSE_INDEX_BASE_ZERO,
                            Br, Bc, Brp.data(), Brp.data()+1,
                            Bci.data(), Bval.data());

    matrix_descr desc{};
    desc.type = SPARSE_MATRIX_TYPE_GENERAL;

    mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE,
                    A, B, &C);

    auto t0 = std::chrono::high_resolution_clock::now();

    mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE,
                    A, B, &C);

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1-t0).count();

    std::cout << Ar << "," << Bc << "," << ms
              << "," << argv[1] << "," << argv[2];
    
    //write_csr_to_tns("C.tns", C);

    mkl_sparse_destroy(A);
    mkl_sparse_destroy(B);
    mkl_sparse_destroy(C);
}

