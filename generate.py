import numpy as np
import scipy.sparse as sp

# -----------------------------------------------------------
# Save CSR sparse matrix in .tns format (1-based: i j value)
# -----------------------------------------------------------
def save_tns(filename, mat):
    mat = mat.tocsr()
    rowptr = mat.indptr
    colidx = mat.indices
    data = mat.data

    with open(filename, "w") as f:
        for i in range(mat.shape[0]):
            for p in range(rowptr[i], rowptr[i+1]):
                j = colidx[p]
                v = data[p]
                f.write(f"{i+1} {j+1} {v:.17g}\n")


# -----------------------------------------------------------
# Generate random sparse matrix with given sparsity
# -----------------------------------------------------------
def generate_sparse_matrix(N, sparsity):
    """
    sparsity = fraction of entries that ARE non-zero.
    Example: sparsity=0.1 → 10% non-zero.
    """
    nnz = int(N * N * sparsity)

    rows = np.random.randint(0, N, size=nnz)
    cols = np.random.randint(0, N, size=nnz)
    vals = np.random.rand(nnz)

    mat = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
    mat.sum_duplicates()
    return mat


# -----------------------------------------------------------
# Main loop: generate matrices & their products
# -----------------------------------------------------------
def generate(N):
    sparsity_levels = [i / 10 for i in range(1, 10)]   # 0.1 to 0.9

    for s in sparsity_levels:
        print(f"Generating N={N}, sparsity={s:.1f}")

        A = generate_sparse_matrix(N, s)

        # Save the generated matrix
        fname_A = f"{N}_{N}_{int(s*100)}.tns"
        save_tns(fname_A, A)
        print(f"  Saved input matrix to {fname_A}")

        # Multiply A @ A
        C = A @ A

        # Save the product
        fname_C = f"{N}_{N}_{int(s*100)}_prod.tns"
        save_tns(fname_C, C)
        print(f"  Saved product matrix to {fname_C}\n")


if __name__ == "__main__":
    # You can generate sparse matrices with controlled sparsity.
    # Matrices are stored in a tns file alongside with their product.
    # _prod  = A @ A
    import sys
    generate(int(sys.argv[1]))
