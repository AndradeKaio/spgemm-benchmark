import time
import torch
import warnings

# Suppress specific PyTorch UserWarning about Sparse CSR tensor support
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Sparse CSR tensor support is in beta state.*",
)



def load_tns_to_torch(filename, device="cpu", dtype=torch.float32):
    rows = []
    cols = []
    vals = []

    with open(filename, "r") as f:
        for line in f:
            i, j, v = line.strip().split()
            rows.append(int(i) - 1)   # convert to 0-based
            cols.append(int(j) - 1)
            vals.append(float(v))

    rows = torch.tensor(rows, dtype=torch.int64)
    cols = torch.tensor(cols, dtype=torch.int64)
    vals = torch.tensor(vals, dtype=dtype)

    indices = torch.stack([rows, cols])     # shape: (2, nnz)

    # Infer matrix dimension from the max index
    N = max(rows.max().item(), cols.max().item()) + 1

    sparse = torch.sparse_coo_tensor(indices, vals, (N, N), device=device)
    sparse = sparse.coalesce()  # important for correct matmul behavior
    return sparse

if __name__ == "__main__":
    import sys
    A_file = sys.argv[1]
    B_file = sys.argv[2]
    A = load_tns_to_torch(A_file).float()
    B = load_tns_to_torch(B_file).float()
    sparse = sys.argv[3] == "0"

    M, K = A.shape[0], B.shape[1]
    if sparse:
        start = time.time()
        C = torch.sparse.mm(A, B)
        end = time.time()
    else:
        A = A.to_dense()
        B = B.to_dense()
        start = time.time()
        C = torch.matmul(A, B)
        end = time.time()
    elapsed_time_millis = (end - start) * 1000
    print(f"{M},{K},{elapsed_time_millis},{A_file},{B_file},{sys.argv[3]}")


