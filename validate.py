import torch

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
    A = load_tns_to_torch(sys.argv[1]).float()
    B = load_tns_to_torch(sys.argv[2]).float()
    C = load_tns_to_torch(sys.argv[3]).float()

    C_torch = torch.sparse.mm(A, B)

    diff = (C_torch.to_dense() - C.to_dense()).abs().max()

    print("Max difference:", diff.item())
    if diff < 1e-5:
        print("✔️ The results match!")
    else:
        print("❌ Difference detected.")

