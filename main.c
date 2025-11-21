//#include "taco_kernel.h"
#include "taco_kernel_opt.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>


double now_ms() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

char *make_result_filename(const char *inputB) {
  size_t len = strlen(inputB);
  const char *suffix = "_result.tns";

  // remove extension if .tns is present
  const char *dot = strrchr(inputB, '.');
  size_t base_len =
      (dot && strcmp(dot, ".tns") == 0) ? (size_t)(dot - inputB) : len;

  char *out = malloc(base_len + strlen(suffix) + 1);
  memcpy(out, inputB, base_len);
  strcpy(out + base_len, suffix);
  return out;
}

// Save a 2-D CSR taco_tensor_t to a .tns file (1-based indices).
// Assumes A is in CSR format (sparse,sparse) exactly like TACO SpGEMM output.
int save_tns(const taco_tensor_t *A, const char *filename) {
  FILE *f = fopen(filename, "w");
  if (!f) {
    fprintf(stderr, "ERROR: cannot open %s for writing\n", filename);
    return 0;
  }

  // rowptr = A->indices[1][0]
  // colidx = A->indices[1][1]
  // values = A->vals
  int *rowptr = (int *)A->indices[1][0];
  int *colidx = (int *)A->indices[1][1];
  double *vals = (double *)A->vals;

  int N = A->dimensions[0]; // number of rows (square N x N)

  for (int i = 0; i < N; i++) {
    for (int p = rowptr[i]; p < rowptr[i + 1]; p++) {
      int j = colidx[p];
      double v = vals[p];

      // Output as 1-based indexing: "i j v"
      fprintf(f, "%d %d %.17g\n", i + 1, j + 1, v);
    }
  }

  fclose(f);
  return 1;
}

// Load a TNS sparse matrix into COO arrays.
// Automatically allocates row[], col[], val[].
// Sets nnz_out = number of entries.
//
// TNS Format: "i j value"   (1-based indices)
int load_tns(const char *filename, int **row_out, int **col_out,
             double **val_out, int *nnz_out, int *max_i_out, int *max_j_out) {
  FILE *f = fopen(filename, "r");
  if (!f) {
    printf("ERROR: Cannot open %s\n", filename);
    return 0;
  }

  int capacity = 1024;
  int count = 0;

  int *row = (int *)malloc(capacity * sizeof(int));
  int *col = (int *)malloc(capacity * sizeof(int));
  double *val = (double *)malloc(capacity * sizeof(double));

  int i, j;
  double v;

  int max_i = 0, max_j = 0;

  while (fscanf(f, "%d %d %lf", &i, &j, &v) == 3) {

    if (count == capacity) {
      capacity *= 2;
      row = (int *)realloc(row, capacity * sizeof(int));
      col = (int *)realloc(col, capacity * sizeof(int));
      val = (double *)realloc(val, capacity * sizeof(double));
    }

    row[count] = i - 1; // convert to 0-based
    col[count] = j - 1;
    val[count] = v;
    count++;

    if (i > max_i)
      max_i = i;
    if (j > max_j)
      max_j = j;
  }

  fclose(f);

  *row_out = row;
  *col_out = col;
  *val_out = val;
  *nnz_out = count;
  *max_i_out = max_i;
  *max_j_out = max_j;

  return 1;
}

void test_small_4x4() {
  printf("\n===== SANITY TEST 4x4 =====\n");

  int N = 4;

  int dims[2] = {N, N};
  int order[2] = {0, 1};
  taco_mode_t modes[2] = {taco_mode_sparse, taco_mode_sparse};

  taco_tensor_t *B = init_taco_tensor_t(2, sizeof(double), dims, order, modes);
  taco_tensor_t *C = init_taco_tensor_t(2, sizeof(double), dims, order, modes);
  taco_tensor_t *A = init_taco_tensor_t(2, sizeof(double), dims, order, modes);

  // Minimal allocations
  B->indices[1][0] = (uint8_t *)malloc(sizeof(int) * (N + 1));
  B->indices[1][1] = (uint8_t *)malloc(sizeof(int));
  B->vals = (uint8_t *)malloc(sizeof(double));

  C->indices[1][0] = (uint8_t *)malloc(sizeof(int) * (N + 1));
  C->indices[1][1] = (uint8_t *)malloc(sizeof(int));
  C->vals = (uint8_t *)malloc(sizeof(double));

  A->indices[1][0] = (uint8_t *)malloc(sizeof(int) * (N + 1));
  A->indices[1][1] = (uint8_t *)malloc(sizeof(int));
  A->vals = (uint8_t *)malloc(sizeof(double));

  // ------------------------------------------------------------
  // B matrix in COO
  // ------------------------------------------------------------
  int Brow[] = {0, 0, 1, 1, 2, 3};
  int Bcol[] = {0, 1, 1, 2, 3, 0};
  double Bval[] = {1, 2, 3, 4, 5, 6};
  int Bnnz = 6;

  // C matrix in COO
  int Crow[] = {0, 1, 2, 3};
  int Ccol[] = {0, 1, 2, 0};
  double Cval[] = {7, 8, 9, 10};
  int Cnnz = 4;

  // Build COO1_pos: {0, nnz}
  int Bpos_arr[2] = {0, Bnnz};
  int Cpos_arr[2] = {0, Cnnz};

  pack_B(B, Bpos_arr, Brow, Bcol, Bval);
  pack_C(C, Cpos_arr, Crow, Ccol, Cval);

  // ------------------------------------------------------------
  // Assemble + compute
  // ------------------------------------------------------------
  assemble(A, B, C);
  compute(A, B, C);

  // ------------------------------------------------------------
  // Print CSR result A
  // ------------------------------------------------------------
  int *Apos = (int *)A->indices[1][0];
  int *Aidx = (int *)A->indices[1][1];
  double *Aval = (double *)A->vals;

  printf("A (CSR format):\n");
  for (int i = 0; i < N; i++) {
    printf("row %d: ", i);
    for (int p = Apos[i]; p < Apos[i + 1]; p++) {
      printf("(%d, %.0f) ", Aidx[p], Aval[p]);
    }
    printf("\n");
  }

  printf("===== END SANITY TEST =====\n\n");
}

// Generate random COO matrix (C-style)
void generate_random_coo(int N, double density, int **row_out, int **col_out,
                         double **val_out, int *nnz_out) {
  int capacity = (int)(N * N * density * 1.2) + 16;
  int *row = (int *)malloc(capacity * sizeof(int));
  int *col = (int *)malloc(capacity * sizeof(int));
  double *val = (double *)malloc(capacity * sizeof(double));

  int nnz = 0;

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      double r = (double)rand() / RAND_MAX;
      if (r < density) {
        if (nnz == capacity) { // resize
          capacity *= 2;
          row = (int *)realloc(row, capacity * sizeof(int));
          col = (int *)realloc(col, capacity * sizeof(int));
          val = (double *)realloc(val, capacity * sizeof(double));
        }
        row[nnz] = i;
        col[nnz] = j;
        val[nnz] = (double)rand() / RAND_MAX;
        nnz++;
      }
    }
  }

  *row_out = row;
  *col_out = col;
  *val_out = val;
  *nnz_out = nnz;
}

// Create an empty CSR A (TACO requires minimal memory before assemble())
taco_tensor_t *make_empty_A(int N) {
  int dims[2] = {N, N};
  int order[2] = {0, 1};
  taco_mode_t modes[2] = {taco_mode_sparse, taco_mode_sparse};

  taco_tensor_t *A = init_taco_tensor_t(2, sizeof(double), dims, order, modes);

  A->indices[1][0] = (uint8_t *)malloc(sizeof(int) * (N + 1)); // rowptr
  A->indices[1][1] = (uint8_t *)malloc(sizeof(int));           // colidx dummy
  A->vals = (uint8_t *)malloc(sizeof(double));                 // values dummy

  int *Apos = (int *)A->indices[1][0];
  for (int i = 0; i <= N; i++)
    Apos[i] = 0;

  return A;
}

// Helper to pack one COO matrix into TACO B or C
void pack_into_taco(taco_tensor_t *T, int *row, int *col, double *val, int nnz,
                    int isB) {
  // COO1_pos = {0, nnz}
  int *COO1_pos = (int *)malloc(2 * sizeof(int));
  COO1_pos[0] = 0;
  COO1_pos[1] = nnz;

  int *COO1_crd = (int *)malloc(nnz * sizeof(int)); // row
  int *COO2_crd = (int *)malloc(nnz * sizeof(int)); // col
  double *COO_vals = (double *)malloc(nnz * sizeof(double));

  for (int i = 0; i < nnz; i++) {
    COO1_crd[i] = row[i];
    COO2_crd[i] = col[i];
    COO_vals[i] = val[i];
  }

  if (isB)
    pack_B(T, COO1_pos, COO1_crd, COO2_crd, COO_vals);
  else
    pack_C(T, COO1_pos, COO1_crd, COO2_crd, COO_vals);

  free(COO1_pos);
  free(COO1_crd);
  free(COO2_crd);
  free(COO_vals);
}

int main_random(int argc, char **argv) {
  if (argc < 3) {
    printf("Usage: ./spgemm N sparsity\n");
    return 1;
  }

  int N = atoi(argv[1]);
  double sparsity = atof(argv[2]);
  double density = 1.0f - sparsity;

  srand(42);

  // -------------------------------
  // 1. Generate COO matrices
  // -------------------------------
  int *Brow, *Bcol, Bnnz;
  double *Bval;

  int *Crow, *Ccol, Cnnz;
  double *Cval;

  generate_random_coo(N, density, &Brow, &Bcol, &Bval, &Bnnz);
  generate_random_coo(N, density, &Crow, &Ccol, &Cval, &Cnnz);

  // printf("B nnz = %d\n", Bnnz);
  // printf("C nnz = %d\n", Cnnz);

  // -------------------------------
  // 2. Create TACO tensors
  // -------------------------------
  int dims[2] = {N, N};
  int order[2] = {0, 1};
  // taco_mode_t modes[2] = {taco_mode_sparse, taco_mode_sparse};
  taco_mode_t modes[2] = {taco_mode_dense, taco_mode_sparse};

  taco_tensor_t *B = init_taco_tensor_t(2, sizeof(double), dims, order, modes);
  taco_tensor_t *C = init_taco_tensor_t(2, sizeof(double), dims, order, modes);
  taco_tensor_t *A = make_empty_A(N);

  // Minimal allocations for B and C (TACO expects them before pack)
  B->indices[1][0] = (uint8_t *)malloc(sizeof(int) * (N + 1));
  B->indices[1][1] = (uint8_t *)malloc(sizeof(int));
  B->vals = (uint8_t *)malloc(sizeof(double));

  C->indices[1][0] = (uint8_t *)malloc(sizeof(int) * (N + 1));
  C->indices[1][1] = (uint8_t *)malloc(sizeof(int));
  C->vals = (uint8_t *)malloc(sizeof(double));

  // -------------------------------
  // 3. Pack CSR B and C
  // -------------------------------
  pack_into_taco(B, Brow, Bcol, Bval, Bnnz, 1);
  pack_into_taco(C, Crow, Ccol, Cval, Cnnz, 0);

  // -------------------------------
  // 4. Assemble + compute
  // -------------------------------
  clock_t t0 = clock();
  assemble(A, B, C);
  compute(A, B, C);
  clock_t t1 = clock();

  double elapsed_ms = 1000.0 * (t1 - t0) / CLOCKS_PER_SEC;
  double time_taken =
      ((double)(t1 - t0)) / CLOCKS_PER_SEC; // Calculate in seconds

  int *Apos = (int *)A->indices[1][0];
  int Annz = Apos[N];

  printf("%d,%d,%.2f,%.2f", N, N, sparsity, time_taken);

  // test_small_4x4();
  return 0;
}

int main_load(int argc, char **argv) {
  if (argc < 3) {
    printf("Usage: ./spgemm_taco B.tns C.tns\n");
    return 1;
  }

  const char *Bfile = argv[1];
  const char *Cfile = argv[2];

  int *Brow, *Bcol, Bnnz;
  double *Bval;
  int Bmax_i, Bmax_j;

  int *Crow, *Ccol, Cnnz;
  double *Cval;
  int Cmax_i, Cmax_j;

  // Load the two matrices
  //printf("Loading %s...\n", Bfile);
  load_tns(Bfile, &Brow, &Bcol, &Bval, &Bnnz, &Bmax_i, &Bmax_j);

  //printf("Loading %s...\n", Cfile);
  load_tns(Cfile, &Crow, &Ccol, &Cval, &Cnnz, &Cmax_i, &Cmax_j);

  // Determine N from largest index
  int N = Bmax_i;
  if (Cmax_i > N)
    N = Cmax_i;
  if (Bmax_j > N)
    N = Bmax_j;
  if (Cmax_j > N)
    N = Cmax_j;

  //printf("Matrix size inferred: %dx%d\n", N, N);
  //printf("B nnz = %d\n", Bnnz);
  //printf("C nnz = %d\n", Cnnz);

  int dims[2] = {N, N};
  int order[2] = {0, 1};
  taco_mode_t modes[2] = {taco_mode_sparse, taco_mode_sparse};

  taco_tensor_t *B = init_taco_tensor_t(2, sizeof(double), dims, order, modes);
  taco_tensor_t *C = init_taco_tensor_t(2, sizeof(double), dims, order, modes);
  taco_tensor_t *A = make_empty_A(N);

  B->indices[1][0] = (uint8_t *)malloc(sizeof(int) * (N + 1));
  B->indices[1][1] = (uint8_t *)malloc(sizeof(int));
  B->vals = (uint8_t *)malloc(sizeof(double));

  C->indices[1][0] = (uint8_t *)malloc(sizeof(int) * (N + 1));
  C->indices[1][1] = (uint8_t *)malloc(sizeof(int));
  C->vals = (uint8_t *)malloc(sizeof(double));

  A->indices[1][0] = (uint8_t *)malloc(sizeof(int) * (N + 1));
  A->indices[1][1] = (uint8_t *)malloc(sizeof(int));
  A->vals = (uint8_t *)malloc(sizeof(double));

  // COO1_pos arrays
  int Bpos_arr[2] = {0, Bnnz};
  int Cpos_arr[2] = {0, Cnnz};

  pack_into_taco(B, Brow, Bcol, Bval, Bnnz, 1);
  pack_into_taco(C, Crow, Ccol, Cval, Cnnz, 0);

  double t0 = now_ms();
  assemble(A, B, C);
  compute(A, B, C);
  double t1 = now_ms();

  char *outname = make_result_filename(Bfile);
  //save_tns(A, outname);
  printf("%d,%d,%.6f", N, N, (t1 - t0));
  return 0;
}

int main(int argc, char **argv) {
  // main_random(argc, argv);
  main_load(argc, argv);
  return 0;
}
