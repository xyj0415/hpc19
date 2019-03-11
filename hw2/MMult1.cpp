// g++ -fopenmp -O3 -march=native MMult1.cpp && ./a.out
#include <cassert>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "utils.h"

#define BLOCK_SIZE 48

// Note: matrices are stored in column major order; i.e. the array elements in
// the (m x n) matrix C are stored in the sequence: {C_00, C_10, ..., C_m0,
// C_01, C_11, ..., C_m1, C_02, ..., C_0n, C_1n, ..., C_mn}
void MMult0(long m, long n, long k, double *a, double *b, double *c) {
  for (long j = 0; j < n; j++) {
    for (long p = 0; p < k; p++) {
      for (long i = 0; i < m; i++) {
        double A_ip = a[i+p*m];
        double B_pj = b[p+j*k];
        double C_ij = c[i+j*m];
        C_ij = C_ij + A_ip * B_pj;
        c[i+j*m] = C_ij;
      }
    }
  }
}

void MMultRearrange(long m, long n, long k, double *a, double *b, double *c) {
  // TODO: See instructions below
  for (long p = 0; p < k; p++)
  {
    for (long i = 0; i < m; i++)
    {
      for (long j = 0; j < n; j++)
      {
        double A_ip = a[i + p * m];
        double B_pj = b[p + j * k];
        double C_ij = c[i + j * m];
        C_ij = C_ij + A_ip * B_pj;
        c[i + j * m] = C_ij;
      }
    }
  }
}

void read_block(double* dest, double* src, long x, long y, long m)
{
  for (long i = 0; i < BLOCK_SIZE; i++)
    for (long j = 0; j < BLOCK_SIZE; j++)
      dest[i + j * BLOCK_SIZE] = src[(i + x * BLOCK_SIZE) + (j + y * BLOCK_SIZE) * m];
}

void write_block(double* dest, double* src, long x, long y, long m)
{
  for (long i = 0; i < BLOCK_SIZE; i++)
    for (long j = 0; j < BLOCK_SIZE; j++)
      dest[(i + x * BLOCK_SIZE) + (j + y * BLOCK_SIZE) * m] = src[i + j * BLOCK_SIZE];
}

// Reference: http://web.cs.ucdavis.edu/~bai/ECS231/optmatmul.pdf
// Pseudocode on page 20
void MMultBlocked(long m, long n, long k, double *a, double *b, double *c)
{
  int M = m / BLOCK_SIZE, N = n / BLOCK_SIZE, K = k / BLOCK_SIZE;
  for (long i = 0; i < M; i++)
  {
    for (long j = 0; j < N; j++)
    {
      double *temp_c = (double *)malloc(BLOCK_SIZE * BLOCK_SIZE * sizeof(double));
      read_block(temp_c, c, i, j, m);
      for (long p = 0; p < K; p++)
      {
        double *temp_a = (double *)malloc(BLOCK_SIZE * BLOCK_SIZE * sizeof(double));
        double *temp_b = (double *)malloc(BLOCK_SIZE * BLOCK_SIZE * sizeof(double));
        read_block(temp_a, a, i, p, m);
        read_block(temp_b, b, p, j, k);
        MMult0(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, temp_a, temp_b, temp_c);
        free(temp_a);
        free(temp_b);
      }
      write_block(c, temp_c, i, j, m);
      free(temp_c);
    }
  }
}

void MMultOpenMP(long m, long n, long k, double *a, double *b, double *c)
{
  #pragma omp parallel for num_threads(4) shared(a, b, c)
  for (long j = 0; j < n; j++)
  {
    for (long p = 0; p < k; p++)
    {
      for (long i = 0; i < m; i++)
      {
        double A_ip = a[i + p * m];
        double B_pj = b[p + j * k];
        double C_ij = c[i + j * m];
        C_ij = C_ij + A_ip * B_pj;
        c[i + j * m] = C_ij;
      }
    }
  }
}

int main(int argc, char **argv)
{
  const long PFIRST = BLOCK_SIZE;
  const long PLAST = 2000;
  const long PINC = std::max(50 / BLOCK_SIZE, 1) * BLOCK_SIZE; // multiple of BLOCK_SIZE

  printf(" Dimension   RefTime    BlockTime  OMPTime    Error\n");
  for (long p = PFIRST; p < PLAST; p += PINC)
  {
    long m = p, n = p, k = p;
    long NREPEATS = 1e8 / (m * n * k) + 1;
    double *a = (double *)aligned_malloc(m * k * sizeof(double));     // m x k
    double *b = (double *)aligned_malloc(k * n * sizeof(double));     // k x n
    double *c_1 = (double *)aligned_malloc(m * n * sizeof(double));   // m x n
    double *c_2 = (double *)aligned_malloc(m * n * sizeof(double));   // m x n
    double *c_ref = (double *)aligned_malloc(m * n * sizeof(double)); // m x n

    // Initialize matrices
    for (long i = 0; i < m * k; i++)
      a[i] = drand48();
    for (long i = 0; i < k * n; i++)
      b[i] = drand48();
    for (long i = 0; i < m * n; i++)
      c_ref[i] = 0;
    for (long i = 0; i < m * n; i++)
      c_1[i] = 0;
    for (long i = 0; i < m * n; i++)
      c_2[i] = 0;

    Timer t;
    t.tic();
    for (long rep = 0; rep < NREPEATS; rep++)
    { // Compute reference solution
      MMult0(m, n, k, a, b, c_ref);
    }
    double time = t.toc();
    printf("%10d %10f", p, time);

    t.tic();
    for (long rep = 0; rep < NREPEATS; rep++)
    {
      MMultBlocked(m, n, k, a, b, c_1);
    }
    time = t.toc();
    printf(" %10f", time);

    t.tic();
    for (long rep = 0; rep < NREPEATS; rep++)
    {
      MMultOpenMP(m, n, k, a, b, c_2);
    }
    time = t.toc();
    printf(" %10f", time);

    double max_err = 0;
    for (long i = 0; i < m * n; i++)
      max_err = std::max(max_err, fabs(c_1[i] - c_ref[i]));
    printf(" %10f\n", max_err);

    aligned_free(a);
    aligned_free(b);
    aligned_free(c_1);
    aligned_free(c_2);
    aligned_free(c_ref);
  }

  return 0;
  }

  // * Using MMult0 as a reference, implement MMult1 and try to rearrange loops to
  // maximize performance. Measure performance for different loop arrangements and
  // try to reason why you get the best performance for a particular order?
  //
  //
  // * You will notice that the performance degrades for larger matrix sizes that
  // do not fit in the cache. To improve the performance for larger matrices,
  // implement a one level blocking scheme by using BLOCK_SIZE macro as the block
  // size. By partitioning big matrices into smaller blocks that fit in the cache
  // and multiplying these blocks together at a time, we can reduce the number of
  // accesses to main memory. This resolves the main memory bandwidth bottleneck
  // for large matrices and improves performance.
  //
  // NOTE: You can assume that the matrix dimensions are multiples of BLOCK_SIZE.
  //
  //
  // * Experiment with different values for BLOCK_SIZE (use multiples of 4) and
  // measure performance.  What is the optimal value for BLOCK_SIZE?
  //
  //
  // * Now parallelize your matrix-matrix multiplication code using OpenMP.
  //
  //
  // * What percentage of the peak FLOP-rate do you achieve with your code?
  //
  //
  // NOTE: Compile your code using the flag -march=native. This tells the compiler
  // to generate the best output using the instruction set supported by your CPU
  // architecture. Also, try using either of -O2 or -O3 optimization level flags.
  // Be aware that -O2 can sometimes generate better output than using -O3 for
  // programmer optimized code.
