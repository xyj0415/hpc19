#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>

const int NTHREADS = 2;

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = A[0];
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i];
  }
}

void scan_omp(long* prefix_sum, const long* A, long n) {
  // TODO: implement multi-threaded OpenMP scan
  if (n == 0) return;
  int length = n / NTHREADS;
  prefix_sum[0] = 0;
  #pragma omp parallel shared(n) num_threads(NTHREADS)
  {
    int thread_num = omp_get_thread_num();
    // printf("%d\n", thread_num);
    prefix_sum[thread_num * length] = A[thread_num * length];
    for (int i = thread_num * length + 1; i < (thread_num + 1) * length; i++)
    {
      prefix_sum[i] = prefix_sum[i - 1] + A[i];
    }
  }
  // for (long i = 0; i < n; i++)
  //   printf("%d ", prefix_sum[i]);
  // printf("\n");
  long* correction = (long*) malloc((NTHREADS - 1) * sizeof(long));
  correction[0] = prefix_sum[length - 1];
  for (int i = 1; i < NTHREADS - 1; i++)
  {
    correction[i] = correction[i - 1] + prefix_sum[length * (i + 1) - 1];
    // printf("%d ", correction[i]);
  }
  // printf("\n");
  

  #pragma omp parallel shared(n) num_threads(NTHREADS)
  {
    int thread_num = omp_get_thread_num();
    for (int i = thread_num * length; i < (thread_num + 1) * length; i++)
    {
      if (thread_num > 0)
        prefix_sum[i] += correction[thread_num - 1];
    }
  }
}

int main() {
  long N = 100000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  for (long i = 0; i < N; i++) A[i] = rand();

  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  // for (long i = 0; i < N; i++) printf("%d ", A[i]);
  // printf("\n");
  // for (long i = 0; i < N; i++) printf("%d ", B0[i]);
  // printf("\n");
  // for (long i = 0; i < N; i++) printf("%d ", B1[i]);
  // printf("\n");
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}
