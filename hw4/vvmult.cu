#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>

void reduction(double* sum_ptr, const double* a, long N){
  double sum = 0;
  #pragma omp parallel for schedule(static) reduction(+:sum)
  for (long i = 0; i < N; i++) sum += a[i];
  *sum_ptr = sum;
}

void Check_CUDA_Error(const char *message){
  cudaError_t error = cudaGetLastError();
  if(error!=cudaSuccess) {
    fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
    exit(-1);
  }
}

__global__
void vec_mult_kernel(double* c, const double* a, const double* b, long N){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) c[idx] = a[idx] * b[idx];
}

#define BLOCK_SIZE 1024

// Warp divergence
__global__ void reduction_kernel0(double* sum, const double* a, long N){
  __shared__ double smem[BLOCK_SIZE];
  int idx = (blockIdx.x) * blockDim.x + threadIdx.x;

  if (idx < N) smem[threadIdx.x] = a[idx];
  else smem[threadIdx.x] = 0;

  __syncthreads();
  if (threadIdx.x %   2 == 0) smem[threadIdx.x] += smem[threadIdx.x + 1];
  __syncthreads();
  if (threadIdx.x %   4 == 0) smem[threadIdx.x] += smem[threadIdx.x + 2];
  __syncthreads();
  if (threadIdx.x %   8 == 0) smem[threadIdx.x] += smem[threadIdx.x + 4];
  __syncthreads();
  if (threadIdx.x %  16 == 0) smem[threadIdx.x] += smem[threadIdx.x + 8];
  __syncthreads();
  if (threadIdx.x %  32 == 0) smem[threadIdx.x] += smem[threadIdx.x + 16];
  __syncthreads();
  if (threadIdx.x %  64 == 0) smem[threadIdx.x] += smem[threadIdx.x + 32];
  __syncthreads();
  if (threadIdx.x % 128 == 0) smem[threadIdx.x] += smem[threadIdx.x + 64];
  __syncthreads();
  if (threadIdx.x % 256 == 0) smem[threadIdx.x] += smem[threadIdx.x + 128];
  __syncthreads();
  if (threadIdx.x % 512 == 0) smem[threadIdx.x] += smem[threadIdx.x + 256];
  __syncthreads();
  if (threadIdx.x == 0) sum[blockIdx.x] = smem[threadIdx.x] + smem[threadIdx.x + 512];
}

void vvmult(double* x, double* y, double* z, double* sum, const long N)
{
  double *x_2, *y_2, *z_2;
  cudaMalloc(&x_2, N*sizeof(double));
  cudaMalloc(&y_2, N*sizeof(double));
  cudaMalloc(&z_2, N*sizeof(double));
  cudaMemcpyAsync(x_2, x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(y_2, y, N*sizeof(double), cudaMemcpyHostToDevice);
  vec_mult_kernel<<<N/BLOCK_SIZE+1,BLOCK_SIZE>>>(z_2, x_2, y_2, N);
  cudaMemcpyAsync(z, z_2, N*sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  double *x_d, *y_d;
  cudaMalloc(&x_d, N*sizeof(double));
  long N_work = 1;
  for (long i = (N+BLOCK_SIZE-1)/(BLOCK_SIZE); i > 1; i = (i+BLOCK_SIZE-1)/(BLOCK_SIZE)) N_work += i;
  cudaMalloc(&y_d, N_work*sizeof(double)); // extra memory buffer for reduction across thread-blocks

  cudaMemcpyAsync(x_d, z, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  double tt = omp_get_wtime();
  double* sum_d = y_d;
  long Nb = (N+BLOCK_SIZE-1)/(BLOCK_SIZE);
  reduction_kernel0<<<Nb,BLOCK_SIZE>>>(sum_d, x_d, N);
  while (Nb > 1) {
    long N = Nb;
    Nb = (Nb+BLOCK_SIZE-1)/(BLOCK_SIZE);
    reduction_kernel0<<<Nb,BLOCK_SIZE>>>(sum_d + N, sum_d, N);
    sum_d += N;
  }


  cudaMemcpyAsync(sum, sum_d, 1*sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  printf("GPU Bandwidth = %f GB/s\n", 1*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);
  cudaFree(x_2);
  cudaFree(y_2);
  cudaFree(z_2);
  cudaFree(x_d);
  cudaFree(y_d);
}

int main() {
  long N = (1UL<<25);

  double *x, *y, *z;
  cudaMallocHost((void**)&x, N * sizeof(double));
  cudaMallocHost((void**)&y, N * sizeof(double));
  cudaMallocHost((void**)&z, N * sizeof(double));
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < N; i++) 
  {
      x[i] = 1.0 / (i + 1);
      y[i] = 1.0 / (i + 2);
      z[i] = x[i] * y[i];
  }


  double sum_ref, sum;
  double tt = omp_get_wtime();
  reduction(&sum_ref, z, N);
  printf("CPU Bandwidth = %f GB/s\n", 1*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);

  vvmult(x, y, z, &sum, N);
  printf("Error = %f\n", fabs(sum-sum_ref));

  cudaFreeHost(x);
  cudaFreeHost(y);
  cudaFreeHost(z);

  return 0;
}

