#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#ifdef _OPENMP
#include "omp.h"
#else
#define omp_get_thread_num() 0
#endif


using namespace std;

const int BLOCK_SIZE = 1024;
const int N = 2000;
const double h = 1.0 / (N + 1);
const double sqrh = h * h;
const int MAX_ITER = 2000;

// double calc_residue(double *u, double *f)
// {
//     double ans = 0, a_times_u = 0;
//     int i, j;
//     #pragma omp parallel for num_threads(NTHREADS) shared(u, f) reduction(+ : ans)
//     for (i = 1; i <= N; i++)
//         for (j = 1; j <= N; j++)
//         {
//             int pos = i * (N + 2) + j;
//             a_times_u = -(-4 * u[pos] + u[pos - 1] + u[pos + 1] + u[pos - N - 2] + u[pos + N + 2]) / sqrh;
//             ans += (a_times_u - f[pos]) * (a_times_u - f[pos]);
//         }
//     return sqrt(ans);
// }

__global__ void update(const double* u, const double* f, double* u2, long N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = idx / (N + 2), j = idx % (N + 2);
    if (i > 0 && j > 0 && i < N + 1 && j < N + 1)
        u2[idx] = (sqrh * f[idx] + u[idx - 1] + u[idx + 1] + u[idx - N - 2] + u[idx + N + 2]) / 4;
}

void jacobi(double* u, double* f)
{
    // double init_residue = calc_residue(u, f), residue = init_residue;
    double *u2;
    cudaMalloc((void**)&u2, (N + 2) * (N + 2) * sizeof(double));
    double *u_d, *f_d;
    cudaMalloc((void**)&u_d, (N + 2) * (N + 2) * sizeof(double));
    cudaMalloc((void**)&f_d, (N + 2) * (N + 2) * sizeof(double));
    cudaMemcpyAsync(f_d, f, (N + 2) * (N + 2) * sizeof(double), cudaMemcpyHostToDevice);
    long Nb = ((N + 2) * (N + 2) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // for (int iter = 0; iter < MAX_ITER && residue * 1e6 > init_residue; iter++)
    for (int iter = 0; iter < MAX_ITER; iter++)
    {
        cudaMemcpy(u_d, u, (N + 2) * (N + 2) * sizeof(double), cudaMemcpyHostToDevice);
        update<<<Nb,BLOCK_SIZE>>>(u_d, f_d, u2, N);
        cudaMemcpy(u_d, u2, (N + 2) * (N + 2) * sizeof(double), cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();
        if (iter % 100 == 0){
        //     cudaMemcpy(u, u2, (N + 2) * (N + 2) * sizeof(double), cudaMemcpyDeviceToHost);
        //     residue = calc_residue(u, f);
        //     cout << residue << endl;
            cout << iter << endl;
        }
    }
    cudaMemcpy(u, u2, (N + 2) * (N + 2) * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(u2);
    cudaFree(u_d);
    cudaFree(f_d);
    // residue = calc_residue(u, f);
    // printf("Iter:%d Residue:%f\n", MAX_ITER, residue);
}

__global__ void initialize(double* u, double* f, long N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        u[idx] = 0;
        f[idx] = 1;
    }
}

int main(int argc, char const *argv[])
{
    double *u, *f;
    cudaMallocHost((void**)&u, (N + 2) * (N + 2) * sizeof(double));
    cudaMallocHost((void**)&f, (N + 2) * (N + 2) * sizeof(double));
    long Nb = ((N + 2) * (N + 2) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    double *u_d, *f_d;
    cudaMalloc((void**)&u_d, (N + 2) * (N + 2) * sizeof(double));
    cudaMalloc((void**)&f_d, (N + 2) * (N + 2) * sizeof(double));
    initialize<<<Nb,BLOCK_SIZE>>>(u_d, f_d, (N + 2) * (N + 2));
    cudaMemcpyAsync(u, u_d, (N + 2) * (N + 2) * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(f, f_d, (N + 2) * (N + 2) * sizeof(double), cudaMemcpyDeviceToHost);
    double tt = omp_get_wtime();
    jacobi(u, f);
    printf("Time: %f\n", omp_get_wtime() - tt);
    cudaFreeHost(u);
    cudaFreeHost(f);
    cudaFree(u_d);
    cudaFree(f_d);
    return 0;
}