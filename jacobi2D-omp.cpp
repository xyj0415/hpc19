#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "utils.h"
#ifdef _OPENMP 
    #include "omp.h"
#else
    #define omp_get_thread_num() 0
#endif

using namespace std;

const int N = 2000;
const double h = 1.0 / (N + 1);
const double sqrh = h * h;
const int MAX_ITER = 2000;
const int NTHREADS = 4;

double calc_residue(double *u, double *f)
{
    double ans = 0, a_times_u = 0;
    int i, j;
    #pragma omp parallel for num_threads(NTHREADS) shared(u, f) reduction(+ : ans)
    for (i = 1; i <= N; i++)
        for (j = 1; j <= N; j++)
        {
            int pos = i * (N + 2) + j;
            a_times_u = -(-4 * u[pos] + u[pos - 1] + u[pos + 1] + u[pos - N - 2] + u[pos + N + 2]) / sqrh;
            ans += (a_times_u - f[pos]) * (a_times_u - f[pos]);
        }
    return sqrt(ans);
}

void jacobi(double* u, double* f)
{
    double init_residue = calc_residue(u, f), residue = init_residue;
    for (int iter = 0; iter < MAX_ITER && residue * 1e6 > init_residue; iter++)
    {
        // if (iter % 200 == 0)
        //     printf("Iter:%d Residue:%f\n", iter, residue);
        double *u2 = (double *) malloc((N + 2) * (N + 2) * sizeof(double));
        int i, j;
        #pragma omp parallel num_threads(NTHREADS) shared(u, u2, f)
        {
            #pragma omp for
            for (i = 1; i <= N; i++)
                for (j = 1; j <= N; j++)
                {
                    int pos = i * (N + 2) + j;
                    u2[pos] = (sqrh * f[pos] + u[pos - 1] + u[pos + 1] + u[pos - N - 2] + u[pos + N + 2]) / 4;
                }
            #pragma omp for
            for (int i = 0; i < (N + 2) * (N + 2); i++)
                u[i] = u2[i];
        }
        free(u2);
        // No need to calculate residue every iteration because it has the same time complexity as itreation.
        // Running time is reduced by ~40% by adding the if clause below.
        if (iter % 10 == 0)
            residue = calc_residue(u, f);
    }
    // residue = calc_residue(u, f);
    // printf("Iter:%d Residue:%f\n", MAX_ITER, residue);
}

int main(int argc, char const *argv[])
{
    double *u = (double *) aligned_malloc((N + 2) * (N + 2) * sizeof(double));
    double *f = (double *) aligned_malloc((N + 2) * (N + 2) * sizeof(double));
    #pragma omp parallel for num_threads(NTHREADS) shared(u, f)
    for (int i = 0; i < (N + 2) * (N + 2); i++)
        u[i] = 0, f[i] = 1;
    Timer t;
    t.tic();
    jacobi(u, f);
    double time = t.toc();
    printf("Time: %f\n", time);
}