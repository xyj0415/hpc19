#include <mpi.h>
#include <stdio.h>

double time_ring(int size, int Nrepeat, int Nsize, MPI_Comm comm)
{
    int rank;
    MPI_Comm_rank(comm, &rank);
    
    int msg_int = 0;
    char* msg = (char*) malloc(Nsize);
    if (Nsize > 1 && rank == 0)
    {
        for (long i = 0; i < Nsize; i++)
            msg[i] = 42;
    }
    
    MPI_Barrier(comm);
    double tt = MPI_Wtime();
    for (long repeat = 0; repeat < Nrepeat; repeat++)
    {
        if (rank != 0)
        {
            if (Nsize == 1)
                MPI_Recv(&msg_int, 1, MPI_INT, rank - 1, repeat, comm, MPI_STATUS_IGNORE);
            else
                MPI_Recv(msg, Nsize, MPI_CHAR, rank - 1, repeat, comm, MPI_STATUS_IGNORE);
        }
        if (Nsize == 1)
        {
            msg_int += rank;
            MPI_Send(&msg_int, 1, MPI_INT, (rank + 1) % size, repeat, comm);
        }
        else
        {
            MPI_Send(msg, Nsize, MPI_CHAR, (rank + 1) % size, repeat, comm);
        }
        
        if (rank == 0)
        {
            if (Nsize == 1)
                MPI_Recv(&msg_int, 1, MPI_INT, size - 1, repeat, comm, MPI_STATUS_IGNORE);
            else
                MPI_Recv(msg, Nsize, MPI_CHAR, size - 1, repeat, comm, MPI_STATUS_IGNORE);
        }
    }
    tt = MPI_Wtime() - tt;
    if (Nsize == 1 && rank == 0)
        printf("value: %d = expected value %d * %d?\n", msg_int, size * (size - 1) / 2, Nrepeat);
    free(msg);
    return tt;
}

int main(int argc, char** argv)
{
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long Nrepeat = 10000;
    double tt = time_ring(size, Nrepeat, 1, MPI_COMM_WORLD);
    if (rank == 0)
        printf("latency: %e ms\n", tt / Nrepeat * 1000);

    long Nsize = 2 *1024 * 1024;
    tt = time_ring(size, Nrepeat, Nsize, MPI_COMM_WORLD);
    if (rank == 0)
        printf("bandwidth: %e GB/s\n", Nrepeat * Nsize * size / tt / 1e9);
    
    MPI_Finalize();
    return 0;
}