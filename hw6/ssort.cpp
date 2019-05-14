// Parallel sample sort
#include <stdio.h>
#include <unistd.h>
#include <mpi.h>
#include <stdlib.h>
#include <algorithm>
#include <fstream>
#include <iostream>

using namespace std;

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int rank, p;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  // Number of random numbers per processor (this should be increased
  // for actual tests or could be passed in through the command line
  int N = 100;

  int* vec = (int*)malloc(N*sizeof(int));
  // seed random number generator differently on every core
  srand((unsigned int) (rank + 393919));

  // fill vector with random integers
  for (int i = 0; i < N; ++i) {
    vec[i] = rand();
  }
  printf("rank: %d, first entry: %d\n", rank, vec[0]);

  // sort locally
  std::sort(vec, vec+N);

  // sample p-1 entries from vector as the local splitters, i.e.,
  // every N/P-th entry of the sorted vector
  int* local_splitter = (int*) malloc((p - 1) * sizeof(int));
  for (int i = 0; i < p - 1; i++)
    local_splitter[i] = vec[(i + 1) * N / p];

  // every process communicates the selected entries to the root
  // process; use for instance an MPI_Gather
  int* gathered_splitter = nullptr;
  if (rank == 0) {
    gathered_splitter = (int*) malloc(p * (p - 1) * sizeof(int));
  }
  MPI_Gather(local_splitter, p - 1, MPI_INT, gathered_splitter, p - 1, MPI_INT, 0, MPI_COMM_WORLD);

  // root process does a sort and picks (p-1) splitters (from the
  // p(p-1) received elements)
  int* global_splitter = (int*) malloc((p - 1) * sizeof(int));
  if (rank == 0) { 
    sort(gathered_splitter, gathered_splitter + p * (p - 1));
    for (int i = 0; i < p - 1; i++)
      global_splitter[i] = gathered_splitter[(i + 1) * (p - 1)];
  }
  

  // root process broadcasts splitters to all other processes
  MPI_Bcast(global_splitter, p - 1, MPI_INT, 0, MPI_COMM_WORLD);

  // every process uses the obtained splitters to decide which
  // integers need to be sent to which other process (local bins).
  // Note that the vector is already locally sorted and so are the
  // splitters; therefore, we can use std::lower_bound function to
  // determine the bins efficiently.
  //
  // Hint: the MPI_Alltoallv exchange in the next step requires
  // send-counts and send-displacements to each process. Determining the
  // bins for an already sorted array just means to determine these
  // counts and displacements. For a splitter s[i], the corresponding
  // send-displacement for the message to process (i+1) is then given by,
  // sdispls[i+1] = std::lower_bound(vec, vec+N, s[i]) - vec;
  int* sdispls = (int*) malloc(p * sizeof(int));
  sdispls[0] = 0;
  for (int i = 0; i < p - 1; i++) {
    sdispls[i + 1] = std::lower_bound(vec, vec + N, global_splitter[i]) - vec;
  }
  
  // send and receive: first use an MPI_Alltoall to share with every
  // process how many integers it should expect, and then use
  // MPI_Alltoallv to exchange the data
  int* sendcount = (int*) malloc(p * sizeof(int));
  for (int i = 0; i < p - 1; i++) {
    sendcount[i] = sdispls[i + 1] - sdispls[i];
  }
  sendcount[p - 1] = N - sdispls[p - 1];

  int* recvcount = (int*) malloc(p * sizeof(int));
  MPI_Alltoall(sendcount, 1, MPI_INT, recvcount, 1, MPI_INT, MPI_COMM_WORLD);

  int* rdispls = (int*) malloc(p * sizeof(int));
  rdispls[0] = 0;
  for (int i = 0; i < p - 1; i++) {
    rdispls[i + 1] = rdispls[i] + recvcount[i];
  }
  int recvbuf_size = rdispls[p - 1] + recvcount[p - 1];

  int* recvbuf = (int*) malloc(recvbuf_size * sizeof(int));
  MPI_Alltoallv(vec, sendcount, sdispls, MPI_INT, recvbuf, recvcount, rdispls, MPI_INT, MPI_COMM_WORLD);

  // do a local sort of the received data
  sort(recvbuf, recvbuf + recvbuf_size);
  
  // every process writes its result to a file
  ofstream outfile;
  outfile.open("output" + to_string(rank));
  for (int i = 0; i < recvbuf_size; i++) {
    outfile << recvbuf[i] << ' ';
  }
  outfile.close();

  free(local_splitter);
  free(gathered_splitter);
  free(global_splitter);
  free(sdispls);
  free(sendcount);
  free(recvcount);
  free(rdispls);
  free(recvbuf);
  free(vec);
  MPI_Finalize();
  return 0;
}
