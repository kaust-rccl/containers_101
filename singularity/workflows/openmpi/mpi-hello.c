#include <mpi.h>
#include <stdio.h>
// Compilation on Shaheen 2:
// cc -o hello mpi-hello.c
//
// Run the example on compute nodes of Shaheen 2:
// srun --nodes 4 --ntasks-per-node 2 ./hello
int main(int argc, char** argv) {
    MPI_Init(NULL, NULL);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    printf("Hello world from processor %s, rank %d out of %d processors\n",
           processor_name, world_rank, world_size);

    MPI_Finalize();
}

