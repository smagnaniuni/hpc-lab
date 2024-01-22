/****************************************************************************
 *
 * mpi-dot.c - Dot product
 *
 * Copyright (C) 2016--2021 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 ****************************************************************************/

/***
% HPC - Dot product
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Last updated: 2022-11-01

The file [mpi-dot.c](mpi-dot.c) contains a MPI program that computes
the dot product between two arrays `a[]` and `b[]` of length $n$. The
dot product $s$ of two arrays `a[]` and `b[]` is defined as:

$$
s = \sum_{i = 0}^{n-1} a[i] \times b[i]
$$

In the provided program, the master performs the whole computation and
is therefore not parallel. The goal of this exercise is to write a
parallel version. Assume that, at the beginning of the program, `a[]`
and `b[]` are known only to the master. Therefore, they must be
distributed across the processes. Each process computes the scalar
product of the assigned portions of the arrays; the master then uses
`MPI_Reduce()` to sum the partial results and compute $s$.

You may initially assume that $n$ is an exact multiple of the number
of MPI processes $P$; then, relax this assumption and modify the
program so that it works with any array length $n$. The simpler
solution is to distribute the arrays using `MPI_Scatter()` and let the
master take care of any excess data. Another possibility is to use
`MPI_Scatterv()` to distribute the input unevenly across the
processes.

To compile:

        mpicc -std=c99 -Wall -Wpedantic mpi-dot.c -o mpi-dot -lm

To execute:

        mpirun -n P ./mpi-dot [n]

Example:

        mpirun -n 4 ./mpi-dot 1000

## Files

- [mpi-dot.c](mpi-dot.c)

***/
#include <stdio.h>
#include <stdlib.h>
#include <math.h> /* for fabs() */
#include <assert.h>
#include <mpi.h>

/*
 * Compute sum { x[i] * y[i] }, i=0, ... n-1
 */
double dot( const double* x, const double* y, int n )
{
    double s = 0.0;
    int i;
    for (i=0; i<n; i++) {
        s += x[i] * y[i];
    }
    return s;
}

int main( int argc, char* argv[] )
{
    const double TOL = 1e-5;
    double *x = NULL, *y = NULL, result = 0.0;
    int i, n = 1000;
    int my_rank, comm_sz;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if ( argc > 1 ) {
        n = atoi(argv[1]);
    }

    if ( 0 == my_rank ) {
        /* The master allocates the vectors */
        x = (double*)malloc( n * sizeof(*x) ); assert(x != NULL);
        y = (double*)malloc( n * sizeof(*y) ); assert(y != NULL);
        for ( i=0; i<n; i++ ) {
            x[i] = i + 1.0;
            y[i] = 1.0 / x[i];
        }
    }
    /* [TODO] This is not a true parallel version, since the master
       does everything */
    //if ( 0 == my_rank ) {
    //    result = dot(x, y, n);
    //}

    //int sendcounts[comm_sz];
    int* sendcounts = (int*)malloc(comm_sz * sizeof(int)); assert(sendcounts != NULL);
    int* displs = (int*)malloc(comm_sz * sizeof(int)); assert(displs != NULL);
    for(size_t i = 0; i < comm_sz; i++) {
        const int start = (n * i) / comm_sz;
        const int end = (n * (i + 1)) / comm_sz;
        displs[i] = start; //displs numero di elementi prima dell'inizio, con l'indicizzazione a zero e' anche uguale all'indice di inizio
        sendcounts[i] = end - start; //non piu uno, end e' l'elemento dopo
    }

    const int local_n = sendcounts[my_rank];
    double* local_x = (double*)malloc(local_n * sizeof(double)); assert(local_x != NULL);
    double* local_y = (double*)malloc(local_n * sizeof(double)); assert(local_y != NULL);
    /*
     * int MPI_Scatterv(const void *sendbuf, const int sendcounts[], const int displs[],
     *      MPI_Datatype sendtype, void *recvbuf, int recvcount,
     *      MPI_Datatype recvtype, int root, MPI_Comm comm)
     */
    MPI_Scatterv(x, sendcounts, displs,
            MPI_DOUBLE, local_x, local_n,
            MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(y, sendcounts, displs,
            MPI_DOUBLE, local_y, local_n,
            MPI_DOUBLE, 0, MPI_COMM_WORLD);

    const double local_result = dot(local_x, local_y, local_n);
    /*
     * int MPI_Reduce(const void *sendbuf, void *recvbuf, int count,
     *      MPI_Datatype datatype, MPI_Op op, int root,
     *      MPI_Comm comm)
     */
    MPI_Reduce(&local_result, &result, 1,
            MPI_DOUBLE, MPI_SUM, 0,
            MPI_COMM_WORLD);

    if (0 == my_rank) {
        printf("Dot product: %f\n", result);
        if ( fabs(result - n) < TOL ) {
            printf("Check OK\n");
        } else {
            printf("Check failed: got %f, expected %f\n", result, (double)n);
        }
    }

    free(x); /* if x == NULL, does nothing */
    free(y);
    free(local_x);
    free(local_y);
    free(sendcounts);
    free(displs);

    MPI_Finalize();

    return EXIT_SUCCESS;
}
