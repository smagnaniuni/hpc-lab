/****************************************************************************
 *
 * cuda-reverse.cu - Array reversal with CUDA
 *
 * Copyright (C) 2017--2022 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
% HPC - Array reversal with CUDA
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Last updated: 2022-11-17

Write a program that reverses an array `v[]` of length $n$, i.e.,
exchanges `v[0]` and `v[n-1]`, `v[1]` and `v[n-2]` and so
on. You should write two versions of the program:

1. the first version reverses an input array `in[]` into a different
   output array `out[]`, so that the input is not modified. You can
   assume that `in[]` and `out[]` are mapped to different,
   non-overlapping memory blocks.

2. The second version reverses an array `in[]` "in place" using $O(1)$
   additional storage.

The file [cuda-reverse.cu](cuda-reverse.cu) provides a CPU-based
implementation of `reverse()` and `inplace_reverse()`.  Modify the
functions to use of the GPU.

**Hint:** `reverse()` can be easily transformed into a kernel executed
by $n$ CUDA threads (one for each array element). Each thread copies
one element from `in[]` to `out[]`. Use one-dimensional _thread
blocks_, since that makes easy to map threads to array elements.
`inplace_reverse()` can be transformed into a kernel as well, but in
this case only $\lfloor n/2 \rfloor$ CUDA threads are required (note
the rounding): each thread swaps an element from the first half of
`in[]` with the appropriate element from the second half. Make sure
that the program works also when the input length $n$ is odd.

To compile:

        nvcc cuda-reverse.cu -o cuda-reverse

To execute:

        ./cuda-reverse [n]

Example:

        ./cuda-reverse

## Files

- [cuda-reverse.cu](cuda-reverse.cu)
- [hpc.h](hpc.h)

***/
#include "hpc.h"
#include <stdio.h>
#include <math.h>
#include <assert.h>

#define BLKDIM 1024

__global__
void reverse_kernel(int *in, int *out, int n)
{
    const uint tid = threadIdx.x;
    for (int i = tid; i < n; i += blockDim.x) {
        const int opposite = n - 1 - i;
        out[opposite] = in[i];
    }
}

/* Reverses `in[]` into `out[]`; assume that `in[]` and `out[]` do not
   overlap.

   [TODO] Modify this function so that it:
   - allocates memory on the device to hold a copy of `in` and `out`;
   - copies `in` to the device
   - launches a kernel (to be defined)
   - copies data back from device to host
   - deallocates memory on the device
 */
void reverse( int *in, int *out, int n )
{
    // int i;
    // for (i=0; i<n; i++) {
    //     const int opp = n - 1 - i;
    //     out[opp] = in[i];
    // }
    int* d_in, *d_out;
    const size_t SIZE_INOUT = n * sizeof(int);
    cudaSafeCall(cudaMalloc((void**)&d_in, SIZE_INOUT));
    cudaSafeCall(cudaMalloc((void**)&d_out, SIZE_INOUT));

    cudaSafeCall(cudaMemcpy(d_in, in, SIZE_INOUT, cudaMemcpyHostToDevice));

    // reverse_kernel<<<1, n>>>(d_in, d_out, n);    // n => invalid configuration argument
    reverse_kernel<<<1, BLKDIM>>>(d_in, d_out, n);
    cudaCheckError();

    cudaSafeCall(cudaMemcpy(out, d_out, SIZE_INOUT, cudaMemcpyDeviceToHost));

    cudaSafeCall(cudaFree(d_in));
    cudaSafeCall(cudaFree(d_out));
}

__global__
void inplace_reverse_kernel(int* in, int n)
{
    const uint tid = threadIdx.x;
    int nhalf = n / 2;
    for(int i = tid; i < nhalf; i += blockDim.x) {
        const int j = n - 1 - i;
        const int tmp = in[j];
        in[j] = in[i];
        in[i] = tmp;
    }
}

/* In-place reversal of in[] into itself.

   [TODO] Modify this function so that it:
   - allocates memory on the device to hold a copy of `in`;
   - copies `in` to the device
   - launches a kernel (to be defined)
   - copies data back from device to host
   - deallocates memory on the device
*/
void inplace_reverse( int *in, int n )
{
    // int i = 0, j = n-1;
    // while (i < j) {
    //     const int tmp = in[j];
    //     in[j] = in[i];
    //     in[i] = tmp;
    //     j--;
    //     i++;
    // }
    int* d_in;
    const size_t SIZE_IN = n * sizeof(int);
    cudaSafeCall(cudaMalloc((void**)&d_in, SIZE_IN));
    cudaSafeCall(cudaMemcpy(d_in, in, SIZE_IN, cudaMemcpyHostToDevice));

    inplace_reverse_kernel<<<1, BLKDIM>>>(d_in, n);
    cudaCheckError();

    cudaSafeCall(cudaMemcpy(in, d_in, SIZE_IN, cudaMemcpyDeviceToHost));
    cudaSafeCall(cudaFree(d_in));
}

void fill( int *x, int n )
{
    int i;
    for (i=0; i<n; i++) {
        x[i] = i;
    }
}

int check( const int *x, int n )
{
    int i;
    for (i=0; i<n; i++) {
        if (x[i] != n - 1 - i) {
            fprintf(stderr, "Test FAILED: x[%d]=%d, expected %d\n", i, x[i], n-1-i);
            return 0;
        }
    }
    printf("Test OK\n");
    return 1;
}

int main( int argc, char* argv[] )
{
    int *in, *out;
    int n = 1024*1024;
    const int MAX_N = 512*1024*1024;

    if ( argc > 2 ) {
        fprintf(stderr, "Usage: %s [n]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if ( argc > 1 ) {
        n = atoi(argv[1]);
    }

    if ( n > MAX_N ) {
        fprintf(stderr, "FATAL: input too large (maximum allowed length is %d)\n", MAX_N);
        return EXIT_FAILURE;
    }

    const size_t SIZE = n * sizeof(*in);

    /* Allocate in[] and out[] */
    in = (int*)malloc(SIZE);
    assert(in != NULL);
    out = (int*)malloc(SIZE);
    assert(out != NULL);
    fill(in, n);

    printf("Reverse %d elements... ", n);
    reverse(in, out, n);
    check(out, n);

    printf("In-place reverse %d elements... ", n);
    inplace_reverse(in, n);
    check(in, n);

    /* Cleanup */
    free(in);
    free(out);

    return EXIT_SUCCESS;
}
