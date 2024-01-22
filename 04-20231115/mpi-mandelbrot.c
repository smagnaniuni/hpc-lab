/****************************************************************************
 *
 * mpi-mandelbrot.c - Mandelbrot set
 *
 * Copyright (C) 2017--2023 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
% HPC - Mandelbrot set
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Last updated: 2023-11-13

![](mandelbrot-set.png)

The file [mpi-mandelbrot.c](mpi-mandelbrot.c) contains the skeleton of
a MPI program that computes the Mandelbrot set; it is not a parallel
version, since the master process does everything.

The program accepts the image height as an optional command-line
parameter; the width is computed automatically to include the whole
set. The resulting image is written to the file `mandebrot.ppm` in PPM
(_Portable Pixmap_) format. To convert the image, e.g., to PNG you can
use the following command on the Linux server:

        convert mandelbrot.ppm mandelbrot.png

The goal of this exercise is to write a parallel version where all MPI
processes contribute to the computation. To do this, you can partition
the image into $P$ vertical blocks where $P$ is the number of MPI
processes, and let each process draws a portion of the image (see
Figure 1).

![Figure 1: Domain decomposition for the computation of the Mandelbrot
 set with 4 MPI processes](mpi-mandelbrot.png)

Specifically, each process computes a portion of the image of size
$\mathit{xsize} \times (\mathit{ysize} / P)$. This is an
_embarrassingly parallel_ computation, since there is no need to
communicate. At the end, the processes send their local result to the
master using the `MPI_Gather()` function, so that the master can
assemble the final result. Since we are dealing with images where
three bytes are used to encode the color of each pixel, the
`MPI_Gather()` operation will transfer blocks of $(3 \times
\mathit{xsize} \times \mathit{ysize} / P)$ elements of type
`MPI_BYTE`.

You can initially assume that _ysize_ is an integer multiple of $P$,
and then relax this assumption, e.g., by letting process 0 take care
of the last `(ysize % P)` rows. Alternatively, you can use
`MPI_Gatherv()` to allow blocks of different sizes to be collected by
the master.

You are suggested to keep the serial program as a reference.  To check
the correctness of the parallel implementation, compare the output
images produced by both versions with the command:

        cmp file1 file2

They must be identical, i.e., the `cmp` program should not print any
message; if not, something is wrong.

To compile:

        mpicc -std=c99 -Wall -Wpedantic mpi-mandelbrot.c -o mpi-mandelbrot

To execute:

        mpirun -n NPROC ./mpi-mandelbrot [ysize]

Example:

        mpirun -n 4 ./mpi-mandelbrot 800

## Files

- [mpi-mandelbrot.c](mpi-mandelbrot.c)

***/
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <mpi.h>

const int MAXIT = 100;

/* The __attribute__(( ... )) definition is gcc-specific, and tells
   the compiler that the fields of this structure should not be padded
   or aligned in any way. */
typedef struct __attribute__((__packed__)) {
    uint8_t r;  /* red   */
    uint8_t g;  /* green */
    uint8_t b;  /* blue  */
} pixel_t;

/* color gradient from https://stackoverflow.com/questions/16500656/which-color-gradient-is-used-to-color-mandelbrot-in-wikipedia */
const pixel_t colors[] = {
    {66, 30, 15}, /* r, g, b */
    {25, 7, 26},
    {9, 1, 47},
    {4, 4, 73},
    {0, 7, 100},
    {12, 44, 138},
    {24, 82, 177},
    {57, 125, 209},
    {134, 181, 229},
    {211, 236, 248},
    {241, 233, 191},
    {248, 201, 95},
    {255, 170, 0},
    {204, 128, 0},
    {153, 87, 0},
    {106, 52, 3} };
const int NCOLORS = sizeof(colors)/sizeof(colors[0]);

/*
 * Iterate the recurrence:
 *
 * z_0 = 0;
 * z_{n+1} = z_n^2 + cx + i*cy;
 *
 * Returns the first `n` such that `z_n > bound`, or `MAXIT` if `z_n` is below
 * `bound` after `MAXIT` iterations.
 */
int iterate( float cx, float cy )
{
    float x = 0.0f, y = 0.0f, xnew, ynew;
    int it;
    for ( it = 0; (it < MAXIT) && (x*x + y*y <= 2.0*2.0); it++ ) {
        xnew = x*x - y*y + cx;
        ynew = 2.0*x*y + cy;
        x = xnew;
        y = ynew;
    }
    return it;
}

/* Draw the rows of the Mandelbrot set from `ystart` (inclusive) to
   `yend` (excluded) to the bitmap pointed to by `p`. Note that `p`
   must point to the beginning of the bitmap where the portion of
   image will be stored; in other words, this function writes to
   pixels p[0], p[1], ... `xsize` and `ysize` MUST be the sizes
   of the WHOLE image. */
void draw_lines( int ystart, int yend, pixel_t* p, int xsize, int ysize )
{
    int x, y;
    for ( y = ystart; y < yend; y++) {
        for ( x = 0; x < xsize; x++ ) {
            const float cx = -2.5 + 3.5 * (float)x / (xsize - 1);
            const float cy = 1 - 2.0 * (float)y / (ysize - 1);
            const int v = iterate(cx, cy);
            if (v < MAXIT) {
                p->r = colors[v % NCOLORS].r;
                p->g = colors[v % NCOLORS].g;
                p->b = colors[v % NCOLORS].b;
            } else {
                p->r = p->g = p->b = 0;
            }
            p++;
        }
    }
}

int main( int argc, char *argv[] )
{
    int my_rank, comm_sz;
    FILE *out = NULL;
    const char* fname="mpi-mandelbrot.ppm";
    pixel_t *bitmap = NULL;
    int xsize, ysize;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if ( argc > 1 ) {
        ysize = atoi(argv[1]);
    } else {
        ysize = 1024;
    }

    xsize = ysize * 1.4;

    int* sendcounts = (int*)malloc(comm_sz * sizeof(int)); assert(sendcounts != NULL);
    int* displs = (int*)malloc(comm_sz * sizeof(int)); assert(displs != NULL);
    int* y_starts = (int*)malloc(comm_sz * sizeof(int)); assert(y_starts != NULL);
    int* y_sizes = (int*)malloc(comm_sz * sizeof(int)); assert(y_sizes != NULL);
    for (size_t i = 0; i < comm_sz; i++) {
        const int local_ystart = (ysize * i) / comm_sz;
        const int local_yend = (ysize * (i + 1)) / comm_sz;
        const int local_ysize = local_yend - local_ystart;
        sendcounts[i] = 3 * xsize * local_ysize;
        displs[i] = 3 * xsize * local_ystart;
        y_starts[i] = local_ystart;
        y_sizes[i] = local_ysize;
    }
    const int local_ysize = y_sizes[my_rank];
    pixel_t* local_bitmap = (pixel_t*)malloc(xsize * local_ysize * sizeof(*local_bitmap));
    draw_lines(y_starts[my_rank], y_starts[my_rank] + local_ysize, local_bitmap, xsize, ysize);

    /* xsize and ysize are known to all processes */
    if ( 0 == my_rank ) {
        out = fopen(fname, "w");
        if ( !out ) {
            fprintf(stderr, "Error: cannot create %s\n", fname);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        /* Write the header of the output file */
        fprintf(out, "P6\n");
        fprintf(out, "%d %d\n", xsize, ysize);
        fprintf(out, "255\n");

        /* Allocate the complete bitmap */
        bitmap = (pixel_t *)malloc(xsize * ysize * sizeof(*bitmap));
        assert(bitmap != NULL);
    }
    /* [TODO] This is not a true parallel version, since the master
        does everything */
    // draw_lines(0, ysize, bitmap, xsize, ysize);

    MPI_Datatype datatype = MPI_BYTE;
    // MPI_Datatype datatype = MPI_UINT8_T;
    MPI_Gatherv(local_bitmap, sendcounts[my_rank], datatype,
        bitmap, sendcounts, displs,
        datatype, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        fwrite(bitmap, sizeof(*bitmap), xsize * ysize, out);
        fclose(out);
        free(bitmap);
    }

    free(sendcounts);
    free(displs);
    free(local_bitmap);
    free(y_starts);
    free(y_sizes);

    MPI_Finalize();

    return EXIT_SUCCESS;
}
