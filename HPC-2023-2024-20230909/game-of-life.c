/****************************************************************************
 *
 * game-of-life.c - Serial implementaiton of the Game of Life
 *
 * Copyright (C) 2017--2021 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
 * Last updated on 2021-10-14
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
 * --------------------------------------------------------------------------
 *
 * Compile with:
 * gcc -std=c99 -Wall -Wpedantic game-of-life.c -o game-of-life
 *
 * Run with:
 * ./game-of-life 100
 *
 * To display the images
 * animate -delay 50 gol*.pbm
 *
 * To create a movie from the images:
 * ffmpeg -framerate 10 -i "gol%04d.pbm" -r 30 -vcodec mpeg4 gol.avi
 *
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>

/* grid size (excluding ghost cells) */
#define SIZE 256

int cur = 0; /* index of current grid (must be 0 or 1) */
unsigned char grid[2][SIZE+2][SIZE+2];

/* some useful constants; starting and ending rows/columns of the domain */
const int LEFT   = 1;
const int RIGHT  = SIZE;
const int TOP    = 1;
const int BOTTOM = SIZE;

/*
  HALO_LEFT   HALO_RIGHT
   | LEFT       RIGHT |
   | |              | |
   v v              v v
  +-+----------------+-+
  |\|\\\\\\\\\\\\\\\\|\| <- HALO_TOP
  +-+----------------+-+
  |\|                |\| <- TOP
  |\|                |\|
  |\|                |\|
  |\|                |\|
  |\|                |\|
  |\|                |\| <- BOTTOM
  +-+----------------+-+
  |\|\\\\\\\\\\\\\\\\|\| <- HALO_BOTTOM
  +-+----------------+-+

 */

/* copy the sides of current grid to the ghost cells. This function
   uses the global variables cur and grid. grid[cur] is modified.*/
#if 1
void copy_sides( void )
{
    int i, j;
    const int HALO_TOP    = TOP-1;
    const int HALO_BOTTOM = BOTTOM+1;
    const int HALO_LEFT   = LEFT-1;
    const int HALO_RIGHT  = RIGHT+1;

    /* copy top and bottom (one should better use memcpy() ) */
    for (j=LEFT; j<=RIGHT; j++) {
        grid[cur][HALO_TOP   ][j] = grid[cur][BOTTOM][j];
        grid[cur][HALO_BOTTOM][j] = grid[cur][TOP   ][j];
    }
    /* copy left and right */
    for (i=TOP; i<=BOTTOM; i++) {
        grid[cur][i][HALO_LEFT ] = grid[cur][i][RIGHT];
        grid[cur][i][HALO_RIGHT] = grid[cur][i][LEFT ];
    }
    /* copy corners */
    grid[cur][HALO_TOP   ][HALO_LEFT ] = grid[cur][BOTTOM][RIGHT];
    grid[cur][HALO_TOP   ][HALO_RIGHT] = grid[cur][BOTTOM][LEFT ];
    grid[cur][HALO_BOTTOM][HALO_LEFT ] = grid[cur][TOP   ][RIGHT];
    grid[cur][HALO_BOTTOM][HALO_RIGHT] = grid[cur][TOP   ][LEFT ];
}
#else
/* Another way to fill the ghost cells: change the ranges of the "for"
   cycles to copy entire rows and columns (including ghost cells). At
   the end, corners get filled with the correct values (draw an
   example to convince yourself). The interesting thing is that you
   can swap the two "for" cycles (i.e., first handle columns, then
   handle rows) and the final result is still correct. */
void copy_sides( )
{
    int i, j;
    const int HALO_TOP    = TOP-1;
    const int HALO_BOTTOM = BOTTOM+1;
    const int HALO_LEFT   = LEFT-1;
    const int HALO_RIGHT  = RIGHT+1;

    /* Copy top and bottom (one can also use memcpy() ). We copy a
       whole row (including ghost cells). */
    for (j=0; j<=HALO_RIGHT; j++) {
        grid[cur][HALO_TOP   ][j] = grid[cur][BOTTOM][j];
        grid[cur][HALO_BOTTOM][j] = grid[cur][TOP   ][j];
    }
    /* Copy left and right. We copy a whole column (including ghost
       cells). */
    for (i=0; i<=HALO_BOTTOM; i++) {
        grid[cur][i][HALO_LEFT ] = grid[cur][i][RIGHT];
        grid[cur][i][HALO_RIGHT] = grid[cur][i][LEFT ];
    }
    /* There is no need to fill the corners */
}
#endif

/* Compute the next grid given the current configuration; this
   function uses the global variables grid and cur; updates are
   written to the (1-cur) grid. */
void step( void )
{
    int i, j, next = 1 - cur;
    for (i=TOP; i<=BOTTOM; i++) {
        for (j=LEFT; j<=RIGHT; j++) {
            /* count live neighbors of cell (i,j) */
            int nbors =
                grid[cur][i-1][j-1] + grid[cur][i-1][j] + grid[cur][i-1][j+1] +
                grid[cur][i  ][j-1] +                     grid[cur][i  ][j+1] +
                grid[cur][i+1][j-1] + grid[cur][i+1][j] + grid[cur][i+1][j+1];
 	    /* apply rules of the game of life to cell (i, j) */
            if ( grid[cur][i][j] && (nbors < 2 || nbors > 3)) {
                grid[next][i][j] = 0;
            } else {
                if ( !grid[cur][i][j] && (nbors == 3)) {
                    grid[next][i][j] = 1;
                } else {
                    grid[next][i][j] = grid[cur][i][j];
                }
            }
        }
    }
}

/* Initialize the current grid grid[cur] with alive cells with density
   p. This function uses the global variables cur and grid. grid[cur]
   is modified. */
void init( float p )
{
    int i, j;
    for (i=TOP; i<=BOTTOM; i++) {
        for (j=LEFT; j<=RIGHT; j++) {
            grid[cur][i][j] = (((float)rand())/RAND_MAX < p);
        }
    }
}

/* Write grid[cur] to file fname in pbm (portable bitmap) format. This
   function uses the global variables cur and grid (neither is
   modified). */
void write_pbm( const char* fname )
{
    int i, j;
    FILE *f = fopen(fname, "w");
    if (!f) {
        printf("Cannot open %s for writing\n", fname);
        abort();
    }
    fprintf(f, "P1\n");
    fprintf(f, "# produced by game-of-life.c\n");
    fprintf(f, "%d %d\n", SIZE, SIZE);
    for (i=TOP; i<=BOTTOM; i++) {
        for (j=LEFT; j<=RIGHT; j++) {
            fprintf(f, "%d ", grid[cur][i][j]);
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

#define BUFSIZE 128

int main( int argc, char* argv[] )
{
    int s, nsteps = 1000;
    char fname[BUFSIZE];

    srand(1234); /* init RNG */

    if ( argc > 2 ) {
        fprintf(stderr, "Usage: %s [nsteps]\n", argv[0]);
        return EXIT_FAILURE;
    }
    if ( argc == 2 ) {
        nsteps = atoi(argv[1]);
    }
    cur = 0;
    init(0.3);
    for (s=0; s<nsteps; s++) {
        snprintf(fname, BUFSIZE, "gol%04d.pbm", s);
        write_pbm(fname);
        copy_sides();
        step();
        cur = 1 - cur;
    }
    return EXIT_SUCCESS;
}
