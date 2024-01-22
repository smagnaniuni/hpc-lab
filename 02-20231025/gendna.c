/****************************************************************************
 *
 * gendna -- Generate input for omp-c-ray
 *
 * Copyright (C) 2022 Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
% HPC -- Generate input for omp-c-ray
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Last modified 2022-03-20

Inspired by <https://twitter.com/pickover/status/1505350972641525765>

Compile with:

        gcc -std=c99 -Wall -Wpedantic gendna.c -o gendna

Run with:

        ./gendna | ./omp-c-ray -o dna.pbm

Better looking, but much slower:

        ./gendna | ./omp-c-ray -s 1024x768 -r 10 -o dna.pbm

***/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

enum {RED, BLUE};

/**
 * A sphere of radius sz and center (cx, cy, cz)
 */
void sphere( float sz, float cx, float cy, float cz, int color )
{
    printf("s  %f %f %f  %f  %s  50 0.2\n",
           cx, cy, cz, sz, color == BLUE ? "0 0 0.2" : "0.2 0 0");
}

int main( int argc, char *argv[] )
{
    int n = 30;

    if (argc > 2) {
        fprintf(stderr, "Usage %s [n_spheres]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (argc > 1) {
        n = atoi(argv[1]);
    }

    printf("# Input for omp-c-ray\n");
    printf("# Generated by %s %d\n", argv[0], n);
    const double r = 2.0;
    const double deltay = 0.25;
    const double deltaa = 0.4;
    double alpha = 0.0;
    double y = 0;
    for (int i=0; i<n; i++) {
        const double x = r * sin(alpha);
        const double z = r * cos(alpha);
        sphere(0.5, x, y, z, RED);
        sphere(0.5, -x, y, -z, BLUE);
        y += deltay;
        alpha += deltaa;
    }
    printf("s 10010 0 0   10000   0.3 0.5 0.3     8.0   0.1\n");  // left wall
    printf("s 0 0 10010   10000   0.5 0.3 0.3    40.0   0.1\n");  // right wall
    printf("s 0 -10003 0  10000   0.2 0.35 0.5   80.0   0.05\n"); // floor
    printf("l -50 50 -20\n"); // lights
    printf("l -20 100 -50\n");
    printf("c  -10 5 -15   90  0 2 0\n"); // camera

    return EXIT_SUCCESS;
}