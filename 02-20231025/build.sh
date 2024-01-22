#!/bin/bash

gcc -std=c99 -Wall -Wpedantic -fopenmp omp-schedule.c -o omp-schedule
gcc -std=c99 -Wall -Wpedantic -fopenmp omp-c-ray.c -o omp-c-ray -lm
gcc -std=c99 -Wall -Wpedantic -fopenmp omp-cat-map.c -o omp-cat-map
gcc -std=c99 -Wall -Wpedantic -fopenmp omp-merge-sort.c -o omp-merge-sort
