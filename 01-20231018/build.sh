#!/bin/bash

gcc -std=c99 -Wall -Wpedantic -fopenmp omp-brute-force.c -o omp-brute-force
gcc -std=c99 -fopenmp -Wall -Wpedantic omp-pi.c -o omp-pi -lm
gcc -std=c99 -Wall -Wpedantic -fopenmp omp-letters.c -o omp-letters
gcc -std=c99 -Wall -Wpedantic -fopenmp omp-letters-sol.c -o omp-letters-sol
gcc -std=c99 -Wall -Wpedantic -fopenmp omp-sieve.c -o omp-sieve
