#!/bin/bash

gcc -std=c99 -Wall -Wpedantic -O2 -march=native simd-map-levels.c -o simd-map-levels

# gcc -O2 -march=native -ftree-vectorize -fopt-info-vec-all simd-dot.c -o simd-dot -lm 2>&1 | grep "loop vectorized"
# gcc -S -c -march=native -O2 -ftree-vectorize simd-dot.c -o simd-dot.s
# gcc -O2 -march=native -ftree-vectorize -fopt-info-vec-all -funsafe-math-optimizations simd-dot.c -o simd-dot -lm 2>&1 | grep "loop vectorized"
gcc -std=c99 -Wall -Wpedantic -O2 -march=native -g -ggdb simd-dot.c -o simd-dot -lm

gcc -march=native -O2 -std=c99 -Wall -Wpedantic -D_XOPEN_SOURCE=600 simd-matmul.c -o simd-matmul
