#!/bin/bash

mpicc -std=c99 -Wall -Wpedantic mpi-dot.c -o mpi-dot -lm
mpicc -std=c99 -Wall -Wpedantic mpi-mandelbrot-serial.c -o mpi-mandelbrot-serial
mpicc -std=c99 -Wall -Wpedantic mpi-mandelbrot.c -o mpi-mandelbrot
mpicc -std=c99 -Wall -Wpedantic mpi-circles-serial.c -o mpi-circles-serial
mpicc -std=c99 -Wall -Wpedantic mpi-circles.c -o mpi-circles
