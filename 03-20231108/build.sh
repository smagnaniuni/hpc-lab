#!/bin/bash

mpicc -std=c99 -Wall -Wpedantic mpi-my-bcast.c -o mpi-my-bcast
mpicc -std=c99 -Wall -Wpedantic mpi-ring.c -o mpi-ring
mpicc -std=c99 -Wall -Wpedantic mpi-pi.c -o mpi-pi -lm
