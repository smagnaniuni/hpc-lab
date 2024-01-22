#!/bin/sh

mpicc -std=c99 -Wall -Wpedantic mpi-rule30-serial.c -o mpi-rule30-serial
mpicc -std=c99 -Wall -Wpedantic mpi-rule30.c -o mpi-rule30
mpicc -std=c99 -Wall -Wpedantic mpi-send-col.c -o mpi-send-col
mpicc -std=c99 -Wall -Wpedantic mpi-bbox-serial.c -o mpi-bbox-serial -lm
mpicc -std=c99 -Wall -Wpedantic mpi-bbox.c -o mpi-bbox -lm
