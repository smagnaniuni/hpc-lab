#!/bin/bash

mpirun -n 4 ./mpi-bbox-serial bbox-1000.in
mpirun -n 4 ./mpi-bbox bbox-1000.in
