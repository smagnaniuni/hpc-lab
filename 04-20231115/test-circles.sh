#!/bin/bash

mpirun -n 4 ./mpi-circles-serial 10000 circles-1000.in
mpirun -n 1 ./mpi-circles 10000 circles-1000.in
mpirun -n 4 ./mpi-circles 10000 circles-1000.in
mpirun -n 4 ./mpi-circles 10001 circles-1000.in
mpirun -n 5 ./mpi-circles 10000 circles-1000.in
