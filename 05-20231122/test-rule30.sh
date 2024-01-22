#!/bin/bash

mpirun -n 4 ./mpi-rule30-serial
convert ./rule30-serial.pbm ./rule30-serial.png

mpirun -n 4 ./mpi-rule30
convert ./rule30.pbm ./rule30.png
