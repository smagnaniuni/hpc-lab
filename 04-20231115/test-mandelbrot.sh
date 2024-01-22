#!/bin/bash

mpirun -n 1 ./mpi-mandelbrot-serial 800
mpirun -n 4 ./mpi-mandelbrot 800
cmp ./mpi-mandelbrot-serial.ppm ./mpi-mandelbrot.ppm
# cmp ./mpi-mandelbrot-serial.png ./mpi-mandelbrot.png # DON'T COMPARE THE PNGs!!!
convert ./mpi-mandelbrot-serial.ppm ./mpi-mandelbrot-serial.png
convert ./mpi-mandelbrot.ppm ./mpi-mandelbrot.png
