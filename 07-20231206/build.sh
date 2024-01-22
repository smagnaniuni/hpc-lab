#!/bin/bash

gcc -std=c99 -Wall -Wpedantic -fopenmp cuda-anneal-omp.c -o cuda-anneal-omp
nvcc cuda-anneal-serial.cu -o cuda-anneal-serial
nvcc cuda-anneal.cu -o cuda-anneal
nvcc cuda-matsum-serial.cu -o cuda-matsum-serial -lm
nvcc cuda-matsum.cu -o cuda-matsum -lm
nvcc cuda-rule30-serial.cu -o cuda-rule30-serial
nvcc cuda-rule30.cu -o cuda-rule30
