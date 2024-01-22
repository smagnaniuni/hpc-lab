#!/bin/bash

nvcc cuda-dot.cu -o cuda-dot
# nvcc -DNO_CUDA_CHECK_ERROR cuda-dot.cu -o cuda-dot
nvcc cuda-reverse.cu -o cuda-reverse
nvcc cuda-odd-even.cu -o cuda-odd-even
