#!/bin/bash

echo "CUDA"
./cuda-anneal
echo "SERIAL"
./cuda-anneal-serial
cmp cuda-anneal-000064.pbm cuda-anneal-serial-000064.pbm
