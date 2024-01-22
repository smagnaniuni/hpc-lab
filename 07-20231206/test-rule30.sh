#!/bin/bash

echo "CUDA"
./cuda-rule30
echo "SERIAL"
./cuda-rule30-serial
cmp cuda-rule30.pbm cuda-rule30-serial.pbm
