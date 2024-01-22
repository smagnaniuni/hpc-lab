#!/bin/bash

./omp-c-ray -s 800x600 < sphfract.small.in > img.ppm
convert img.ppm img.jpeg
