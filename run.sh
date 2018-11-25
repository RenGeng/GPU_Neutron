#!/bin/bash

nvcc -O3 neutron2.cu -o neutron

./neutron 1.0 100000000 0.5 0.5

rm neutron