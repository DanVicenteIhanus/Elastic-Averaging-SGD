#!/bin/bash

# Arrays for the parameter grids
taus=(2 4 8)
workers=(3 5 9)

# Loop over each combination of parameters and run both scripts
for tau in "${taus[@]}"; do
    for N in "${workers[@]}"; do
        echo "Running easgd with tau=$tau, alpha=0.25, N=$N"
        mpiexec -n $N ./easgd $tau 0.25
    done
done