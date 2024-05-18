#!/bin/bash

# Arrays for the parameter grids
taus=(2 4 8)
betas=(2 4)
deltas=(0.7 0.9)
workers=(3 5 9)

# Loop over each combination of parameters and run both scripts
for tau in "${taus[@]}"; do
  for beta in "${betas[@]}"; do
    for delta in "${deltas[@]}"; do
      for N in "${workers[@]}"; do
        echo "Running eamsgd with tau=$tau, beta=$beta, delta=$delta, N=$N"
        mpiexec -n $N ./eamsgd $tau $beta $delta
      done
    done
  done
done

for delta in "${deltas[@]}"; do
  echo "Running msgd momentum=$delta"
  ./msgd $delta
done