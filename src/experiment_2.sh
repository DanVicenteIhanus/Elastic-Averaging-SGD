#!/bin/bash
mpiexec -n 3 ./eamsgd 2 0.25 0.9
mpiexec -n 5 ./eamsgd 8 0.25 0.9
mpiexec -n 9 ./eamsgd 4 0.25 0.9
mpiexec -n 9 ./eamsgd 8 0.25 0.9