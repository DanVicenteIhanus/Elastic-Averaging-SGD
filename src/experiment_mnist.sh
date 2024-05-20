#!/bin/bash
mpiexec -n 5 ./eamsgd_mnist 4 0.25 0.9
mpiexec -n 5 ./easgd_mnist 4 0.25
./msgd_mnist 0.9
./msgd_mnist 0.0