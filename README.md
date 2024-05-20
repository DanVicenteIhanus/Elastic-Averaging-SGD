____
____
# Evaluation of the Elastic Averaging Stochastic Gradient Descent algorithm #
____
____
#### Final project in the course SF2568 - Parallel Computations For Large-Scale Problems at KTH #####
*Code written By **Dan Vicente** (*danvi@kth.se*) and **Erik Lindé** (*elinde2@kth.se*) during the spring of 2024, except for the code for parsing the `CIFAR-10` dataset as a torch data object which was written by Lei Mao (https://github.com/leimao) in 2021 and modified by us in 2024.*
____

This repository contains all the scripts used in our project in SF2568 at KTH. To get all the scripts and classes to your system, we recommend that you simply clone this repository into a suitable directory on your system. Full instructions for dependancies of this project follow.

EASGD is a parallel algorithm for training Neural Networks and acts as a parallel alternative to regular SGD or momentum methods like MSGD, ADAM etc. For the paper describing the idea, algorithm etc. we refer to [1]. 

We have evaluated the performance of both the EASGD and EAMGSD algorithms introduced in [1] using CPUs communicating using the $\texttt{MPI}$ framework. We consider Convolutional Neural Networks for evaluating the algorithms and build them using $\texttt{PyTorch}$. To efficiently make use of $\texttt{MPI}$ we use $\texttt{Libtorch}$, the $\texttt{C++}$ API of $\texttt{PyTorch}$.
____

#### **STEP 1: MPI** ####
To download openMPI follow the link, download $\texttt{openMPI}$ for your system configurations.
https://www.open-mpi.org/software/ompi/v2.0/

After download has been made, you unzip/untar to a folder openmpi-2.0.x/ and run

```
> cd openmpi-2.0.x/
> ./configure --prefix=$HOME/your/path/here
> make all
> make install
> $HOME/your/path/here/mpirun --version
 mpirun (Open MPI) 2.0.x
 Report bugs to http://www.open-mpi.org/community/help/
```

____
#### **STEP 2: LIBTORCH** ####
To install Libtorch, follow the link and download the appropriate build for your system configurations. https://pytorch.org/

**NOTE:** we are using Mac (with M1) and thus the default C++/Java build for $\texttt{arm64}$. This version does not support ROCm and CUDA at the time of writing this documentation. We build Libtorch in `/usr/local/libtorch`, but you may specify the exact path in `/src/CMakeLists.txt`.

The datasets we have used are $\texttt{MNIST}$ and $\texttt{CIFAR10}$. To make our scripts work, download the datasets into `../dataset/mnist` and `../dataset/cifar-10-batches-bin` from the following links,

$\texttt{MNIST}:$ https://github.com/cvdfoundation/mnist?tab=readme-ov-file

$\texttt{CIFAR-10}:$ https://www.cs.toronto.edu/~kriz/cifar.html
____
#### **STEP 3: TRAIN THE CNN** ####
We use CMake to build/compile the scripts. At the moment of writing this documentation, $\texttt{libtorch}$ requires compilers with support for $\texttt{C++}17$. To build everything according to `../src/CMakeLists.txt`, run

```
> cd build
> cmake ../src
> cmake --build . --config Release
> cd ..
```
If everything is setup correctly, you should have executables for running the MSGD (sequential) script,  EASGD and EAMSGD (parallel) scripts for both the $\texttt{MNIST}$ and $\texttt{CIFAR-10}$ datasets. To run the training with EAMSGD, we specify the parameters in the command-line arguments in the following order $\tau$, $\beta$, $\delta$. We are using the learning rate $0.01$ for every run, but you can change this hyperparameter manually in the corresponding script.

**EXAMPLE:** Training the CNN on the  $\texttt{CIFAR-10}$ dataset using the EAMSGD training routine with 8 workers (and one root/center node) and the hyperparameters $\tau = 4$, $\alpha = 0.125$ and $\delta = 0.9$ is done by running

```
> cd src 
> mpiexec -n 9 ./eamsgd_cifar 4 0.125 0.9
```
**EXAMPLE:** Training the CNN on the $\texttt{MNIST}$ dataset using the EAMSGD training routine with 4 workers (and one root/center node) and the hyperparameters $\tau = 2$, $\alpha = 0.25$ and $\delta = 0.9$ is done by running

```
> cd src 
> mpiexec -n 5 ./eamsgd_mnist 2 0.25 0.9
```
The training using MSGD is done analogously. **EXAMPLE:** Training on $\texttt{MNIST}$ with $\delta = 0.9$ is done by running
```
> cd src
> ./msgd_cifar 0.9
```

There are some predefined experiment setups contained in bash-scripts (e.g `../src/experiments.sh`, `../src/experiment_easgd.sh` etc.)

**EXAMPLE:** Running EAMSGD and MSGD training for many different values of the hyperparameters. (Generating most of the figures in the report)

```
> cd src 
> sh ./experiments.sh
```

 **WARNING:** This bash-job can take several days to finish depending on the grid size.

## References
<a id="1">[1] Sixing Zhang, Anna Choromanska, Yann LeCun, Deep learning with Elastic Averaging SGD, https://arxiv.org/abs/1412.6651 </a> 
