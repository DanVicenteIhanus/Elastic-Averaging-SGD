____
____
# Evaluation of the Elastic Averaging Stochastic Gradient Descent algorithm #
____
____
#### Final project in the course SF2568 - Parallel Computations For Large-Scale Problems at KTH #####
*Code written By **Dan Vicente** (*danvi@kth.se*) and **Erik Lindé** (*elinde2@kth.se*) during the spring of 2024, except for the code for parsing the $\texttt{CIFAR-10}$ dataset as a torch data object which was written by Lei Mao (https://github.com/leimao) in 2021 and modified by us in 2024.*
____

This repository contains all the scripts used in our project in SF2568 at KTH. To get all the scripts and classes to your system, we recommend that you simply clone this repository into a suitable directory on your system. Full instructions for dependancies of this project follow.

EASGD is a parallel algorithm for training Neural Networks and acts as a parallel alternative to regular SGD or momentum methods like MSGD, ADAM etc. For the paper describing the idea, algorithm etc. we refer to [1]. 

We have evaluated the performance of both the EASGD and EAMGSD algorithms introduced in [1] using CPUs communicating using the $\texttt{MPI}$ framework. We consider Convolutional Neural Networks for evaluating the algorithms and build them using $\texttt{pytorch}$. To efficiently make use of $\texttt{MPI}$ we use $\texttt{Libtorch}$, the $\texttt{c++}$ API of $\texttt{pytorch}$.
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
**NOTE:** we are using Mac (with M1) and thus the default C++/Java build for $\texttt{arm64}$. This version does not support ROCm and CUDA at the time of writing this documentation. We build Libtorch in $\texttt{/usr/local/libtorch}$, but you may specify the exact path in $\texttt{CMakeLists.txt}$.

The datasets we have used are $\texttt{MNIST}$ and $\texttt{CIFAR10}$. To make our scripts work, download the datasets into $\texttt{"../dataset"}$ from the following links,

$\texttt{MNIST}:$ https://github.com/cvdfoundation/mnist?tab=readme-ov-file

$\texttt{CIFAR-10}:$ https://www.cs.toronto.edu/~kriz/cifar.html

We use CMake to build/compile the scripts. At the moment of writing this documentation, $\texttt{libtorch}$ requires compilers with support for $\texttt{c++}17$. The current experiments are using $\texttt{EAMSGD\char`_cifar.cpp}$ and $\texttt{MSGD\char`_cifar.cpp}$. But you can modify the CMake file to run experiments for the $\texttt{MNIST}$ dataset as well.
____
#### **STEP 3: TRAIN THE CNN** ####
To build the projects, run the following in the terminal 
```
> cd src
> cmake --build . --config Release
```

If everything is setup correctly, you should have executables for running the MSGD (sequential) script and the EAMSGD (parallel) script. To run the training with EAMSGD, we specify the parameters in the command-line arguments in the following order $\tau$, $\beta$, $\delta$. 

**EXAMPLE:** Training the CNN using the EAMSGD training routine with 8 workers (and one root/center node) and the hyperparameters $\tau = 4$, $\beta = 2$ and $\delta = 0.9$ is done by running

```
> cd src 
> mpiexec -n 9 ./eamsgd 4 2 0.9
```
The training using MSGD is done analogously. **EXAMPLE:** with $\delta = 0.9$ you run
```
> cd src
> ./msgd 0.9
```
**NOTES:** We are using the learning rate $0.01$ for every run, but you can change this hyperparameter manually in the corresponding script.

If you prefer to run all the experiments you can run 

```
> cd src 
> sh ./experiments.sh
```
In $\texttt{./src/experiments.sh}$, we run EAMSGD and MSGD training for many different values of the hyperparameters. **WARNING:** This bash-job can take several days to finish depending on the grid size

## References
<a id="1">[1] Sixing Zhang, Anna Choromanska, Yann LeCun, Deep learning with Elastic Averaging SGD, https://arxiv.org/abs/1412.6651 </a> 
