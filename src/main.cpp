//#include "../include/ConvNet.hpp"
#include <mpi.h>
#include <iostream>
/* 

TODO:
  1. Write a regular backprop- of the convnet (no MPI)
  2. Store the results
  3. Need to create/ammend the cmake file for building?
  4. 
*/
int main(int argc, char* argv[]) {
  //ConvNet net;
  
  // == Hyperparameters == //
  const int input_size = 784; // mnist img-size (28^2)
  const int num_classes = 10;
  const int batch_size = 1; 
  const int num_epochs = 5; 
  const double learning_rate = 0.001;
  
  // ================ //
  //    MPI-setup     //
  // ================ // 
  int size, rank, ierr;
  ierr = MPI_Init(&argc, &argv);
  ierr = MPI_Comm_size(MPI_COMM_WORLD, &size);
  ierr = MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // testing mpi // looks like it works
  std::cout << "Hi, this is process " << rank << ", whats up? \n";
  /*
  const std::string MNIST_path = "../data/mnist/";
  auto train_dataset =
      torch::data::datasets::MNIST(MNIST_path)
          .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
          .map(torch::data::transforms::Stack<>());

  int num_train_samples = train_dataset.size().value();
  */
  ierr = MPI_Finalize();
  return 0;
}