#include "ConvNet.hpp"
#include <mpi.h>

/* 

TODO:

*/
int main() {
  ConvNet net;
  
  // == Hyperparameters == //
  const int64_t input_size = 784; // mnist size
  const int64_t num_classes = 10;
  const int64_t batch_size = 1; 
  const size_t num_epochs = 5; 
  const double learning_rate = 0.001;

  const std::string MNIST_path = "../data/mnist/";
  auto train_dataset = torch::data::datasets::MNIST(MNIST_path)
    .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
    .map(torch::data::transforms::Stack<>);
  int num_train_samples = train_dataset.size().value();

  return 0;
}