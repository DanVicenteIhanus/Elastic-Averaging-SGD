//#include "../include/ConvNet.hpp"
#include <mpi.h>
#include <iostream>
#include <torch/torch.h>
#include "../include/convnet.h"

torch::Device device(torch::kCPU);

/* 

TODO:
  1. MNIST training + test sets <- OK
  2. Use the MNIST dataset to define CNN <- dimensions ??
  3. backprop (sequentially)
  4. Store the results
  5. Print classification error?
*/
int main(int argc, char* argv[]) {
  // == Hyperparameters == //
  const int input_size = 784; // mnist img-size (28^2)
  const int hidden_size = 100; // hidden-size (not needed for CNN?)
  const int num_classes = 10;
  const int batch_size = 1; 
  const int num_epochs = 100; 
  const double lr = 0.001;
  
  // ================ //
  //    MPI-setup     //
  // ================ // 
  int size, rank, ierr;
  ierr = MPI_Init(&argc, &argv);
  ierr = MPI_Comm_size(MPI_COMM_WORLD, &size);
  ierr = MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // testing MPI
  std::cout << "Hi, this is process " << rank << ", whats up? \n";
  
  //MNIST data from pytorch datasets
  const std::string MNIST_path = "../data/mnist/";
  auto train_dataset =
    torch::data::datasets::MNIST(MNIST_path)
      .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
      .map(torch::data::transforms::Stack<>());
  auto test_dataset = 
    torch::data::datasets::MNIST(MNIST_path, torch::data::datasets::MNIST::Mode::kTest)
    .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
    .map(torch::data::transforms::Stack<>());

  int num_train_samples = train_dataset.size().value(); // 60,000 samples
  auto num_test_samples = test_dataset.size().value();  // 10,000 samples
  
  auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
    std::move(train_dataset), batch_size);
  
  ConvNet model(num_classes);
  model->to(device);
  torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(lr));

  for (int epoch = 0; epoch < num_epochs; epoch++) {
    for (auto& batch : *train_loader) {
      auto data = batch.data.view({batch_size, -1}).to(device);
      auto target = batch.target.to(device);

      // forward pass
      auto output = model->forward(data);
      auto loss = torch::nn::functional::cross_entropy(output, target);

      auto prediction = output.argmax(1);

      optimizer.zero_grad();
      loss.backward();
      optimizer.step();
    }
  }

  ierr = MPI_Finalize();
  return 0;
}