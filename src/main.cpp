//#include "../include/ConvNet.hpp"
#include <mpi.h>
#include <iostream>
#include <torch/torch.h>
#include "../include/convnet.h"

torch::Device device(torch::kCPU);

/* 

TODO:
  1. Use the MNIST dataset to define CNN <- OK?
  2. backprop (sequentially) <- 
  3. Store the results
  4. Print classification error?
*/
int main(int argc, char* argv[]) {
  // == Hyperparameters == //
  const int num_classes = 10;
  const int batch_size = 1; 
  const int num_epochs = 10; 
  const double lr = 0.001;
  
  // ================ //
  //    MPI-setup     //
  // ================ // 
  int size, rank, ierr;
  /*
  ierr = MPI_Init(&argc, &argv);
  ierr = MPI_Comm_size(MPI_COMM_WORLD, &size);
  ierr = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  // testing MPI
  std::cout << "Hi, this is process " << rank << ", whats up? \n";
  */

  // MNIST data from pytorch datasets
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
  
  // create training and test data
  auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
    std::move(train_dataset), batch_size);
  
  auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(test_dataset), batch_size);
  
  // create CNN
  ConvNet model(num_classes);
  model->to(device);
  
  // define optimizer
  torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(lr));

  // ============== //
  // TRAINING PHASE //
  // ============== //

  // Training
  std::cout << std::fixed << std::setprecision(4);
  std::cout << "Training CNN...\n";

  for (int epoch = 0; epoch < num_epochs; epoch++) {
    double running_loss = 0.0;
    size_t num_correct = 0;
    int bar_width = 70;
    for (auto& batch : *train_loader) {
      auto data = batch.data.to(device);
      auto target = batch.target.to(device);

      // forward pass
      auto output = model->forward(data);
      
      // compute + update loss
      auto loss = torch::nn::functional::cross_entropy(output, target);
      running_loss += loss.item<double>() * data.size(0);

      // predict
      auto prediction = output.argmax(1);

      // update # correctly classified samples
      num_correct += prediction.eq(target).sum().item<int64_t>();

      // backprop + gradient step
      optimizer.zero_grad();
      loss.backward();
      optimizer.step();
    }
    auto sample_mean_loss = running_loss / num_train_samples;
    auto accuracy = static_cast<double>(num_correct) / num_train_samples;

    std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "], Trainset - Loss: "
        << sample_mean_loss << ", Accuracy: " << accuracy << '\n';
  }

  // ============= //
  // TESTING PHASE //
  // ============= //

  std::cout << "Training finished!\n\n";
  std::cout << "Testing...\n";
  // Test the model
  model->eval();
  torch::InferenceMode no_grad;

  double running_loss = 0.0;
  size_t num_correct = 0;

  for (const auto& batch : *test_loader) {
    auto data = batch.data.to(device);
    auto target = batch.target.to(device);

    auto output = model->forward(data);

    auto loss = torch::nn::functional::cross_entropy(output, target);
    running_loss += loss.item<double>() * data.size(0);

    auto prediction = output.argmax(1);
    num_correct += prediction.eq(target).sum().item<int64_t>();
  }

  std::cout << "Testing finished!\n";

  auto test_accuracy = static_cast<double>(num_correct) / num_test_samples;
  auto test_sample_mean_loss = running_loss / num_test_samples;

  std::cout << "Testset - Loss: " << test_sample_mean_loss << ", Accuracy: " << test_accuracy << '\n';
  //ierr = MPI_Finalize();
  return 0;
}