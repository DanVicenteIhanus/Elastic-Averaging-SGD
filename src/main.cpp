//#include "../include/ConvNet.hpp"
#include <mpi.h>
#include <iostream>
#include <torch/torch.h>
#include "../include/convnet.h"

torch::Device device(torch::kCPU);

/* 

TODO:
  1. Asynchronous EASGD <-
  2. Tune hyperparameters (elastic force etc)
  3. 
*/

int main(int argc, char* argv[]) {
  // == Hyperparameters == //
  const int num_classes = 10;
  const int batch_size = 100; 
  const int num_epochs = 50; 
  const double rho = 0.001;
  const double lr = 0.01;
  
  // ================ //
  //    MPI-setup     //
  // ================ // 
  int size, rank, ierr;
  /*
  ierr = MPI_Init(&argc, &argv);
  ierr = MPI_Comm_size(MPI_COMM_WORLD, &size);p
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
  
  auto test_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(test_dataset), batch_size);
  
  // create CNN
  ConvNet model(num_classes);
  model->to(device);
  
  // define optimizer
  torch::optim::SGDOptions options(lr);
  //options.weight_decay(rho);

  torch::optim::SGD optimizer(model->parameters(), options);

  // ============== //
  // TRAINING PHASE //
  // ============== //
  std::cout << std::fixed << std::setprecision(4);
  std::cout << "--------------------------------------------------------\n";
  std::cout << "Training CNN... num_epochs = " << num_epochs << ", batch_size = " << batch_size <<"\n";
  std::cout << "--------------------------------------------------------\n";
  for (int epoch = 0; epoch < num_epochs; epoch++) {
    double running_loss = 0.0;
    size_t num_correct = 0;
    for (auto& batch : *train_loader) {
      auto data = batch.data.to(device);
      auto target = batch.target.to(device);
    
      // forward pass
      auto output = model->forward(data);
      
      // compute loss
      auto loss = torch::nn::functional::cross_entropy(output, target);
      running_loss += loss.item<double>() * data.size(0);

      // predict
      auto prediction = output.argmax(1);

      // update # correctly classified samples
      num_correct += prediction.eq(target).sum().item<int64_t>(); //np.sum(prediction == target)

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
  std::cout << "--------------------------------------------------------\n";
  std::cout << "Testing... num_test_samples = " << num_test_samples << "\n";
  std::cout << "--------------------------------------------------------\n";

  // Test the model
  model->eval();
  torch::InferenceMode no_grad;

  double test_running_loss = 0.0;
  size_t test_num_correct = 0;

  for (const auto& batch : *test_loader) {
    auto data = batch.data.to(device);
    auto target = batch.target.to(device);

    // forward pass
    auto output = model->forward(data);

    // loss update
    auto loss = torch::nn::functional::cross_entropy(output, target);
    test_running_loss += loss.item<double>() * data.size(0);

    // prediction & accuracy
    auto prediction = output.argmax(1);
    test_num_correct += prediction.eq(target).sum().item<int64_t>();
  
  }

  std::cout << "Testing finished!\n";

  auto test_accuracy = static_cast<double>(test_num_correct) / num_test_samples;
  auto test_sample_mean_loss = test_running_loss / num_test_samples;

  std::cout << "Testset - Loss: " << test_sample_mean_loss << ", Accuracy: " << test_accuracy << '\n';
  //ierr = MPI_Finalize();
  return 0;
}