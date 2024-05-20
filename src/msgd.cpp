#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <torch/torch.h>
#include "../include/convnet.h"
#include <chrono>
using namespace std::chrono;

torch::Device device(torch::kCPU);

std::fstream setup_result_file(double delta) {
    std::ostringstream filename;
    filename << "../results/mnist/msgd/stats_mnist_MSGD_delta_" << delta << ".txt";
    
    // Open file for writing
    std::fstream file;
    file.open(filename.str(), std::fstream::out | std::fstream::app);
    
    // Check if the file is new to write the header
    file.seekg(0, std::ios::end); // go to the end of file
    if (file.tellg() == 0) { // if file size is 0, it's new
      file << "Duration,Accuracy,Sample_Mean_Loss,Test_Accuracy,Test_Mean_loss\n"; // write the header
    }
    return file;
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << "<delta>" << std::endl;
    return 1;
  }
  // == Hyperparameters == //
  const int num_classes = 10;
  const int batch_size = 1000; 
  const int num_epochs = 10; 
  const double lr = 0.01;
  const double delta = std::stod(argv[1]);

  // allocate mem for stats
  double test_accuracy;
  double test_sample_mean_loss;
  double sample_mean_loss;
  double accuracy;
  auto start = high_resolution_clock::now(); // timing the training

  // Setup file for results
  std::fstream file = setup_result_file(delta);

  // MNIST data from pytorch datasets
  const std::string MNIST_path = "../dataset/mnist/";
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
  options.momentum(delta);
  torch::optim::SGD optimizer(model->parameters(), options);

  // get model-size and define array of parameters  
  auto sz = model->named_parameters().size();
  auto param = model->named_parameters();
  int num_elem_param = 0;
  for (int i = 0; i < sz; i++) {
      num_elem_param += param[i].value().numel();
  }
  std::cout << "Number of parameters - " << sz << std::endl;
  std::cout << "Number of elements - " << num_elem_param << std::endl;
  float param_partner[num_elem_param];
  
  auto param_elem_size = param[0].value().element_size();
  
  // initializing parameters
  for (int i = 0; i < num_elem_param; i++) {
      param_partner[i] = 0.0;
  }

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
		for (auto &batch : *train_loader) {

			auto data = batch.data.to(device);
			auto target = batch.target.to(device);

			// forward pass
			auto output = model->forward(data);
			
			// compute loss
			auto loss = torch::nn::functional::cross_entropy(output, target);
			running_loss += loss.item<double>() * data.size(0);

			// predict
			auto prediction = output.argmax(1);
      
      optimizer.zero_grad();
      loss.backward();
      optimizer.step();
		  num_correct += prediction.eq(target).sum().item<int64_t>(); //np.sum(prediction == target)
		} // batch loop

    // print epoch results in terminal
    sample_mean_loss = running_loss / num_train_samples;
    accuracy = static_cast<double>(num_correct) / num_train_samples;
    std::cout << " Epoch [" << (epoch + 1) << "/" << num_epochs << "], Trainset - Loss: "
        << sample_mean_loss << ", Accuracy: " << accuracy << '\n';
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    std::cout << "duration = " << duration.count() << "\n";

    // ============= //
    // == Testing == //
    // ============= //
    model->eval();
    double test_running_loss = 0.0;
    size_t test_num_correct = 0;
    { 
      torch::InferenceMode no_grad;
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
      test_accuracy = static_cast<double>(test_num_correct) / num_test_samples;
      test_sample_mean_loss = test_running_loss / num_test_samples;
      std::cout << "Testset - Loss: " << test_sample_mean_loss << ", Accuracy: " << test_accuracy << '\n';
    }
    file << duration.count() << "," << accuracy << "," << sample_mean_loss << "," << test_accuracy << "," << test_sample_mean_loss <<"\n";
  } 
  file.close();
  return 0;
}