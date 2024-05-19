#include <mpi.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <torch/torch.h>
#include "../include/convnet.h"
#include "../include/cifar10.h"
#include <chrono>
using namespace std::chrono;

torch::Device device(torch::kCPU);

std::fstream setup_result_file(int size, int rank, int tau, double alpha) {
    std::ostringstream filename;
    filename << "../results/cifar/easgd/stats_cifar_EASGD_size" << size << "_rank_" << rank
             << "_tau_" << tau << "_alpha_" << alpha << ".txt";
    
    // Open file for writing
    std::fstream file;
    file.open(filename.str(), std::fstream::out | std::fstream::app);

    file.seekg(0, std::ios::end); 
    if (file.tellg() == 0) {
        if (rank == 0) {
            file << "Duration,Accuracy,Sample_Mean_Loss,Testing_accuracy,Testing_Mean_Loss\n"; // write the header
        } else {
            file << "Duration,Accuracy,Sample_Mean_loss\n";
        }
    }

    return file;
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <tau> <alpha>" << std::endl;
    return 1;
  }
  // ================ //
  //    MPI-setup     //
  // ================ // 
  int size, rank, ierr;
  ierr = MPI_Init(&argc, &argv);
  ierr = MPI_Comm_size(MPI_COMM_WORLD, &size);
  ierr = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Status statuses[2];
  MPI_Request reqs[2];

  // == Hyperparameters == //
  const int num_classes = 10;
  const int batch_size = 1000; 
  const int num_epochs = 15; 
  const double lr = 0.01;
  const int tau = std::stoi(argv[1]); // communication period
  const double alpha = std::stod(argv[2]);
  
  // allocate mem for stats
  double comm_time;
  double total_comm_time = 0;
  double test_accuracy;
  double test_sample_mean_loss;

  // timing the training  
  auto start = high_resolution_clock::now();

  // setup file for results
  std::fstream file = setup_result_file(size, rank, tau, alpha);

  // ============ //
  // CIFAR10 data //
  // ============ //
  const std::string dataset_root{"../dataset/cifar-10-batches-bin"};
  CIFAR10 train_set{dataset_root, CIFAR10::Mode::kTrain};
  CIFAR10 test_set{dataset_root, CIFAR10::Mode::kTest};

  auto train_dataset = train_set
      .map(torch::data::transforms::Normalize<>({0.4914, 0.4822, 0.4465}, {0.2023, 0.1994, 0.2010}))
      .map(torch::data::transforms::Stack<>());

  auto test_dataset = test_set
      .map(torch::data::transforms::Normalize<>({0.4914, 0.4822, 0.4465}, {0.2023, 0.1994, 0.2010}))
      .map(torch::data::transforms::Stack<>());

  int num_train_samples = train_dataset.size().value(); 
  auto num_test_samples = test_dataset.size().value(); 
  std::cout << "number of training samples = "<< num_train_samples << "\n";
  std::cout << "number of testing samples = " << num_test_samples << "\n";

  auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
      std::move(train_dataset), torch::data::DataLoaderOptions().batch_size(batch_size));

  auto test_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
      std::move(test_dataset), torch::data::DataLoaderOptions().batch_size(batch_size));
  
  // create CNN
  ConvNet model(num_classes, 3);
  model->to(device);
  
  // define optimizer
  torch::optim::SGDOptions options(lr);
  torch::optim::SGD optimizer(model->parameters(), options);

  // get model-size and define array of parameters  
  auto sz = model->named_parameters().size();
  auto param = model->named_parameters();
  int num_elem_param = 0;
  for (int i = 0; i < sz; i++) {
      num_elem_param += param[i].value().numel();
  }
  if (rank == 0) {
      std::cout << "Number of parameters - " << sz << std::endl;
      std::cout << "Number of elements - " << num_elem_param << std::endl;
  }
  float param_partner[num_elem_param];
  
  auto param_elem_size = param[0].value().element_size();
  for (int i = 0; i < num_elem_param; i++) {
      param_partner[i] = 0.0;
  }

  // ============== //
  // TRAINING PHASE //
  // ============== //
  int t = 0;
  if (rank == 0) {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "--------------------------------------------------------\n";
    std::cout << "Training CNN... num_epochs = " << num_epochs << ", batch_size = " << batch_size <<"\n";
    std::cout << "--------------------------------------------------------\n";
    for (int epoch = 0; epoch < num_epochs; epoch++) {
      double running_loss = 0.0;
      size_t num_correct = 0;
      for (auto &batch : *train_loader) {
        
        model->train();
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
        if (t % tau == 0) {
          for (int p = 1; p < size; p++) {
            for (auto i = 0; i < sz; i++) {
              int num_dim = param[i].value().dim();
              std::vector<int64_t> dim_array;
              for (int j = 0; j < num_dim; j++) {
                  dim_array.push_back(param[i].value().size(j));
              }

              // flattening the tensor and copying it to a 1-D vector
              auto flat = torch::flatten(param[i].value());
              auto temp = (float *)calloc(flat.numel(),
                                          flat.numel() * param_elem_size);
              for (int j = 0; j < flat.numel(); j++) {
                  *(temp + j) = flat[j].item<float>();
              }

              // send parameters to root
              MPI_Isend(temp, flat.numel(), MPI_FLOAT, p, 0 , MPI_COMM_WORLD, &reqs[0]);
              MPI_Irecv(param_partner, flat.numel(), MPI_FLOAT, p, 0, MPI_COMM_WORLD, &reqs[1]);
              MPI_Waitall(2, reqs, statuses);

              // unpack 1-D vector
              auto p_recv = (float *)calloc(
                  flat.numel(), flat.numel() * param_elem_size);
              for (int j = 0; j < flat.numel(); j++) {
                  *(p_recv + j) = *(param_partner + j);
              }

              torch::Tensor x_temp = torch::from_blob(p_recv, dim_array, torch::kFloat).clone();              
              x_temp.data().subtract_(param[i].value().data()); // x_temp - \tilde{x}
              x_temp.data().multiply_(alpha);                   // alpha(x_temp - \tilde{x})
              param[i].value().data().add_(x_temp.data());      // x = x + alpha(x_i - \tilde{x})

              // freeing temp arrays
              free(temp);
              free(p_recv);
              }
            }
        }
      t++;
      }
      // print epoch results in terminal
      auto sample_mean_loss = running_loss / num_train_samples;
      auto accuracy = static_cast<double>(num_correct) / num_train_samples;
      std::cout << "RANK: "<< rank << " Epoch [" << (epoch + 1) << "/" << num_epochs << "], Trainset - Loss: "
          << sample_mean_loss << ", Accuracy: " << accuracy << '\n';
      auto stop = high_resolution_clock::now();
      auto duration = duration_cast<milliseconds>(stop - start);

      // Test the model
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
    // Log to file in txt
    file << duration.count() << "," << accuracy << "," << sample_mean_loss << "," << test_accuracy << "," << test_sample_mean_loss <<"\n";
    }
  } else {
  // non-root processes
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
        
        if (t % tau == 0) {
          for (auto i = 0; i < sz; i++) {
            int num_dim = param[i].value().dim();
            std::vector<int64_t> dim_array;
            for (int j = 0; j < num_dim; j++) {
                dim_array.push_back(param[i].value().size(j));
            }

            // flattening the tensor and copying it to a 1-D vector
            auto flat = torch::flatten(param[i].value());

            auto temp = (float *)calloc(flat.numel(),
                                        flat.numel() * param_elem_size);
            for (int j = 0; j < flat.numel(); j++) {
                *(temp + j) = flat[j].item<float>();
            }

            // receive from communication partner
            MPI_Irecv(param_partner, flat.numel(), MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &reqs[1]);
            MPI_Isend(temp, flat.numel(), MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &reqs[0]);
            MPI_Waitall(2, reqs, statuses);
            
            // unpack 1-D vector and form tensor
            auto root_recv = (float *)calloc(
                flat.numel(), flat.numel() * param_elem_size);
            for (int j = 0; j < flat.numel(); j++) {
                *(root_recv + j) = *(param_partner + j);
            }
            torch::Tensor x_temp =
                torch::from_blob(root_recv, dim_array, torch::kFloat).clone();
            // x  = x + alpha*(x_i - x)
            x_temp.data().subtract_(param[i].value().data()); // x_temp - \tilde{x}
            x_temp.data().multiply_(alpha); 
            param[i].value().data().add_(x_temp.data()); // update tensor parameters

            // freeing temp arrays
            free(temp);
            free(root_recv);
          }
        } 
        optimizer.step();
        t ++;
      }

      auto sample_mean_loss = running_loss / num_train_samples;
      auto accuracy = static_cast<double>(num_correct) / num_train_samples;
      std::cout << "RANK: "<< rank << " Epoch [" << (epoch + 1) << "/" << num_epochs << "], Trainset - Loss: "
          << sample_mean_loss << ", Accuracy: " << accuracy << '\n';
      auto stop = high_resolution_clock::now();
      auto duration = duration_cast<milliseconds>(stop - start);
      file << duration.count() << "," << accuracy << "," << sample_mean_loss << "\n";

    }
  }  
  file.close();
  ierr = MPI_Finalize();
  return 0;
}