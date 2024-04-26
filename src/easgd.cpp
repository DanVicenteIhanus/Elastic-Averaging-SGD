#include <mpi.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <torch/torch.h>
#include "../include/convnet.h"
#include <chrono>
using namespace std::chrono;

torch::Device device(torch::kCPU);

/* 
TODO:
  1. Build nonparallell training
  1. Tune hyperparameters (elastic force etc) ? tau = {4, 16, 32}
  2. Grid-search to generate more data
  3. Refactor code..
*/

int main(int argc, char* argv[]) {
  
  // == Hyperparameters == //
  const int num_classes = 10;
  const int batch_size = 1000; 
  const int num_epochs = 20; 
  const double lr = 0.01;

  const int tau = 16; // communication period
  const double beta = 4;
  
  auto start = high_resolution_clock::now(); // timing the training
  
  // ================ //
  //    MPI-setup     //
  // ================ // 
  int size, rank, ierr;
  ierr = MPI_Init(&argc, &argv);
  ierr = MPI_Comm_size(MPI_COMM_WORLD, &size);
  ierr = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Status statuses[2];
  MPI_Request reqs[2];
  

  // ====================== //
  // Setup file for results
  // ====================== //
  std::ostringstream filename;
  filename << "../data/training_stats_size" <<  size << "_rank_" << rank << "_tau_" << tau << "_beta_" << beta << ".txt";
  
  // Open file for writing
  std::fstream file;
  file.open(filename.str(), std::fstream::out | std::fstream::app);
  
  // Check if the file is new to write the header
  file.seekg(0, std::ios::end); // go to the end of file
  if (file.tellg() == 0) { // if file size is 0, it's new
    file << "Duration,Accuracy,Sample_Mean_Loss\n"; // write the header
  }

  // elastic hyperparameter
  //const float alpha = beta/(tau*(size - 1)); // depends on beta, tau (for stability)
  const float alpha = 0.3;
  
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
  // initializing left and right params
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
         // complexity = O(epoch*num_batches*P*network_size)
        // getting dimensions of tensor
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

              // receive from partner
              MPI_Waitall(2, reqs, statuses);

              // unpack 1-D vector
              auto p_recv = (float *)calloc(
                  flat.numel(), flat.numel() * param_elem_size);
              for (int j = 0; j < flat.numel(); j++) {
                  *(p_recv + j) = *(param_partner + j);
              }

              torch::Tensor x_temp =
                  torch::from_blob(p_recv, dim_array, torch::kFloat).clone();
              
              // x  = x + alpha*(x_i - x)
              x_temp.data().subtract_(param[i].value().data()); // x_temp - \tilde{x}
              x_temp.data().multiply_(alpha); 

              param[i].value().data().add_(x_temp.data()); // update tensor parameters

              // freeing temp arrays
              free(temp);
              free(p_recv);
              }
            } // parameter loop
        } // process loop
      t++;
      } // batch loop

    // print epoch results in terminal
    auto sample_mean_loss = running_loss / num_train_samples;
    auto accuracy = static_cast<double>(num_correct) / num_train_samples;
    std::cout << "RANK: "<< rank << " Epoch [" << (epoch + 1) << "/" << num_epochs << "], Trainset - Loss: "
        << sample_mean_loss << ", Accuracy: " << accuracy << '\n';
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    std::cout << "duration = " << duration.count() << "\n";

    // Log to file in txt
    file << duration.count() << "," << accuracy << "," << sample_mean_loss << "\n";

    } // epoch loop
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
            // getting dimensions of tensor

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
            //MPI_Irecv(param_partner, flat.numel(), MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &req2);
            // //send parameters to root
            // MPI_Isend(temp, flat.numel(), MPI_FLOAT, 0, 0,MPI_COMM_WORLD, &req1);
            MPI_Waitall(2, reqs, statuses);
            // unpack 1-D vector form corresponding displacement and form
            // tensor
            auto root_recv = (float *)calloc(
                flat.numel(), flat.numel() * param_elem_size);
            // fp << "left - " << std::endl;
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
          } // parameter loop
        } // tau loop
        optimizer.step();
        t ++;
      } // batch loop

      auto sample_mean_loss = running_loss / num_train_samples;
      auto accuracy = static_cast<double>(num_correct) / num_train_samples;
      std::cout << "RANK: "<< rank << " Epoch [" << (epoch + 1) << "/" << num_epochs << "], Trainset - Loss: "
          << sample_mean_loss << ", Accuracy: " << accuracy << '\n';
      auto stop = high_resolution_clock::now();
      auto duration = duration_cast<milliseconds>(stop - start);
      file << duration.count() << "," << accuracy << "," << sample_mean_loss << "\n";

    } // epoch loop
  }  
  // ============= //
  // TESTING PHASE //
  // ============= //
  if (rank == 0 ) {
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
  }
  file.close();
  ierr = MPI_Finalize();
  return 0;
}