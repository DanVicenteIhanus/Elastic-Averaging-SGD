//#include "../include/ConvNet.hpp"
#include <mpi.h>
#include <iostream>
#include <torch/torch.h>
#include "../include/convnet.h"

torch::Device device(torch::kCPU);

/* 

TODO:
  1. Asynchronous EASGD <-
    - Step 1. Fix deadlock in communication. <-
    - Step 2. Add/Subtract/Multiply parameters of model
  2.Plotting / Saving data efficiently (from root)
  3. Tune hyperparameters (elastic force etc) ? tau = {4, 16, 32}
  4. 
*/

int main(int argc, char* argv[]) {
  
  // == Hyperparameters == //
  const int num_classes = 10;
  const int batch_size = 10000; 
  const int num_epochs = 2; 
  const double rho = 0.001;
  const double lr = 0.01;
  const int tau = 4; // communication period
  
  // ================ //
  //    MPI-setup     //
  // ================ // 
  int size, rank, ierr;
  ierr = MPI_Init(&argc, &argv);
  ierr = MPI_Comm_size(MPI_COMM_WORLD, &size);
  ierr = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Status status;
  MPI_Request req1, req2;

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

  auto sz = model->named_parameters().size();
  auto param = model->named_parameters();
  
  // counting total number of elements in the model
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
      for (int k = 0; k < num_train_samples/batch_size; k++) {
        
        // testing periodic mpi communication
        std::vector<int> send_data(100);
        std::vector<int> recv_data(100);
        std::vector<MPI_Request> requests(size - 1);
        
        for (int i = 0; i < 100; i++) {
          send_data[i] = i;
        }
        for (int p = 1; p < size; p++) {
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

            // send parameters to root
            MPI_Isend(temp, flat.numel(), MPI_FLOAT, p, 0, MPI_COMM_WORLD, &req1);
            // receive from root
            MPI_Irecv(param_partner, flat.numel(), MPI_FLOAT, p, 0, MPI_COMM_WORLD, &req2);
            MPI_Wait(&req1, &status);
            MPI_Wait(&req2, &status);

            // unpack 1-D vector
            auto p_recv = (float *)calloc(
                flat.numel(), flat.numel() * param_elem_size);
            for (int j = 0; j < flat.numel(); j++) {
                *(p_recv + j) = *(param_partner + j);
            }

            torch::Tensor p_tensor =
                torch::from_blob(p_recv, dim_array, torch::kFloat)
                    .clone();
            // freeing temp arrays
            free(temp);
            free(p_recv);
          }
        }
      }
    }
    
  }
  else {
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
        optimizer.step();

        // testing periodic mpi communication
        std::vector<int> recv_data(100);
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

            // receive from root
            MPI_Irecv(param_partner, flat.numel(), MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &req2);
            // send parameters to root
            MPI_Isend(temp, flat.numel(), MPI_FLOAT, 0, 0,MPI_COMM_WORLD, &req1);
            MPI_Wait(&req1, &status);
            MPI_Wait(&req2, &status);

            // unpack 1-D vector form corresponding displacement and form
            // tensor
            auto root_recv = (float *)calloc(
                flat.numel(), flat.numel() * param_elem_size);
            // fp << "left - " << std::endl;
            for (int j = 0; j < flat.numel(); j++) {
                *(root_recv + j) = *(param_partner + j);
            }

            torch::Tensor root_tensor =
                torch::from_blob(root_recv, dim_array, torch::kFloat)
                    .clone();
            /*                    
            // average gradients
            param[i].value().data().add_(root_tensor.data());
            param[i].value().data().add_(right_tensor.data());
            param[i].value().data().div_(3);
            */ 
            // freeing temp arrays
            free(temp);
            free(root_recv);
          }
          //MPI_Recv(recv_data.data(), 100, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          //std::cout << "processor: " << rank << " received: " << recv_data[rank] << " at iteration " << t << "\n";
        }
        t ++;
      }
    
      auto sample_mean_loss = running_loss / num_train_samples;
      auto accuracy = static_cast<double>(num_correct) / num_train_samples;

      std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "], Trainset - Loss: "
          << sample_mean_loss << ", Accuracy: " << accuracy << '\n';
    }
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
  ierr = MPI_Finalize();
  return 0;
}