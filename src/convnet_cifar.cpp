/*
CNN with 3 conv-layers and 1 pooling layer.
Forward pass is implemented here and the backprop is implemented in "main.cpp"
*/
#include "../include/convnet.h"
#include <torch/torch.h>

ConvNetImpl::ConvNetImpl(int64_t num_classes, int64_t num_channels) {
  conv1 = torch::nn::Sequential(
    torch::nn::Conv2d(torch::nn::Conv2dOptions(num_channels, 30, 5).stride(1)),
    torch::nn::ReLU(),
    torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
  );

  conv2 = torch::nn::Sequential(
    torch::nn::Conv2d(torch::nn::Conv2dOptions(30, 40, 5).stride(1)),
    torch::nn::ReLU(),
    torch::nn::BatchNorm2d(40),
    torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
  );

  // Correcting the input size of the fully connected layer
  int64_t conv_output_size = 40 * 5 * 5; // Correct output size after conv layers

  fc1 = torch::nn::Sequential(
    torch::nn::Linear(conv_output_size, 700),
    torch::nn::ReLU()
    // torch::nn::Dropout2d(torch::nn::Dropout2dOptions(0.3)) // too much dropout?
  );

  fc2 = torch::nn::Sequential(
    torch::nn::Linear(700, num_classes)
  );

  register_module("conv1", conv1);
  register_module("conv2", conv2);
  register_module("fc1", fc1);
  register_module("fc2", fc2);
}

torch::Tensor ConvNetImpl::forward(torch::Tensor x) {
  x = conv1->forward(x);
  x = conv2->forward(x);
  x = x.view({x.size(0), 40 * 5 * 5}); 
  x = fc1->forward(x);
  return fc2->forward(x);
}