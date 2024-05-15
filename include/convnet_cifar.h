#ifndef CONVNET_H
#define CONVNET_H

#include <torch/torch.h>
#include <iostream>

class ConvNetImpl : public torch::nn::Module {
public:
    explicit ConvNetImpl(int64_t num_classes = 10, int64_t num_channels = 3);
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Sequential conv1;
    torch::nn::Sequential conv2;
    torch::nn::Sequential fc1;
    torch::nn::Sequential fc2;
};

TORCH_MODULE(ConvNet);

#endif // CONVNET_H